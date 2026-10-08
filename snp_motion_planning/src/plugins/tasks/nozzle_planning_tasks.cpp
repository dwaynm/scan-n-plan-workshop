// SNP-specific seed validation and freespace endpoint handling.
#include "../../level_axis_profile.h"
#include <tesseract_command_language/composite_instruction.h>
#include <tesseract_command_language/cartesian_waypoint.h>
#include <tesseract_command_language/joint_waypoint.h>
#include <tesseract_command_language/state_waypoint.h>
#include <tesseract_motion_planners/trajopt/profile/trajopt_default_plan_profile.h>
#include <tesseract_task_composer/planning/profiles/contact_check_profile.h>
#include <tesseract_command_language/profile_dictionary.h>
#include <tesseract_command_language/utils.h>
#include <tesseract_motion_planners/planner_utils.h>
#include <tesseract_motion_planners/descartes/descartes_motion_planner.h>
#include <tesseract_motion_planners/descartes/descartes_collision_edge_evaluator.h>
#include <tesseract_motion_planners/descartes/profile/descartes_profile.h>
#include <tesseract_task_composer/core/task_composer_context.h>
#include <tesseract_task_composer/core/task_composer_data_storage.h>
#include <tesseract_task_composer/core/task_composer_task.h>
#include <tesseract_task_composer/core/task_composer_task_plugin_factory.h>
#include <tesseract_task_composer/core/task_composer_plugin_factory_utils.h>
#include <console_bridge/console.h>
#include <boost/serialization/base_object.hpp>
#include <tesseract_common/serialization.h>

namespace snp_motion_planning
{
using namespace tesseract_planning;

// Keep the optimized raster configuration, including its free nozzle roll.
// A Cartesian waypoint with a seed is still Cartesian to OMPL: it recomputes
// fixed-orientation IK and can discard the only collision-free branch.
class PinFreespaceEndpointsTask : public TaskComposerTask
{
public:
  static TaskComposerNodePorts ports()
  {
    TaskComposerNodePorts p;
    for (const auto& key : {"program", "environment", "profiles"})
      p.input_required[key] = TaskComposerNodePorts::SINGLE;
    p.output_required["program"] = TaskComposerNodePorts::SINGLE;
    return p;
  }
  PinFreespaceEndpointsTask() : TaskComposerTask("PinFreespaceEndpointsTask", ports(), true) {}
  PinFreespaceEndpointsTask(std::string name, const YAML::Node& config, const TaskComposerPluginFactory&)
    : TaskComposerTask(std::move(name), ports(), config) { validatePorts(); }
protected:
  TaskComposerNodeInfo runImpl(TaskComposerContext& context, OptionalTaskComposerExecutor) const override
  {
    TaskComposerNodeInfo info(*this);
    auto program = getData(*context.data_storage, "program").as<CompositeInstruction>();
    for (auto* move : {program.getFirstMoveInstruction(), program.getLastMoveInstruction()})
    {
      if (!move) throw std::runtime_error("Empty freespace program");
      const auto& wp = move->getWaypoint();
      auto q = getJointPosition(wp);
      auto names = getJointNames(wp);
      if (q.size() == 0 || !q.allFinite() || q.size() != static_cast<Eigen::Index>(names.size()))
        throw std::runtime_error("Freespace endpoint has no valid joint seed");
      move->getWaypoint() = JointWaypoint(names, q); // constrained, zero tolerance
    }
    setData(*context.data_storage, "program", program);
    info.return_value = info.status_code = 1;
    info.color = "green";
    info.status_message = "Pinned freespace endpoints to raster joint states";
    return info;
  }
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive& ar, const unsigned int)
  { ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(TaskComposerTask); }
};

// Refine only LINEAR Cartesian edges; keep composite/strip boundaries and all
// original waypoints so coverage and transfer identity survive a retry.
static void refineRaster(CompositeInstruction& program, double spacing, std::optional<Eigen::Isometry3d>& previous)
{
  auto refined = program;
  refined.clear();
  for (auto& instruction : program)
  {
    if (instruction.isCompositeInstruction())
    {
      auto child = instruction.as<CompositeInstruction>();
      refineRaster(child, spacing, previous);
      refined.push_back(child);
      continue;
    }
    if (instruction.isMoveInstruction())
    {
      auto& move = instruction.as<MoveInstructionPoly>();
      if (move.getWaypoint().isCartesianWaypoint())
      {
        const Eigen::Isometry3d target = move.getWaypoint().as<CartesianWaypointPoly>().getTransform();
        if (previous && move.getMoveType() == MoveInstructionType::LINEAR)
        {
          int count = std::max(1, static_cast<int>(std::ceil((target.translation()-previous->translation()).norm()/spacing)));
          for (int i = 1; i < count; ++i)
          {
            double f = static_cast<double>(i)/count;
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose.translation() = (1-f)*previous->translation() + f*target.translation();
            pose.linear() = Eigen::Quaterniond(previous->linear()).slerp(f, Eigen::Quaterniond(target.linear())).toRotationMatrix();
            auto child = move.createChild();
            child.getWaypoint() = CartesianWaypoint(pose);
            refined.push_back(child);
          }
        }
        previous = target;
      }
      else previous.reset();
    }
    refined.push_back(instruction);
  }
  program = std::move(refined);
}

class CheckedDescartesTask : public PinFreespaceEndpointsTask
{
public:
  CheckedDescartesTask() = default;
  using PinFreespaceEndpointsTask::PinFreespaceEndpointsTask;
protected:
  TaskComposerNodeInfo runImpl(TaskComposerContext& context, OptionalTaskComposerExecutor) const override
  {
    TaskComposerNodeInfo info(*this);
    info.return_value = info.status_code = 0;
    auto program = getData(*context.data_storage, "program").as<CompositeInstruction>();
    auto env = getData(*context.data_storage, "environment").as<std::shared_ptr<const tesseract_environment::Environment>>();
    auto profiles = getData(*context.data_storage, "profiles").as<std::shared_ptr<ProfileDictionary>>();
    const std::string ns = "DescartesMotionPlannerTask";
    const auto profile_name = program.getProfile(ns);
    auto original = std::dynamic_pointer_cast<const LevelAxisDescartesPlanProfile<float>>(
        profiles->getProfile(DescartesPlanProfile<float>::getStaticKey(), ns, profile_name));
    if (!original) throw std::runtime_error("Checked Descartes requires the SNP seed profile");
    auto profile = std::make_shared<LevelAxisDescartesPlanProfile<float>>(*original);
    profile->edge_cache = std::make_shared<SeedEdgeCache>();
    profile->sample_cache = std::make_shared<SeedSampleCache<float>>();
    auto local_profiles = std::make_shared<ProfileDictionary>();
    local_profiles->addProfile(ns, profile_name, profile);
    local_profiles->addProfile(ns, profile_name,
        profiles->getProfile(DescartesSolverProfile<float>::getStaticKey(), ns, profile_name));
    PlannerRequest request;
    request.instructions = program;
    request.env = env;
    request.profiles = local_profiles;
    request.format_result_as_input = true;
    auto manip = env->getJointGroup(program.getManipulatorInfo().manipulator);
    const auto seed_limits = seedJointLimits(manip->getLimits().joint_limits, manip->getJointNames(), profile->seed_joint_bounds);
    const auto anchor = env->getCurrentJointValues(manip->getJointNames());
    if ((anchor.array() < seed_limits.col(0).array()).any() || (anchor.array() > seed_limits.col(1).array()).any())
      throw std::runtime_error("Planning anchor is outside the configured connected seed joint range");
    auto collision_config = profile->edge_collision_check_config;
    collision_config.type = tesseract_collision::CollisionEvaluatorType::LVS_DISCRETE;
    collision_config.longest_valid_segment_length = std::min(collision_config.longest_valid_segment_length, M_PI/180.0);
    DescartesCollisionEdgeEvaluator<float> check(*env, manip, collision_config, false);
    DescartesMotionPlanner<float> planner(ns);
    // Preserve the global nominal search when it works. If it fails, recover
    // individual rasters: free transfers will be solved after raster optimization
    // anyway. This localizes both expensive angular IK and failure reporting.
    bool missing_vertices = false;
    auto search = [&](const CompositeInstruction& input, CompositeInstruction& output) {
      missing_vertices = false;
      request.instructions = input;
      for (int attempt = 0; attempt < profile->seed_attempts; ++attempt)
      {
        if (context.isAborted()) { info.status_message = "Seed search aborted"; return false; }
        auto response = planner.solve(request);
        if (!response)
        {
          info.status_message = "Raster seed: " + response.message;
          for (const auto& instruction : input.flatten(&moveFilter))
          {
            const auto& move = instruction.get().as<MoveInstructionPoly>();
            const auto it = profile->sample_cache->find(profile->sampleCacheKey(move));
            if (it != profile->sample_cache->end() && it->second->ready && it->second->values.empty())
              missing_vertices = true;
          }
          return false; // an identical graph cannot fix a missing vertex/edge
        }
        auto moves = response.results.flatten(&moveFilter);
        int rejected = 0;
        for (std::size_t i = 1; i < moves.size(); ++i)
        {
          const auto& next = moves[i].get().as<MoveInstructionPoly>();
          if (next.getMoveType() != MoveInstructionType::LINEAR) continue;
          const auto a = getJointPosition(moves[i-1].get().as<MoveInstructionPoly>().getWaypoint()).cast<float>().eval();
          const auto b = getJointPosition(next.getWaypoint()).cast<float>().eval();
          auto key = seedEdgeKey(a, b);
          auto found = profile->edge_cache->find(key);
          bool valid;
          if (found != profile->edge_cache->end()) valid = found->second;
          else
          {
            valid = check.evaluate(descartes_light::State<float>(a), descartes_light::State<float>(b)).first;
            profile->edge_cache->emplace(std::move(key), valid);
          }
          if (!valid)
          {
            ++rejected;
            CONSOLE_BRIDGE_logWarn("Seed collision at move %zu (%s)", i, next.getDescription().c_str());
          }
        }
        CONSOLE_BRIDGE_logInform("Seed search %d/%d (%s, angular %d): %d colliding edges", attempt+1,
                                profile->seed_attempts, input.getDescription().c_str(),
                                profile->sample_angular_tolerance, rejected);
        if (!rejected) { output = std::move(response.results); return true; }
        info.status_message = "Raster seed collision after bounded alternate-branch searches";
      }
      return false;
    };
    auto accept = [&](const CompositeInstruction& output) {
      setData(*context.data_storage, "program", output);
      info.return_value = info.status_code = 1;
      info.color = "green";
      info.status_message = "Collision-checked continuous raster seed";
    };
    CompositeInstruction result;
    if (search(program, result)) { accept(result); return info; }
    auto recovered = program;
    bool valid = program.size() >= 3 && program.size() % 2 == 1;
    for (std::size_t i = 1; valid && i + 1 < program.size(); i += 2)
    {
      if (!program[i].isCompositeInstruction() || !program[i-1].isCompositeInstruction())
        throw std::runtime_error("Raster seed recovery requires alternating transfer/raster composites");
      auto raster = program[i].as<CompositeInstruction>();
      raster.setManipulatorInfo(raster.getManipulatorInfo().getCombined(program.getManipulatorInfo()));
      const auto* first = program[i-1].as<CompositeInstruction>().getLastMoveInstruction();
      if (!first) throw std::runtime_error("Raster seed recovery has no first waypoint");
      raster.insert(raster.begin(), *first);
      profile->sample_angular_tolerance = false;
      profile->target_pose_sample_resolution = original->target_pose_sample_resolution;
      profile->target_pose_sample_max = original->target_pose_sample_max;
      valid = search(raster, result);
      // Adding intermediate Cartesian points cannot repair an existing point
      // with no IK candidates. Go directly to angular recovery in that case.
      if (!valid && (!missing_vertices || !profile->angular_recovery))
      {
        auto refined = raster;
        std::optional<Eigen::Isometry3d> previous;
        refineRaster(refined, profile->retry_spacing, previous);
        profile->target_pose_sample_resolution = std::min(original->target_pose_sample_resolution, profile->retry_roll_resolution);
        if (profile->target_pose_sample_min <= -M_PI)
          profile->target_pose_sample_max = M_PI - profile->target_pose_sample_resolution;
        CONSOLE_BRIDGE_logInform("Seed refinement (%s): spacing %.3f m, roll %.2f deg", raster.getDescription().c_str(),
                                profile->retry_spacing, profile->target_pose_sample_resolution*180.0/M_PI);
        valid = search(refined, result);
      }
      if (!valid && profile->angular_recovery)
      {
        // Original spacing retains every requested point and allows a bounded
        // search before the caller splits blocked turns into separate passes.
        profile->sample_angular_tolerance = true;
        profile->target_pose_sample_resolution = std::min(original->target_pose_sample_resolution, profile->retry_roll_resolution);
        if (profile->target_pose_sample_min <= -M_PI)
          profile->target_pose_sample_max = M_PI - profile->target_pose_sample_resolution;
        CONSOLE_BRIDGE_logInform("Angular seed recovery (%s): existing tolerance %.4f, %.4f rad",
            raster.getDescription().c_str(), profile->angular_tolerance.x(), profile->angular_tolerance.y());
        valid = search(raster, result);
      }
      if (!valid)
      {
        info.status_message += " (" + raster.getDescription() + ")";
        break;
      }
      *recovered[i-1].as<CompositeInstruction>().getLastMoveInstruction() = *result.getFirstMoveInstruction();
      result.erase(result.begin()); // the preceding transfer owns the first pose
      recovered[i] = result;
    }
    if (valid) { accept(recovered); return info; }
    info.color = "red";
    CONSOLE_BRIDGE_logError("%s", info.status_message.c_str());
    return info;
  }
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive& ar, const unsigned int)
  { ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PinFreespaceEndpointsTask); }
};
// TrajOpt may move an already valid seed into a tiny contact accepted by its
// numerical convergence tolerance. Keep the original seed only after an
// independent check of every Cartesian constraint, joint limit and interpolated
// collision edge. The usual timing, contact and kinematic tasks still follow.
class ValidateRasterSeedTask : public PinFreespaceEndpointsTask
{
public:
  ValidateRasterSeedTask() = default;
  using PinFreespaceEndpointsTask::PinFreespaceEndpointsTask;
protected:
  TaskComposerNodeInfo runImpl(TaskComposerContext& context, OptionalTaskComposerExecutor) const override
  {
    TaskComposerNodeInfo info(*this);
    info.return_value = info.status_code = 0;
    info.color = "red";
    auto program = getData(*context.data_storage, "program").as<CompositeInstruction>();
    auto env = getData(*context.data_storage, "environment").as<std::shared_ptr<const tesseract_environment::Environment>>();
    auto profiles = getData(*context.data_storage, "profiles").as<std::shared_ptr<ProfileDictionary>>();
    const auto mi = program.getManipulatorInfo();
    auto kin = env->getJointGroup(mi.manipulator);
    const auto names = kin->getJointNames();
    const auto limits = kin->getLimits().joint_limits;
    const auto tcp = env->findTCPOffset(mi);
    const auto cart = getProfile<TrajOptDefaultPlanProfile>("TrajOptMotionPlannerTask",
        program.getProfile("TrajOptMotionPlannerTask"), *profiles);
    const auto contact = getProfile<ContactCheckProfile>("DiscreteContactCheckTask",
        program.getProfile("DiscreteContactCheckTask"), *profiles);
    if (!cart || !contact || !cart->cartesian_constraint_config.enabled ||
        !cart->cartesian_constraint_config.use_tolerance_override)
      throw std::runtime_error("Raster seed validation needs explicit Cartesian and collision profiles");
    const auto& tolerance = cart->cartesian_constraint_config;
    auto collision_config = contact->config;
    collision_config.type = tesseract_collision::CollisionEvaluatorType::LVS_DISCRETE;
    collision_config.longest_valid_segment_length = std::min(collision_config.longest_valid_segment_length, M_PI/180.0);
    DescartesCollisionEdgeEvaluator<double> collision(*env, kin, collision_config, false);
    auto moves = program.flatten(&moveFilter);
    if (moves.size() < 2) throw std::runtime_error("Raster seed has fewer than two points");
    Eigen::VectorXd previous;
    Eigen::Isometry3d previous_target;
    std::size_t samples = 0;
    for (std::size_t i = 0; i < moves.size(); ++i)
    {
      auto& move = moves[i].get().as<MoveInstructionPoly>();
      if (!move.getWaypoint().isCartesianWaypoint() || (i && move.getMoveType() != MoveInstructionType::LINEAR))
        throw std::runtime_error("Raster seed validation requires Cartesian LINEAR passes");
      const auto target = move.getWaypoint().as<CartesianWaypointPoly>().getTransform();
      const auto q = getJointPosition(names, move.getWaypoint());
      if (!q.allFinite() || (q.array() < limits.col(0).array()).any() || (q.array() > limits.col(1).array()).any())
      { info.status_message = "Raster seed violates physical joint limits"; return info; }
      const auto start = i ? previous : q;
      if (!collision.evaluate(descartes_light::State<double>(start), descartes_light::State<double>(q)).first)
      { info.status_message = "Raster seed has an interpolated collision"; return info; }
      const int steps = std::max(1, static_cast<int>(std::ceil((q-start).norm()/(M_PI/180.0))));
      for (int j = 0; j <= steps; ++j)
      {
        const double f = static_cast<double>(j)/steps;
        Eigen::Isometry3d desired = target;
        if (i)
        {
          desired.translation() = (1-f)*previous_target.translation() + f*target.translation();
          desired.linear() = Eigen::Quaterniond(previous_target.linear()).slerp(f, Eigen::Quaterniond(target.linear())).toRotationMatrix();
        }
        const auto state = env->getState(names, start + f*(q-start));
        const auto actual = state.link_transforms.at(mi.working_frame).inverse() * state.link_transforms.at(mi.tcp_frame) * tcp;
        const auto error = tesseract_common::calcTransformError(desired, actual);
        for (Eigen::Index k = 0; k < 6; ++k)
          if (tolerance.coeff[k] > 0 &&
              (error[k] < tolerance.lower_tolerance[k]-1e-6 || error[k] > tolerance.upper_tolerance[k]+1e-6))
          { info.status_message = "Raster seed exceeds Cartesian tolerance at move " + std::to_string(i); return info; }
        ++samples;
      }
      previous = q;
      previous_target = target;
      move.getWaypoint() = StateWaypoint(names, q);
    }
    setData(*context.data_storage, "program", program);
    info.return_value = info.status_code = 1;
    info.color = "green";
    info.status_message = "Validated original raster seed: " + std::to_string(samples) + " interpolated states";
    CONSOLE_BRIDGE_logInform("%s (%s)", info.status_message.c_str(), program.getDescription().c_str());
    return info;
  }
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive& ar, const unsigned int)
  { ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(PinFreespaceEndpointsTask); }
};

// Assert after both TrajOpt and OMPL routes, before the transfer is assembled
// with neighboring rasters. Never accept a successful solver with shifted ends.
class CheckFreespaceEndpointsTask : public TaskComposerTask
{
public:
  static TaskComposerNodePorts ports()
  {
    TaskComposerNodePorts p;
    p.input_required["program"] = TaskComposerNodePorts::SINGLE;
    p.input_required["reference"] = TaskComposerNodePorts::SINGLE;
    return p;
  }
  CheckFreespaceEndpointsTask() : TaskComposerTask("CheckFreespaceEndpointsTask", ports(), true) {}
  CheckFreespaceEndpointsTask(std::string name, const YAML::Node& config, const TaskComposerPluginFactory&)
    : TaskComposerTask(std::move(name), ports(), config) { validatePorts(); }
protected:
  TaskComposerNodeInfo runImpl(TaskComposerContext& context, OptionalTaskComposerExecutor) const override
  {
    TaskComposerNodeInfo info(*this);
    const auto program = getData(*context.data_storage, "program").as<CompositeInstruction>();
    const auto reference = getData(*context.data_storage, "reference").as<CompositeInstruction>();
    const auto* a = reference.getFirstMoveInstruction();
    const auto* b = reference.getLastMoveInstruction();
    const auto* c = program.getFirstMoveInstruction();
    const auto* d = program.getLastMoveInstruction();
    double error = std::numeric_limits<double>::infinity();
    if (a && b && c && d)
    {
      const auto names = getJointNames(a->getWaypoint());
      error = std::max((getJointPosition(names, a->getWaypoint()) - getJointPosition(names, c->getWaypoint())).cwiseAbs().maxCoeff(),
                       (getJointPosition(names, b->getWaypoint()) - getJointPosition(names, d->getWaypoint())).cwiseAbs().maxCoeff());
    }
    const bool valid = std::isfinite(error) && error <= 1e-6;
    info.return_value = info.status_code = valid ? 1 : 0;
    info.color = valid ? "green" : "red";
    info.status_message = "Freespace endpoint error (rad): " + std::to_string(error);
    if (!valid) CONSOLE_BRIDGE_logError("%s", info.status_message.c_str());
    return info;
  }
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive& ar, const unsigned int)
  { ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(TaskComposerTask); }
};
using ValidateRasterSeedTaskFactory = TaskComposerTaskFactory<ValidateRasterSeedTask>;
using CheckFreespaceEndpointsTaskFactory = TaskComposerTaskFactory<CheckFreespaceEndpointsTask>;
using PinFreespaceEndpointsTaskFactory = TaskComposerTaskFactory<PinFreespaceEndpointsTask>;
using CheckedDescartesTaskFactory = TaskComposerTaskFactory<CheckedDescartesTask>;
} // namespace snp_motion_planning
TESSERACT_SERIALIZE_ARCHIVES_INSTANTIATE(snp_motion_planning::PinFreespaceEndpointsTask)
TESSERACT_SERIALIZE_ARCHIVES_INSTANTIATE(snp_motion_planning::CheckedDescartesTask)
BOOST_CLASS_EXPORT_IMPLEMENT(snp_motion_planning::PinFreespaceEndpointsTask)
BOOST_CLASS_EXPORT_IMPLEMENT(snp_motion_planning::CheckedDescartesTask)
TESSERACT_ADD_TASK_COMPOSER_NODE_PLUGIN(snp_motion_planning::PinFreespaceEndpointsTaskFactory, PinFreespaceEndpointsTaskFactory)
TESSERACT_ADD_TASK_COMPOSER_NODE_PLUGIN(snp_motion_planning::CheckedDescartesTaskFactory, CheckedDescartesTaskFactory)

TESSERACT_SERIALIZE_ARCHIVES_INSTANTIATE(snp_motion_planning::CheckFreespaceEndpointsTask)
BOOST_CLASS_EXPORT_IMPLEMENT(snp_motion_planning::CheckFreespaceEndpointsTask)
TESSERACT_ADD_TASK_COMPOSER_NODE_PLUGIN(snp_motion_planning::CheckFreespaceEndpointsTaskFactory, CheckFreespaceEndpointsTaskFactory)

TESSERACT_SERIALIZE_ARCHIVES_INSTANTIATE(snp_motion_planning::ValidateRasterSeedTask)
BOOST_CLASS_EXPORT_IMPLEMENT(snp_motion_planning::ValidateRasterSeedTask)
TESSERACT_ADD_TASK_COMPOSER_NODE_PLUGIN(snp_motion_planning::ValidateRasterSeedTaskFactory, ValidateRasterSeedTaskFactory)
