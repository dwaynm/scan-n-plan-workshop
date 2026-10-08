#pragma once

/**
 * SNP Descartes profile: bounded joint steps, request-local seed caches,
 * optional connected seed-joint range, and Cartesian angular recovery.
 * The legacy horizontal wheel-axis preference remains optional (weight zero
 * for the nozzle). Recovery keeps original targets and uses only the existing
 * optimizer tolerances. Selected edges are collision checked by the seed task.
 */

#include <memory>
#include <map>
#include <vector>
#include <cmath>
#include <string>
#include <utility>
#include <mutex>
#include <boost/uuid/uuid_io.hpp>
#include <tesseract_common/utils.h>
#include <tesseract_motion_planners/descartes/descartes_vertex_evaluator.h>
#include <descartes_light/core/waypoint_sampler.h>

#include <Eigen/Geometry>
#include <descartes_light/core/edge_evaluator.h>
#include <descartes_light/core/state_evaluator.h>
#include <descartes_light/edge_evaluators/compound_edge_evaluator.h>
#include <tesseract_command_language/poly/move_instruction_poly.h>
#include <tesseract_common/manipulator_info.h>
#include <tesseract_environment/environment.h>
#include <tesseract_kinematics/core/kinematic_group.h>
#include <tesseract_motion_planners/descartes/profile/descartes_default_plan_profile.h>

namespace snp_motion_planning
{
/** @brief Cost = weight * angle (rad) of `axis` (in `link`) away from horizontal. */
template <typename FloatType>
class LevelAxisStateEvaluator : public descartes_light::StateEvaluator<FloatType>
{
public:
  LevelAxisStateEvaluator(std::shared_ptr<const tesseract_kinematics::KinematicGroup> manip, std::string link,
                          Eigen::Vector3d axis, double weight)
    : manip_(std::move(manip)), link_(std::move(link)), axis_(axis.normalized()), weight_(weight)
  {
  }

  std::pair<bool, FloatType> evaluate(const descartes_light::State<FloatType>& solution) const override
  {
    const Eigen::VectorXd q = solution.values.template cast<double>();
    const tesseract_common::TransformMap tf = manip_->calcFwdKin(q);
    const auto it = tf.find(link_);
    if (it == tf.end())
      return std::make_pair(true, static_cast<FloatType>(0.0));

    // Component of the axis along world up; 0 => perfectly horizontal.
    const Eigen::Vector3d a = it->second.linear() * axis_;
    const double vertical = std::min(1.0, std::abs(a.z()));
    return std::make_pair(true, static_cast<FloatType>(weight_ * std::asin(vertical)));
  }

private:
  std::shared_ptr<const tesseract_kinematics::KinematicGroup> manip_;
  std::string link_;
  Eigen::Vector3d axis_;
  double weight_;
};

/** @brief Rejects an edge that asks any ONE joint to move more than `limit` radians.
 *
 * Cost is left at zero: the Euclidean evaluator it is compounded with already
 * prices joint travel, and this one only has an opinion about what is allowed.
 */
template <typename FloatType>
class MaxJointStepEdgeEvaluator : public descartes_light::EdgeEvaluator<FloatType>
{
public:
  explicit MaxJointStepEdgeEvaluator(double limit) : limit_(static_cast<FloatType>(limit)) {}

  std::pair<bool, FloatType> evaluate(const descartes_light::State<FloatType>& start,
                                      const descartes_light::State<FloatType>& end) const override
  {
    const auto step = (end.values - start.values).cwiseAbs().maxCoeff();
    return std::make_pair(step <= limit_, static_cast<FloatType>(0.0));
  }

private:
  FloatType limit_;
};

// Per-solve cache: graph construction only reads; validation writes after all
// graph workers have joined. Never reused across requests/environments.
using SeedEdgeKey = std::vector<long long>;
using SeedEdgeCache = std::map<SeedEdgeKey, bool>;
template <typename DerivedA, typename DerivedB>
SeedEdgeKey seedEdgeKey(const Eigen::MatrixBase<DerivedA>& a, const Eigen::MatrixBase<DerivedB>& b)
{
  SeedEdgeKey key;
  for (Eigen::Index i = 0; i < a.size(); ++i) key.push_back(std::llround(a[i] * 1e6));
  for (Eigen::Index i = 0; i < b.size(); ++i) key.push_back(std::llround(b[i] * 1e6));
  return key;
}
template <typename FloatType>
class CachedSeedEdgeEvaluator : public descartes_light::EdgeEvaluator<FloatType>
{
public:
  explicit CachedSeedEdgeEvaluator(std::shared_ptr<SeedEdgeCache> cache) : cache_(std::move(cache)) {}
  std::pair<bool, FloatType> evaluate(const descartes_light::State<FloatType>& a,
                                    const descartes_light::State<FloatType>& b) const override
  {
    auto it = cache_->find(seedEdgeKey(a.values, b.values));
    return {it == cache_->end() || it->second, 0};
  }
private:
  std::shared_ptr<SeedEdgeCache> cache_;
};

// Memoize IK within one solve only. Repeated graph searches may change edge
// validity, but their waypoint IK and collision scene are unchanged.
template <typename FloatType>
struct SeedSamples
{
  std::mutex mutex;
  bool ready{false};
  std::vector<descartes_light::StateSample<FloatType>> values;
};
template <typename FloatType>
using SeedSampleCache = std::map<std::string, std::shared_ptr<SeedSamples<FloatType>>>;
template <typename FloatType>
class CachedWaypointSampler : public descartes_light::WaypointSampler<FloatType>
{
public:
  CachedWaypointSampler(std::unique_ptr<descartes_light::WaypointSampler<FloatType>> sampler,
                        std::shared_ptr<SeedSamples<FloatType>> samples)
    : sampler_(std::move(sampler)), samples_(std::move(samples)) {}
  std::vector<descartes_light::StateSample<FloatType>> sample() const override
  {
    std::lock_guard<std::mutex> lock(samples_->mutex);
    if (!samples_->ready) { samples_->values = sampler_->sample(); samples_->ready = true; }
    return samples_->values;
  }
private:
  std::unique_ptr<descartes_light::WaypointSampler<FloatType>> sampler_;
  std::shared_ptr<SeedSamples<FloatType>> samples_;
};

// Intersect seed search bounds with physical limits; never change the robot
// model or widen its range. A workcell may exclude an island disconnected from
// its home by tool self-collision.
inline Eigen::MatrixX2d seedJointLimits(const Eigen::MatrixX2d& physical,
                                      const std::vector<std::string>& names,
                                      const std::map<std::string, std::pair<double, double>>& bounds)
{
  Eigen::MatrixX2d limits = physical;
  for (const auto& entry : bounds)
  {
    const auto it = std::find(names.begin(), names.end(), entry.first);
    if (it == names.end()) throw std::runtime_error("Unknown seed joint: " + entry.first);
    const auto i = std::distance(names.begin(), it);
    limits(i, 0) = std::max(limits(i, 0), entry.second.first);
    limits(i, 1) = std::min(limits(i, 1), entry.second.second);
    if (!std::isfinite(entry.second.first) || !std::isfinite(entry.second.second) || limits(i, 0) >= limits(i, 1))
      throw std::runtime_error("Invalid seed joint bounds: " + entry.first);
  }
  return limits;
}

// Use exactly the optimizer's rotation-vector tolerance. Euler tilts followed
// by free roll are coupled: testing the Euler angles alone accepts invalid IK.
inline tesseract_common::VectorIsometry3d sampleNozzleTolerance(
    const Eigen::Isometry3d& target, const Eigen::Vector2d& tolerance,
    double resolution, double minimum, double maximum)
{
  tesseract_common::VectorIsometry3d poses;
  for (double rx : {0.0, -tolerance.x(), tolerance.x()})
    for (double ry : {0.0, -tolerance.y(), tolerance.y()})
      for (int k = 0; minimum + k * resolution <= maximum + 1e-9; ++k)
      {
        Eigen::Isometry3d pose = target;
        pose.linear() *= (Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(minimum + k * resolution, Eigen::Vector3d::UnitZ())).toRotationMatrix();
        const auto error = tesseract_common::calcTransformError(target, pose);
        if (std::abs(error[3]) <= tolerance.x() + 1e-9 && std::abs(error[4]) <= tolerance.y() + 1e-9)
          poses.push_back(pose);
      }
  return poses;
}

/** @brief DescartesDefaultPlanProfile that scores states by how level `level_axis` is. */
template <typename FloatType>
class LevelAxisDescartesPlanProfile : public tesseract_planning::DescartesDefaultPlanProfile<FloatType>
{
public:
  using Ptr = std::shared_ptr<LevelAxisDescartesPlanProfile<FloatType>>;

  std::string level_link;                        //!< link the axis belongs to (e.g. the wheel body)
  Eigen::Vector3d level_axis{ 0, 0, 1 };         //!< the axis, in that link's frame
  double level_weight{ 0.0 };                    //!< 0 disables and restores stock behaviour
  double max_joint_step{ 0.0 };                  //!< rad, per joint per edge; 0 disables

  int seed_attempts{3};
  double retry_spacing{0.015};
  double retry_roll_resolution{2.5 * M_PI / 180.0};
  std::map<std::string, std::pair<double, double>> seed_joint_bounds;
  Eigen::Vector2d angular_tolerance{0.0, 0.0}; // existing Cartesian tolerances, never enlarged
  bool angular_recovery{false};
  bool sample_angular_tolerance{false}; // transient recovery phase
  std::shared_ptr<SeedEdgeCache> edge_cache; // transient; rebuilt per solve
  std::shared_ptr<SeedSampleCache<FloatType>> sample_cache; // same lifetime as edge_cache

  std::string sampleCacheKey(const tesseract_planning::MoveInstructionPoly& move) const
  {
    return boost::uuids::to_string(move.getUUID()) + ":" +
           std::to_string(this->target_pose_sample_resolution) + ":" +
           std::to_string(sample_angular_tolerance);
  }

  std::unique_ptr<descartes_light::WaypointSampler<FloatType>>
  createWaypointSampler(const tesseract_planning::MoveInstructionPoly& move,
                        const tesseract_common::ManipulatorInfo& mi,
                        const std::shared_ptr<const tesseract_environment::Environment>& env) const override
  {
    auto sampler = tesseract_planning::DescartesDefaultPlanProfile<FloatType>::createWaypointSampler(move, mi, env);
    if (!sample_cache) return sampler;
    const auto key = sampleCacheKey(move);
    auto& samples = (*sample_cache)[key];
    if (!samples) samples = std::make_shared<SeedSamples<FloatType>>();
    return std::make_unique<CachedWaypointSampler<FloatType>>(std::move(sampler), samples);
  }

  std::unique_ptr<descartes_light::StateEvaluator<FloatType>>
  createStateEvaluator(const tesseract_planning::MoveInstructionPoly& move_instruction,
                       const tesseract_common::ManipulatorInfo& composite_manip_info,
                       const std::shared_ptr<const tesseract_environment::Environment>& env) const override
  {
    if (level_weight <= 0.0 || level_link.empty())
      return tesseract_planning::DescartesDefaultPlanProfile<FloatType>::createStateEvaluator(
          move_instruction, composite_manip_info, env);

    tesseract_common::ManipulatorInfo manip_info =
        composite_manip_info.getCombined(move_instruction.getManipulatorInfo());
    if (!this->manipulator_ik_solver.empty())
      manip_info.manipulator_ik_solver = this->manipulator_ik_solver;
    if (manip_info.empty())
      throw std::runtime_error("LevelAxisDescartesPlanProfile: manipulator info is empty!");

    auto manip = tesseract_planning::DescartesPlanProfile<FloatType>::createKinematicGroup(manip_info, *env);
    return std::make_unique<LevelAxisStateEvaluator<FloatType>>(manip, level_link, level_axis, level_weight);
  }

  std::unique_ptr<descartes_light::EdgeEvaluator<FloatType>>
  createEdgeEvaluator(const tesseract_planning::MoveInstructionPoly& move_instruction,
                      const tesseract_common::ManipulatorInfo& composite_manip_info,
                      const std::shared_ptr<const tesseract_environment::Environment>& env) const override
  {
    auto base = tesseract_planning::DescartesDefaultPlanProfile<FloatType>::createEdgeEvaluator(
        move_instruction, composite_manip_info, env);
    if (move_instruction.getMoveType() != tesseract_planning::MoveInstructionType::LINEAR)
      return base;

    // Compound ANDs validity and sums cost, so the stock evaluator keeps pricing
    // the edge and this one only removes the ones that jump.
    auto compound = std::make_unique<descartes_light::CompoundEdgeEvaluator<FloatType>>();
    compound->evaluators.push_back(std::shared_ptr<descartes_light::EdgeEvaluator<FloatType>>(std::move(base)));
    if (max_joint_step > 0.0)
      compound->evaluators.push_back(std::make_shared<MaxJointStepEdgeEvaluator<FloatType>>(max_joint_step));
    if (edge_cache)
      compound->evaluators.push_back(std::make_shared<CachedSeedEdgeEvaluator<FloatType>>(edge_cache));
    return compound;
  }

protected:
  std::unique_ptr<tesseract_planning::DescartesVertexEvaluator>
  createVertexEvaluator(const tesseract_planning::MoveInstructionPoly&,
                        const std::shared_ptr<const tesseract_kinematics::KinematicGroup>& kin,
                        const std::shared_ptr<const tesseract_environment::Environment>&) const override
  {
    return std::make_unique<tesseract_planning::DescartesJointLimitsVertexEvaluator>(
        seedJointLimits(kin->getLimits().joint_limits, kin->getJointNames(), seed_joint_bounds));
  }

  tesseract_planning::PoseSamplerFn createPoseSampler(
      const tesseract_planning::MoveInstructionPoly& move,
      const std::shared_ptr<const tesseract_kinematics::KinematicGroup>& kin,
      const std::shared_ptr<const tesseract_environment::Environment>& env) const override
  {
    if (!sample_angular_tolerance)
      return tesseract_planning::DescartesDefaultPlanProfile<FloatType>::createPoseSampler(move, kin, env);
    return [tolerance = angular_tolerance, resolution = this->target_pose_sample_resolution,
            minimum = this->target_pose_sample_min, maximum = this->target_pose_sample_max](const Eigen::Isometry3d& target) {
      return sampleNozzleTolerance(target, tolerance, resolution, minimum, maximum);
    };
  }

  // The task composer archives the planning problem, so any profile type it may
  // encounter has to be registered with boost::serialization -- otherwise every
  // plan dies with "unregistered class - derived class not registered".
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, const unsigned int);  // NOLINT
};

}  // namespace snp_motion_planning

BOOST_CLASS_EXPORT_KEY(snp_motion_planning::LevelAxisDescartesPlanProfile<float>)
BOOST_CLASS_EXPORT_KEY(snp_motion_planning::LevelAxisDescartesPlanProfile<double>)

#include <boost/serialization/version.hpp>
BOOST_CLASS_VERSION(snp_motion_planning::LevelAxisDescartesPlanProfile<float>, 2)
BOOST_CLASS_VERSION(snp_motion_planning::LevelAxisDescartesPlanProfile<double>, 2)
