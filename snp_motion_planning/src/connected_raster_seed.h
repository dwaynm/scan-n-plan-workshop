#pragma once

#include <Eigen/Cholesky>
#include <tesseract_common/utils.h>
#include <tesseract_environment/environment.h>
#include <tesseract_kinematics/core/joint_group.h>
#include <functional>

namespace snp_motion_planning
{
// A bounded local least-squares solve. Continuation supplies the preceding
// joint state; acceptance uses the real Cartesian tolerances, not the cost.
class ConnectedRasterIK
{
public:
  ConnectedRasterIK(std::shared_ptr<const tesseract_kinematics::JointGroup> kin,
                    std::string base, std::string tip, Eigen::Isometry3d tcp,
                    Eigen::MatrixX2d limits, Eigen::VectorXd tolerance,
                    std::function<void()> check, bool constrain_roll = true)
    : kin_(std::move(kin)), base_(std::move(base)), tip_(std::move(tip)), tcp_(std::move(tcp)),
      limits_(std::move(limits)), tolerance_(std::move(tolerance)), check_(std::move(check)),
      constrain_roll_(constrain_roll) {}

  Eigen::Isometry3d pose(const Eigen::VectorXd& q) const
  {
    const auto tf = kin_->calcFwdKin(q);
    return tf.at(base_).inverse() * tf.at(tip_) * tcp_;
  }

  bool valid(const Eigen::Isometry3d& target, const Eigen::VectorXd& q) const
  {
    if (!q.allFinite() || (q.array() < limits_.col(0).array()).any() ||
        (q.array() > limits_.col(1).array()).any()) return false;
    return (tesseract_common::calcTransformError(target, pose(q)).cwiseAbs().array()
            <= tolerance_.array() + 1e-7).all();
  }

  Eigen::VectorXd solve(const Eigen::Isometry3d& target, Eigen::VectorXd q) const
  {
    q = q.cwiseMax(limits_.col(0)).cwiseMin(limits_.col(1));
    double damping = 1e-5;
    for (int iteration = 0; iteration < 160; ++iteration)
    {
      check_();
      const auto actual = pose(q);
      const Eigen::VectorXd residual = error(target, actual);
      if (residual.norm() < 1e-8) break;
      const Eigen::MatrixXd geometric = kin_->calcJacobian(q, base_, tip_, tcp_.translation());
      Eigen::MatrixXd jacobian(9, q.size());
      jacobian.topRows(3) = 2.0 * geometric.topRows(3);
      for (Eigen::Index j = 0; j < q.size(); ++j)
      {
        const Eigen::Vector3d axis = geometric.block<3, 1>(3, j);
        jacobian.block<3, 1>(3, j) = 0.4 * axis.cross(actual.linear().col(2));
        jacobian.block<3, 1>(6, j) = (constrain_roll_ ? 0.4 : 0.0) * axis.cross(actual.linear().col(0));
      }
      Eigen::MatrixXd h = jacobian.transpose() * jacobian;
      h.diagonal().array() += damping;
      Eigen::VectorXd delta = h.ldlt().solve(-jacobian.transpose() * residual);
      if (!delta.allFinite()) break;
      const double largest = delta.cwiseAbs().maxCoeff();
      if (largest > 0.2) delta *= 0.2 / largest;
      bool improved = false;
      for (double scale : {1.0, 0.5, 0.25, 0.125})
      {
        const Eigen::VectorXd trial = (q + scale * delta).cwiseMax(limits_.col(0)).cwiseMin(limits_.col(1));
        const double cost = error(target, pose(trial)).squaredNorm();
        if (cost < residual.squaredNorm())
        {
          q = trial;
          damping = std::max(1e-10, damping * 0.3);
          improved = true;
          break;
        }
      }
      if (!improved) damping = std::min(1e3, damping * 10);
      if (delta.norm() < 1e-9) break;
    }
    return q;
  }

private:
  Eigen::VectorXd error(const Eigen::Isometry3d& target, const Eigen::Isometry3d& actual) const
  {
    Eigen::VectorXd e(9);
    e << 2.0 * (actual.translation() - target.translation()),
         0.4 * (actual.linear().col(2) - target.linear().col(2)),
         (constrain_roll_ ? 0.4 : 0.0) * (actual.linear().col(0) - target.linear().col(0));
    return e;
  }
  std::shared_ptr<const tesseract_kinematics::JointGroup> kin_;
  std::string base_, tip_;
  Eigen::Isometry3d tcp_;
  Eigen::MatrixX2d limits_;
  Eigen::VectorXd tolerance_;
  std::function<void()> check_;
  bool constrain_roll_;
};
}  // namespace snp_motion_planning
