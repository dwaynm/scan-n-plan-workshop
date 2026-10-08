#include "level_axis_profile.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <tesseract_common/eigen_serialization.h>

namespace snp_motion_planning
{
template <typename FloatType>
template <class Archive>
void LevelAxisDescartesPlanProfile<FloatType>::serialize(Archive& ar, const unsigned int version)
{
  ar& boost::serialization::make_nvp(
      "base", boost::serialization::base_object<tesseract_planning::DescartesDefaultPlanProfile<FloatType>>(*this));
  ar& BOOST_SERIALIZATION_NVP(level_link);
  ar& BOOST_SERIALIZATION_NVP(level_axis);
  ar& BOOST_SERIALIZATION_NVP(level_weight);
  if (version > 0)
  {
    ar& BOOST_SERIALIZATION_NVP(max_joint_step);
    ar& BOOST_SERIALIZATION_NVP(seed_attempts);
    ar& BOOST_SERIALIZATION_NVP(retry_spacing);
    ar& BOOST_SERIALIZATION_NVP(retry_roll_resolution);
  }
  if (version > 1)
  {
    ar& BOOST_SERIALIZATION_NVP(seed_joint_bounds);
    ar& boost::serialization::make_nvp("angular_tolerance_x", angular_tolerance.x());
    ar& boost::serialization::make_nvp("angular_tolerance_y", angular_tolerance.y());
    ar& BOOST_SERIALIZATION_NVP(angular_recovery);
  }
}

template class LevelAxisDescartesPlanProfile<float>;
template class LevelAxisDescartesPlanProfile<double>;

}  // namespace snp_motion_planning

#include <tesseract_common/serialization.h>
TESSERACT_SERIALIZE_ARCHIVES_INSTANTIATE(snp_motion_planning::LevelAxisDescartesPlanProfile<float>)
TESSERACT_SERIALIZE_ARCHIVES_INSTANTIATE(snp_motion_planning::LevelAxisDescartesPlanProfile<double>)
BOOST_CLASS_EXPORT_IMPLEMENT(snp_motion_planning::LevelAxisDescartesPlanProfile<float>)
BOOST_CLASS_EXPORT_IMPLEMENT(snp_motion_planning::LevelAxisDescartesPlanProfile<double>)
