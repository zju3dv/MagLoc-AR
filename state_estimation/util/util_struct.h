#ifndef STATE_ESTIMATION_UTIL_UTIL_STRUCT_H_
#define STATE_ESTIMATION_UTIL_UTIL_STRUCT_H_

#include <vector>

#include "util/result_format.h"

namespace state_estimation {

namespace util {

struct DiscretizationResolutions {
  double position_resolution = 1.0;
  double yaw_resolution = 0.2;
  double bluetooth_offset_resolution = 1.0;
  double wifi_offset_resolution = 1.0;
  double geomagnetism_bias_resolution_x = 1.0;
  double geomagnetism_bias_resolution_y = 1.0;
  double geomagnetism_bias_resolution_z = 1.0;

  std::vector<double> discretization_resolutions(void) {
    std::vector<double> resolutions = {this->position_resolution,
                                       this->yaw_resolution,
                                       this->bluetooth_offset_resolution,
                                       this->wifi_offset_resolution,
                                       this->geomagnetism_bias_resolution_x,
                                       this->geomagnetism_bias_resolution_y,
                                       this->geomagnetism_bias_resolution_z};
    return resolutions;
  }
};

struct SmoothInfo {
  int smoothed_steps = 0;
  int all_stored_steps = 0;
};

struct EvaluationResult {
  double gt_overall_log_likelihood = 0.0;
  double est_overall_log_likelihood = 0.0;
  double gt_prediction_log_likelihood = 0.0;
  double est_prediction_log_likelihood = 0.0;
  double gt_observation_log_likelihood = 0.0;
  double est_observation_log_likelihood = 0.0;
  int evaluated_steps = 0;
  int gt_steps = 0;
  int est_steps = 0;
  std::vector<double> smoothed_position_errors;
  std::vector<double> smoothed_yaw_errors;
  std::vector<util::Result> smoothed_results;
  double average_dead_reckoning_step_time = 0.0;
  double average_update_step_time = 0.0;
  double average_smoothed_position_error = 0.0;
  double average_smoothed_yaw_error = 0.0;
  SmoothInfo smooth_info;
};

}  // namepsace util

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_UTIL_UTIL_STRUCT_H_
