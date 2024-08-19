/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-12-27 15:49:00
 * @LastEditTime: 2023-03-02 23:45:11
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#ifndef EXAMPLE_WIRELESSPOSITIONINGMULTIPLEXINGAGENT_H_
#define EXAMPLE_WIRELESSPOSITIONINGMULTIPLEXINGAGENT_H_

#include <Eigen/Eigen>
#include <deque>
#include <vector>
#include <memory>
#include <mutex>

#include "configuration/configuration.h"
#include "distribution/distribution_map.h"
#include "distribution/probability_mapper_2d.h"
#include "evaluation/filter_evaluation.h"
#include "prediction_model/compound_prediction_model.h"
#include "prediction_model/motion_model_2d_local_velocity_1d_rotation.h"
#include "prediction_model/motion_model_yaw_differential.h"
#include "prediction_model/parameter_model_random_walk.h"
#include "filter/particle_filter.h"
#include "observation_model/fusion_observation_model.h"
#include "util/client_request.h"
#include "util/result_format.h"
#include "variable/position.h"
#include "compound_state_sampler_wireless_positioning_multiplexing_agent_wo_cov_map_prior.h"
#include "pdr.h"
#include "filter_predict/PredictFilter.h"
#include "sampler/uniform_range_sampler.h"

#ifdef PREDICTION_TIME_PROFILE
#include "util/misc.h"

#endif
namespace state_estimation {

enum class WPAInitializationStatus {
  kSuccess = 0,
  kWPAConfigFileNotFound,
  kNoMapFound,
  kNoMapCovarianceFound,
  kNoCoarseMapFound,
  kNoCoarseMapCovarianceFound,
};

enum class WPAState {
  kInitializing = 0,
  kTracking,
};

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

class WirelessPositioningAgent {
 public:
  WirelessPositioningAgent(void);
  ~WirelessPositioningAgent();

  WPAInitializationStatus Init(std::string config_path);
  WPAInitializationStatus Init(configuration::ConfigParas config_paras);

  util::Result GetPredictionResultInTheMiddle(double predict_timestamp, int stage, const std::vector<Eigen::Vector4d>& gyroscope_data, const std::vector<Eigen::Vector4d>& accelerometer_data);
  util::Result GetPredictionResult(double predict_timestamp, int stage, const std::vector<STSLAMCommon::IMUData>& imu_data);

  util::Result GetResult(const util::ClientRequest& client_request, bool need_update = true, bool prior_p_valid = false, Eigen::Vector3d prior_p = Eigen::Vector3d::Zero());

  double GetBluetoothOffsetFromMap(util::ClientRequest client_request, variable::Position gt_position);

  double GetWifiOffsetFromMap(util::ClientRequest client_request, variable::Position gt_position);

  Eigen::Vector3d GetGeomagnetismBiasFromMap(util::ClientRequest client_request, variable::Position gt_position, Eigen::Quaterniond gt_q_ms);

  SmoothInfo ParticleFilterSmooth(int smooth_start_index = -1);

  EvaluationResult Evaluate(std::vector<prediction_model::CompoundPredictionModelState>& true_states, int smooth_start_index = -1);

  int InjectSpecifiedState(prediction_model::CompoundPredictionModelState specified_state, int particle_index);

  prediction_model::CompoundPredictionModelState GetSpecifiedParticleState(int particle_index);

  int DumpParticleFilterState(std::ofstream &dump_file);

  int DumpParticleFilterStateMemory(std::string dump_filepath);

  int GetParticleFilterOnlineMemorySize(void);

  DiscretizationResolutions resolutions(void) {
    return this->resolutions_;
  }

  void resolutions(DiscretizationResolutions resolutions) {
    this->resolutions_ = resolutions;
  }

  std::deque<double> dead_reckoning_step_times(void) {
    return this->dead_reckoning_step_times_;
  }

  std::deque<double> update_step_times(void) {
    return this->update_step_times_;
  }

  int time_profile_window_size(void) {
    return this->time_profile_window_size_;
  }

  bool is_initialized(void) {
    return this->is_initialized_;
  }

  bool include_ideal_prediction(void) {
    return this->include_ideal_prediction_;
  }

  void include_ideal_prediction(bool include_ideal_prediction) {
    this->include_ideal_prediction_ = include_ideal_prediction;
  }

  void Reset(void);

  void localization_steps(int localization_steps) {
    this->localization_steps_ = localization_steps;
  }

  int localization_steps(void) {
    return this->localization_steps_;
  }

  int step_count(void) {
    return this->step_count_;
  }

  WPAState wpa_state(void) {
    return this->wpa_state_;
  }

  pdr::PdrResult pdr_result(void) {
    return this->pdr_result_;
  }

#ifdef PREDICTION_TIME_PROFILE
  double prediction_time_consumed(void) {
    return this->prediction_time_consumed_;
  }

  int prediction_imu_size(void) {
    return this->prediction_imu_size_;
  }

  double prediction_timestamp(void) {
    return this->prediction_timestamp_;
  }

  double prediction_imu_first_timestamp(void) {
    return this->prediction_imu_first_timestamp_;
  }

  double prediction_imu_last_timestamp(void) {
    return this->prediction_imu_last_timestamp_;
  }

 #endif
 private:
  configuration::ConfigParas config_paras_;
  distribution::DistributionMap bluetooth_distribution_map_;
  distribution::DistributionMap wifi_distribution_map_;
  distribution::DistributionMap geomagnetism_distribution_map_;
  distribution::DistributionMap geomagnetism_coarse_distribution_map_;
  observation_model::FusionObservationModel<distribution::ProbabilityMapper2D,
                                            distribution::ProbabilityMapper2D,
                                            distribution::ProbabilityMapper2D> observation_model_;
  prediction_model::CompoundPredictionModel compound_prediction_model_;
  prediction_model::MotionModel2dLocalVelocity1dRotation prediction_model_0_;
  prediction_model::ParameterModelRandomWalk prediction_model_1_;
  prediction_model::CompoundPredictionModelStateSamplerWPA compound_state_sampler_;
  prediction_model::MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler state_sampler_0_;
  prediction_model::ParameterModelRandomWalkUniformStateSampler state_sampler_1_;
  filter::ParticleFilter<prediction_model::CompoundPredictionModelState,
                         prediction_model::CompoundPredictionModelControlInput,
                         prediction_model::CompoundPredictionModelStateSamplerWPA,
                         prediction_model::CompoundPredictionModel,
                         observation_model::FusionObservation,
                         observation_model::FusionObservationState,
                         observation_model::FusionObservationModel<distribution::ProbabilityMapper2D,
                                                                   distribution::ProbabilityMapper2D,
                                                                   distribution::ProbabilityMapper2D>> particle_filter_;
  evaluation::FilterEvaluator<prediction_model::CompoundPredictionModelState,
                              prediction_model::CompoundPredictionModelControlInput,
                              prediction_model::CompoundPredictionModelStateSamplerWPA,
                              prediction_model::CompoundPredictionModel,
                              observation_model::FusionObservation,
                              observation_model::FusionObservationState,
                              observation_model::FusionObservationModel<distribution::ProbabilityMapper2D,
                                                                        distribution::ProbabilityMapper2D,
                                                                        distribution::ProbabilityMapper2D>> filter_evaluator_;
  Eigen::Matrix<double, 3, 3> R_mw_ = Eigen::Matrix<double, 3, 3>::Identity();
  bool is_initialized_ = false;
  std::mutex is_initialized_mutex_;
  Eigen::Matrix3d R_ws_sensor_pre_ = Eigen::Matrix3d::Identity();
  DiscretizationResolutions resolutions_ = DiscretizationResolutions();
  std::deque<double> dead_reckoning_step_times_ = std::deque<double>();
  std::deque<double> update_step_times_ = std::deque<double>();
  int time_profile_window_size_ = 0;
  bool include_ideal_prediction_ = false;
  int localization_steps_ = 0;
  int step_count_ = 0;
  int update_step_count_ = 0;
  WPAState wpa_state_ = WPAState::kInitializing;
  pdr::PdrComputer pdr_computer_;
  pdr::PdrResult pdr_result_;
  std::unique_ptr<ST_Predict::PredictFilter> predict_filter_ptr_;
  // mutex to support running predict_filter in a separate thread.
  std::mutex predict_filter_mutex_;
#ifdef PREDICTION_TIME_PROFILE
  MyTimer timer_;
  double prediction_time_consumed_ = 0.0;
  int prediction_imu_size_ = 0;
  double prediction_timestamp_ = -1.0;
  double prediction_imu_first_timestamp_ = -1.0;
  double prediction_imu_last_timestamp_ = -1.0;
#endif
  // sampler to represent the prior distribution of static and moving
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double> static_prior_sampler_;
  bool prior_initialized_ = false;
};

}  // namepace state_estimation

#endif  // EXAMPLE_WIRELESSPOSITIONINGMULTIPLEXINGAGENT_H_
