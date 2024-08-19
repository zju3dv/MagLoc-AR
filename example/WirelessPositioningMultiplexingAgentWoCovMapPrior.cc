/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-12-27 15:49:09
 * @LastEditTime: 2023-03-02 23:47:11
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#include "WirelessPositioningMultiplexingAgentWoCovMapPrior.h"

#include <algorithm>
#include <string>
#include <vector>

#include "sampler/gaussian_sampler.h"
#include "observation_model/geomagnetism_observation_model.h"
#include "observation_model/orientation_observation_model.h"

namespace state_estimation {

WirelessPositioningAgent::WirelessPositioningAgent(void) {}

WirelessPositioningAgent::~WirelessPositioningAgent(){};

WPAInitializationStatus WirelessPositioningAgent::Init(std::string config_path) {
  configuration::Configurator configurator;
  if (!configurator.Init(config_path)) {
    return WPAInitializationStatus::kWPAConfigFileNotFound;
  }
  configuration::ConfigParas config_paras = configurator.config_paras();
  return this->Init(config_paras);
}

WPAInitializationStatus WirelessPositioningAgent::Init(configuration::ConfigParas config_paras) {
  config_paras.Justify();
  config_paras_ = config_paras;

  this->static_prior_sampler_.Init(0.0, 1.0);

  this->include_ideal_prediction_ = false;

  // set time_profile_window_size
  assert(config_paras_.time_profile_window_size >= 0);
  time_profile_window_size_ = config_paras_.time_profile_window_size;

  // set member variables
  localization_steps_ = config_paras_.number_of_localization_steps;

  // load map
  bluetooth_distribution_map_.Init(config_paras_.number_of_map_label_fields,
                                   config_paras_.number_of_map_feature_fields);
  int bluetooth_insertion_success = bluetooth_distribution_map_.Insert(config_paras_.bluetooth_distribution_map_path);
  wifi_distribution_map_.Init(config_paras_.number_of_map_label_fields,
                              config_paras_.number_of_map_feature_fields);
  int wifi_insertion_success = wifi_distribution_map_.Insert(config_paras_.wifi_distribution_map_path);
  geomagnetism_distribution_map_.Init(config_paras_.number_of_map_label_fields, config_paras_.number_of_map_feature_fields);
  int geomagnetism_insertion_success = geomagnetism_distribution_map_.Insert(config_paras_.geomagnetism_distribution_map_path);
  bool map_covariance_insertion_no_problem = true;
  if (config_paras_.use_map_covariance) {
    if (geomagnetism_distribution_map_.InsertKeyCovariance(config_paras_.geomagnetism_distribution_map_covariance_path) <= 0) {
      map_covariance_insertion_no_problem = false;
    };
  }

  if (!(bluetooth_insertion_success || wifi_insertion_success || geomagnetism_insertion_success)) {
    return WPAInitializationStatus::kNoMapFound;
  }

  if (!map_covariance_insertion_no_problem) {
    return WPAInitializationStatus::kNoMapCovarianceFound;
  }

  // filter out sub-area and hand over to probability_mapper
  std::set<std::string> bluetooth_all_keys = bluetooth_distribution_map_.GetAllKeys();
  std::set<std::string> wifi_all_keys = wifi_distribution_map_.GetAllKeys();
  std::set<std::string> geomag_all_keys = geomagnetism_distribution_map_.GetAllKeys();
  distribution::ProbabilityMapper2D bluetooth_probability_mapper;
  bluetooth_probability_mapper.Init(bluetooth_all_keys,
                                    &bluetooth_distribution_map_,
                                    config_paras_.bluetooth_uncertainty_scale_factor,
                                    config_paras_.bluetooth_zero_centering);
  distribution::ProbabilityMapper2D wifi_probability_mapper;
  wifi_probability_mapper.Init(wifi_all_keys,
                               &wifi_distribution_map_,
                               config_paras_.wifi_uncertainty_scale_factor,
                               config_paras_.wifi_zero_centering);
  distribution::ProbabilityMapper2D geomagnetism_probability_mapper;
  geomagnetism_probability_mapper.Init(geomag_all_keys,
                                       &geomagnetism_distribution_map_,
                                       config_paras_.geomagnetism_uncertainty_scale_factor,
                                       0);

  // setup observation model
  observation_model_.Init(bluetooth_probability_mapper,
                          wifi_probability_mapper,
                          geomagnetism_probability_mapper,
                          config_paras_.bluetooth_map_spatial_interval,
                          config_paras_.wifi_map_spatial_interval,
                          config_paras_.geomagnetism_map_spatial_interval,
                          config_paras_.bluetooth_zero_centering,
                          config_paras_.wifi_zero_centering,
                          config_paras_.sensor_weights["bluetooth"],
                          config_paras_.sensor_weights["wifi"],
                          config_paras_.sensor_weights["mf"],
                          config_paras_.use_orientation_sensor_constraint,
                          config_paras_.orientation_sensor_constraint_abs_yaw_diff_rad);

  observation_model_.SetGeomagnetismDenseRelativeObservationStepLength(config_paras_.dense_relative_observation_step_length);
  observation_model_.SetGeomagnetismAdditiveNoiseStd(config_paras_.geomagnetism_additive_noise_std);

  // load the coordinate transformation matrix
  R_mw_ = VectorTo2DMatrixC(config_paras_.R_mw_vector, 3);

  // setup prediction_model
  prediction_model_0_ = prediction_model::MotionModel2dLocalVelocity1dRotation(R_mw_);
  if (config_paras_.resample_jitter_state) {
    prediction_model_0_.position_jitter_flag(config_paras_.resample_position_2d_jitter);
    prediction_model_0_.SetPositionJitteringDistributionCovariance(config_paras_.position_2d_jitter_covariance);
    prediction_model_0_.yaw_jitter_flag(config_paras_.resample_yaw_jitter);
    prediction_model_0_.SetYawJitteringDistributionVariance(std::pow(config_paras_.yaw_jitter_std_rad, 2.0));
  }
  Eigen::Matrix<double, 5, 5> parameter_jitter_covariance = Eigen::Matrix<double, 5, 5>::Zero();
  parameter_jitter_covariance(2, 2) = std::pow(config_paras_.geomagnetism_bias_jitter_std, 2.0);
  parameter_jitter_covariance(3, 3) = std::pow(config_paras_.geomagnetism_bias_jitter_std, 2.0);
  parameter_jitter_covariance(4, 4) = std::pow(config_paras_.geomagnetism_bias_jitter_std, 2.0);
  prediction_model_1_.SetParameterJitteringDistributionCovariance(parameter_jitter_covariance);
  std::vector<prediction_model::PredictionModel*> prediction_model_ptrs = {&prediction_model_0_, &prediction_model_1_};
  compound_prediction_model_.Init(prediction_model_ptrs);
#ifdef RANDOMNESS_OFF
  compound_prediction_model_.Seed(config_paras_.random_seed);
#endif

  // setup the state sampler
  std::pair<double, double> bluetooth_offset_range;
  if (config_paras_.bluetooth_offset_estimation) {
    bluetooth_offset_range = {-config_paras_.bluetooth_max_absolute_offset, config_paras_.bluetooth_max_absolute_offset};
  } else {
    bluetooth_offset_range = {-0.0, 0.0};
  }
  std::pair<double, double> wifi_offset_range;
  if (config_paras_.wifi_offset_estimation) {
    wifi_offset_range = {-config_paras_.wifi_max_absolute_offset, config_paras_.wifi_max_absolute_offset};
  } else {
    wifi_offset_range = {-0.0, 0.0};
  }
  std::pair<double, double> geomagnetism_bias_single_dimension_range;
  if (config_paras_.geomagnetism_bias_estimation) {
    geomagnetism_bias_single_dimension_range = {-config_paras_.geomagnetism_max_absolute_bias, config_paras_.geomagnetism_max_absolute_bias};
  } else {
    geomagnetism_bias_single_dimension_range = {-0.0, 0.0};
  }
  std::vector<std::pair<double, double>> parameter_sampling_ranges = {bluetooth_offset_range,
                                                                      wifi_offset_range,
                                                                      geomagnetism_bias_single_dimension_range,
                                                                      geomagnetism_bias_single_dimension_range,
                                                                      geomagnetism_bias_single_dimension_range};
  state_sampler_1_.Init(parameter_sampling_ranges);
  std::vector<prediction_model::StateSampler*> state_sampler_ptrs = {&state_sampler_0_, &state_sampler_1_};
  compound_state_sampler_.Init(state_sampler_ptrs);
  compound_state_sampler_.geomagnetism_bias_use_map_prior(config_paras_.geomagnetism_bias_use_map_prior);
#ifdef RANDOMNESS_OFF
  compound_state_sampler_.Seed(config_paras_.random_seed);
#endif

  bool coarse_distribution_map_no_problem = true;
  bool coarse_distribution_map_covariance_no_problem = true;
  if (config_paras_.number_of_initial_coarse_map_update_steps > 0) {
    if (config_paras_.use_separate_geomagnetism_coarse_map) {
      geomagnetism_coarse_distribution_map_.Init(config_paras_.number_of_map_label_fields, config_paras_.number_of_map_feature_fields);
      if (geomagnetism_coarse_distribution_map_.Insert(config_paras_.geomagnetism_coarse_distribution_map_path) <= 0) {
        coarse_distribution_map_no_problem = false;
      }
      if (config_paras_.use_map_covariance) {
        if (geomagnetism_coarse_distribution_map_.InsertKeyCovariance(config_paras_.geomagnetism_coarse_distribution_map_covariance_path) <= 0) {
          coarse_distribution_map_covariance_no_problem = false;
        }
      }
      std::set<std::string> geomag_coarse_all_keys = geomagnetism_coarse_distribution_map_.GetAllKeys();
      distribution::ProbabilityMapper2D geomagnetism_coarse_probability_mapper;
      geomagnetism_coarse_probability_mapper.Init(geomag_coarse_all_keys,
                                                  &geomagnetism_coarse_distribution_map_,
                                                  config_paras_.coarse_map_uncertainty_scale_factor,
                                                  0);
      observation_model_.SetGeomagnetismProbabilityMapper(geomagnetism_coarse_probability_mapper);
    } else {
      observation_model_.SetGeomagnetismScaleFactor(config_paras_.geomagnetism_uncertainty_scale_factor * config_paras_.coarse_map_uncertainty_scale_factor);
    }
  }

  if (!coarse_distribution_map_no_problem) {
    return WPAInitializationStatus::kNoCoarseMapFound;
  }

  if (!coarse_distribution_map_covariance_no_problem) {
    return WPAInitializationStatus::kNoCoarseMapCovarianceFound;
  }

  // setup the particle_fitler
  particle_filter_.Init(compound_prediction_model_,
                        compound_state_sampler_,
                        observation_model_,
                        config_paras_.n_init_particles,
                        config_paras_.effective_population_ratio,
                        config_paras_.population_expansion_ratio,
                        config_paras_.filter_state_memory_size,
                        config_paras_.max_number_of_particles,
                        config_paras_.relative_observation_window_size,
                        config_paras_.use_relative_observation,
                        config_paras_.use_dense_relative_observation,
                        config_paras_.geomagnetism_bias_use_exponential_averaging,
                        config_paras_.use_orientation_geomagnetism_bias_correlated_jittering);
  particle_filter_.SetLocalResampling(config_paras_.particle_filter_local_resampling);
  particle_filter_.SetLocalResamplingRegionSizeInMeters(config_paras_.particle_filter_local_resampling_region_size_in_meters);
#ifdef RANDOMNESS_OFF
  particle_filter_.Seed(config_paras_.random_seed);
#endif

  // setup evaluator
  filter_evaluator_.Init(compound_state_sampler_,
                         compound_prediction_model_,
                         observation_model_);

  is_initialized_ = false;

  // setup the predict_filter_
  ST_Predict::PredictParameter predict_parameter;
  predict_parameter.imu_freq = config_paras_.imu_sampling_frequency;
  predict_parameter.render_freq = config_paras_.prediction_frequency;
  {
    std::lock_guard<std::mutex> lock(this->predict_filter_mutex_);
    this->predict_filter_ptr_ = std::make_unique<ST_Predict::PredictFilter>(predict_parameter);
  }

  return WPAInitializationStatus::kSuccess;
}

util::Result WirelessPositioningAgent::GetPredictionResultInTheMiddle(double predict_timestamp, int stage, const std::vector<Eigen::Vector4d>& gyroscope_data, const std::vector<Eigen::Vector4d>& accelerometer_data) {
  std::vector<STSLAMCommon::IMUData> imu_data;
  for (int i = 0; i < accelerometer_data.size() && i < gyroscope_data.size(); i++) {
    if (accelerometer_data.at(i)(0) * 1e-9 >= predict_timestamp) {
      break;
    }
    imu_data.emplace_back(STSLAMCommon::IMUData());
    imu_data.back().t = accelerometer_data.at(i)(0) * 1e-9;
    imu_data.back().acc = accelerometer_data.at(i).block(1, 0, 3, 1).cast<float>();
    imu_data.back().gyr = gyroscope_data.at(i).block(1, 0, 3, 1).cast<float>();
  }

  return this->GetPredictionResult(predict_timestamp, stage, imu_data);
}

util::Result WirelessPositioningAgent::GetPredictionResult(double predict_timestamp, int stage, const std::vector<STSLAMCommon::IMUData>& imu_data) {
  // the actual predict time is decided by IMU
  util::Result result;
  result.timestamp(predict_timestamp);

  bool is_initialized;
  {
    std::lock_guard<std::mutex> lock(this->is_initialized_mutex_);
    is_initialized = this->is_initialized_;
  }

  if (!is_initialized) {
    return result;
  }

  std::vector<STSLAMCommon::IMUData> imu_data_down_sampled;
  for (int i = 0; i < imu_data.size(); i += this->config_paras_.prediction_imu_down_sampling_divider) {
    imu_data_down_sampled.emplace_back(imu_data.at(i));
  }

  ST_Predict::PredictState predict_state;
  {
    std::lock_guard<std::mutex> lock(this->predict_filter_mutex_);
#ifdef PREDICTION_TIME_PROFILE
    timer_.Start();
#endif
    this->predict_filter_ptr_->Predict(predict_timestamp, imu_data_down_sampled, stage);
#ifdef PREDICTION_TIME_PROFILE
    this->prediction_imu_size_ = imu_data_down_sampled.size();
    this->prediction_time_consumed_ = timer_.TimePassed();
    if (imu_data_down_sampled.size() > 0) {
      this->prediction_imu_first_timestamp_ = imu_data_down_sampled.at(0).t;
      this->prediction_imu_last_timestamp_ = imu_data_down_sampled.at(imu_data_down_sampled.size() - 1).t;
    } else {
      this->prediction_imu_first_timestamp_ = -1.0;
      this->prediction_imu_last_timestamp_ = -1.0;
    }
    this->prediction_timestamp_ = predict_timestamp;
#endif
    predict_state = this->predict_filter_ptr_->GetSmoothPredictState(predict_timestamp);
  }

  variable::Position temp_position;
  temp_position.x(predict_state.t_wb(0));
  temp_position.y(predict_state.t_wb(1));
  temp_position.z(predict_state.t_wb(2));

  Eigen::Quaterniond q_m(predict_state.r_wb);
  variable::Orientation temp_orientation;
  temp_orientation.q(q_m);
  Eigen::Matrix3d Rz_m = CalculateRzFromOrientation(q_m);
  double yaw_m = GetAngleByAxisFromAngleAxis(Eigen::AngleAxisd(Rz_m), Eigen::Vector3d({0.0, 0.0, 1.0}));

  result.est_position(temp_position);
  result.est_yaw(yaw_m);
  result.est_orientation(temp_orientation);

  return result;
}

util::Result WirelessPositioningAgent::GetResult(const util::ClientRequest& client_request, bool need_update, bool prior_p_valid, Eigen::Vector3d prior_p) {
  util::Result result;
  result.timestamp(client_request.timestamp);

  // whatever the WPA status is, pass accelerometer data to the pdr_computer to detect the activity type
  for (int i = 0; i < client_request.accelerometer_lines.size(); i++) {
    this->pdr_computer_.OnNewAcceByString(client_request.accelerometer_lines.at(i));
  }
  pdr::PdrResult pdr_result = this->pdr_computer_.GetPdrResult(client_request.timestamp);
  this->pdr_result_ = pdr_result;
  switch (pdr_result.motion_status) {
    case pdr::MotionStatus::MOTION_STATUS_STAND:
      result.activity_type(util::ActivityType::kActivityStanding);
      break;
    case pdr::MotionStatus::MOTION_STATUS_MOVE:
      result.activity_type(util::ActivityType::kActivityWalking);
      break;
    case pdr::MotionStatus::MOTION_STATUS_STEP_DETECTED:
      result.activity_type(util::ActivityType::kActivityWalking);
      break;
    default:
      result.activity_type(util::ActivityType::kActivityUnknown);
  }

  if (!is_initialized_) {
    if (client_request.geomagnetism_lines.size() > 0 && client_request.gravity_s_timestamp > -0.5) {
      // set the prior for initial orientation estimation
      // set the client_request.orientation_sensor_pose_ws as the gt_orientation_ws to exclude yaw-estimation.
      observation_model::GeomagnetismObservation geomagnetism_observation_temp;
      geomagnetism_observation_temp.Init(config_paras_.client_buffer_duration, client_request.timestamp, 1.0, config_paras_.R_mw_vector, observation_model::GeomagnetismFeatureVectorType::kThreeDimensionalVector);
      geomagnetism_observation_temp.GetObservationFromLines(client_request.geomagnetism_lines, config_paras_.time_unit_in_second);
      Eigen::Vector3d geomagnetism_s = geomagnetism_observation_temp.GetFeatureValuesVector();
      bool use_gt_orientation;
      if (config_paras_.use_gt_global_orientation) {
        use_gt_orientation = true;
      } else {
        use_gt_orientation = false;
      }
      state_sampler_0_.Init(observation_model_.geomagnetism_observation_model().probability_mapper(), config_paras_.geomagnetism_map_spatial_interval, R_mw_, geomagnetism_s, config_paras_.geomagnetism_s_covariance, client_request.gravity_s, config_paras_.gravity_s_covariance, client_request.gt_orientation, use_gt_orientation);
      state_sampler_0_.samples_per_position_for_traversing(config_paras_.number_of_samples_per_grid_for_traversing);
#ifdef RANDOMNESS_OFF
      state_sampler_0_.Seed(config_paras_.random_seed);
#endif
      is_initialized_ = true;
    } else {
      return result;
    }
  }

  this->step_count_++;

  Eigen::Matrix<double, 3, 1> z_vector = {0.0, 0.0, 1.0};

  // setup observations
  observation_model::FusionObservation fusion_observation;
  fusion_observation.Init(config_paras_.client_buffer_duration, client_request.timestamp, 1.0, config_paras_.R_mw_vector, observation_model::GeomagnetismFeatureVectorType::kThreeDimensionalVector);
  fusion_observation.GetObservationFromLines(client_request.bluetooth_lines, client_request.wifi_lines, client_request.geomagnetism_lines, client_request.gravity_lines, config_paras_.time_unit_in_second);

  observation_model::OrientationObservation orientation_observation;
  orientation_observation.Init(config_paras_.client_buffer_duration, this->R_mw_);
  orientation_observation.GetObservationFromOrientation(client_request.timestamp, client_request.orientation_sensor_pose_ws);
  fusion_observation.orientation_observation(orientation_observation);

  Eigen::Matrix3d R_ws_sensor_current;
  if (config_paras_.use_gt_local_rotation) {
    R_ws_sensor_current = this->R_mw_.transpose() * client_request.gt_orientation;
  } else {
    R_ws_sensor_current = client_request.imu_pose_ws.toRotationMatrix();
  }

  prediction_model::MotionModel2dLocalVelocity1dRotationControlInput* control_input_0 = new prediction_model::MotionModel2dLocalVelocity1dRotationControlInput();
  prediction_model::MotionModelYawDifferentialControlInput motion_model_1d_rotation_control_input;
  motion_model_1d_rotation_control_input.dR(R_ws_sensor_current * this->R_ws_sensor_pre_.transpose());
  motion_model_1d_rotation_control_input.q_sgs(Eigen::Quaterniond(fusion_observation.geomagnetism_observation().R_sgs()));
  if (config_paras_.sample_local_rotation) {
    motion_model_1d_rotation_control_input.dR_error_log_cov(config_paras_.rotation_covariance);
  } else {
    motion_model_1d_rotation_control_input.dR_error_log_cov(Eigen::Matrix3d::Zero());
  }
  if (config_paras_.use_gt_local_velocity) {
    Eigen::Vector3d gt_velocity_local = client_request.gt_orientation.conjugate() * client_request.gt_velocity;
    control_input_0->v_local(gt_velocity_local);
  } else if (config_paras_.use_constant_heading_velocity) {
    double heading_speed = 1.5;
    double rolling_dice = this->static_prior_sampler_.Sample();
    if (rolling_dice > 0.95) {
      heading_speed = 0.0;
    }
    Eigen::Vector3d local_velocity = Eigen::Vector3d::Zero();
    local_velocity(2) = -heading_speed;
    control_input_0->v_local(local_velocity);
  } else if (config_paras_.use_pdr_heading_velocity) {
    double heading_speed = client_request.pdr_heading_speed;
    Eigen::Vector3d local_velocity = Eigen::Vector3d::Zero();
    local_velocity(2) = -heading_speed;
    control_input_0->v_local(local_velocity);
  } else {
    control_input_0->v_local(client_request.INS_v_local);
  }
  if (config_paras_.sample_local_velocity) {
    if (config_paras_.use_constant_heading_velocity) {
      Eigen::Matrix3d local_velocity_covariance = Eigen::Matrix3d::Zero();
      local_velocity_covariance(2, 2) = 0.3 * 0.3;
      control_input_0->v_local_covariance(local_velocity_covariance);
    } else if (config_paras_.use_pdr_heading_velocity) {
      Eigen::Matrix3d local_velocity_covariance = Eigen::Matrix3d::Zero();
      local_velocity_covariance(2, 2) = 0.2 * 0.2;
      control_input_0->v_local_covariance(local_velocity_covariance);
    } else {
      control_input_0->v_local_covariance(config_paras_.INS_v_local_covariance);
    }
  } else {
    control_input_0->v_local_covariance(Eigen::Matrix3d::Zero());
  }
  // set the client_request.gravity_s with gt_gravity to exclude INS gravity error;
  if (config_paras_.use_gt_gravity) {
    Eigen::Vector3d z_vector = {0.0, 0.0, 1.0};
    control_input_0->gravity(client_request.gt_orientation.conjugate() * z_vector);
  } else {
    control_input_0->gravity(client_request.gravity_s);
  }
  control_input_0->motion_model_1d_rotation_control_input(motion_model_1d_rotation_control_input);
  if (config_paras_.translation_module_load_gt_orientation_instead_of_oriention_sensor) {
    control_input_0->orientation_sensor_pose_ws(Eigen::Quaterniond(R_mw_.transpose() * client_request.gt_orientation));
  } else {
    control_input_0->orientation_sensor_pose_ws(client_request.orientation_sensor_pose_ws);
  }
  if (config_paras_.predict_translation_with_estimated_orientation) {
    control_input_0->use_estimated_yaw(true);
  } else {
    control_input_0->use_estimated_yaw(false);
  }

  // behavior when static
  if (config_paras_.use_static_detection) {
    if (result.activity_type() == util::ActivityType::kActivityStanding) {
      if (config_paras_.sample_local_velocity) {
        control_input_0->v_local_covariance(Eigen::Matrix3d::Zero());
      }
      if (!config_paras_.use_gt_local_velocity) {
        control_input_0->v_local(Eigen::Vector3d::Zero());
      }
      need_update = false;
    }
  }

  prediction_model::ParameterModelRandomWalkControlInput* control_input_1 = new prediction_model::ParameterModelRandomWalkControlInput();
  control_input_1->Init(5);
  Eigen::Matrix<double, 5, 1> parameter_covariance_diagonal;
  parameter_covariance_diagonal << config_paras_.bluetooth_offset_variance,
      config_paras_.wifi_offset_variance,
      config_paras_.geomagnetism_bias_variance,
      config_paras_.geomagnetism_bias_variance,
      config_paras_.geomagnetism_bias_variance;
  Eigen::Matrix<double, 5, 5> parameter_covariance(parameter_covariance_diagonal.asDiagonal());
  if (!config_paras_.bluetooth_offset_estimation) {
    parameter_covariance(0, 0) = 0.0;
  }
  if (!config_paras_.wifi_offset_estimation) {
    parameter_covariance(1, 1) = 0.0;
  }
  if (!config_paras_.geomagnetism_bias_estimation) {
    parameter_covariance(2, 2) = 0.0;
    parameter_covariance(3, 3) = 0.0;
    parameter_covariance(4, 4) = 0.0;
  }
  control_input_1->parameter_covariance(parameter_covariance);

  std::vector<prediction_model::ControlInput*> control_input_ptrs = {control_input_0, control_input_1};
  prediction_model::CompoundPredictionModelControlInput compound_control_input;
  compound_control_input.Init(control_input_ptrs);

  // update tminus states
  R_ws_sensor_pre_ = R_ws_sensor_current;

  // setup control_input for particle_filter_control_input
  filter::ParticleFilterControlInput<prediction_model::CompoundPredictionModelControlInput> filter_control_input;
  filter_control_input.control_input(compound_control_input);
  filter_control_input.timestamp(client_request.timestamp);

  // correctly set the wpa_state_
  if (this->step_count_ > this->localization_steps_) {
    this->wpa_state_ = WPAState::kTracking;
  } else {
    this->wpa_state_ = WPAState::kInitializing;
  }

  MyTimer my_timer;
  my_timer.Start();
  if (this->localization_steps_ > 0 && particle_filter_.running_mode() == filter::ParticleFilterModeType::kNormal) {
    particle_filter_.StartLocalizationStage(filter_control_input, fusion_observation);
  // } else if (need_update && (particle_filter_.IsWindowFull() || !(config_paras_.use_relative_observation))) {
  } else if (need_update) {
    particle_filter_.SimpleStep(filter_control_input, fusion_observation, config_paras_.resample_jitter_state, this->include_ideal_prediction_, 0, prior_p_valid, prior_p);
    if (time_profile_window_size_ > 0) {
      update_step_times_.push_back(my_timer.TimePassed());
    }
    update_step_count_++;
    if (update_step_count_ == config_paras_.number_of_initial_coarse_map_update_steps) {
      if (config_paras_.use_separate_geomagnetism_coarse_map) {
        distribution::ProbabilityMapper2D geomagnetism_probability_mapper;
        std::set<std::string> geomag_all_keys = geomagnetism_distribution_map_.GetAllKeys();
        geomagnetism_probability_mapper.Init(geomag_all_keys,
                                             &geomagnetism_distribution_map_,
                                             config_paras_.geomagnetism_uncertainty_scale_factor,
                                             0);
        this->observation_model_.SetGeomagnetismProbabilityMapper(geomagnetism_probability_mapper);
      } else {
        this->observation_model_.SetGeomagnetismScaleFactor(config_paras_.geomagnetism_uncertainty_scale_factor);
      }
      this->particle_filter_.SetObservationModel(this->observation_model_);
    }
  } else {
    particle_filter_.DeadReckoningStep(filter_control_input, this->include_ideal_prediction_);
    particle_filter_.PushObservationToObservationWindow(fusion_observation);
    if (time_profile_window_size_ > 0) {
      dead_reckoning_step_times_.push_back(my_timer.TimePassed());
    }
  }
  if (particle_filter_.running_mode() == filter::ParticleFilterModeType::kLocalization && this->step_count_ > this->localization_steps_) {
    particle_filter_.StartTrackingStage(config_paras_.n_init_particles, config_paras_.resample_jitter_state, fusion_observation);
  }
  my_timer.Close();
  while (update_step_times_.size() > time_profile_window_size_) {
    update_step_times_.pop_front();
  }
  while (dead_reckoning_step_times_.size() > time_profile_window_size_) {
    dead_reckoning_step_times_.pop_front();
  }

  prediction_model::CompoundPredictionModelState est_state;
  std::string success_info = "SUCCESS";
  if (!particle_filter_.EstimateState(&est_state, this->include_ideal_prediction_)) {
    success_info = "FAIL";
  } else {
    success_info = "SUCCESS";
  }

  std::vector<std::pair<std::string, double>> state_named_values;
  est_state.GetAllNamedValues(&state_named_values);
  double bluetooth_offset, wifi_offset;
  Eigen::Vector3d geomag_bias = Eigen::Vector3d::Zero();
  for (int i = 0; i < state_named_values.size(); i++) {
    if (state_named_values.at(i).first == state_estimation::util::kNameBluetoothDynamicOffset) {
      bluetooth_offset = state_named_values.at(i).second;
    }
    if (state_named_values.at(i).first == state_estimation::util::kNameWifiDynamicOffset) {
      wifi_offset = state_named_values.at(i).second;
    }
    if (state_named_values.at(i).first == state_estimation::util::kNameGeomagnetismBiasX) {
      geomag_bias(0) = state_named_values.at(i).second;
    }
    if (state_named_values.at(i).first == state_estimation::util::kNameGeomagnetismBiasY) {
      geomag_bias(1) = state_named_values.at(i).second;
    }
    if (state_named_values.at(i).first == state_estimation::util::kNameGeomagnetismBiasZ) {
      geomag_bias(2) = state_named_values.at(i).second;
    }
  }

  result.est_position(est_state.position());
  if (config_paras_.return_orientation_sensor_orientation) {
    if (client_request.orientation_sensor_timestamp > 0.0) {
      Eigen::AngleAxisd angleaxis_msg(CalculateRzFromOrientation(Eigen::Quaterniond(this->R_mw_ * client_request.orientation_sensor_pose_ws)));
      result.est_yaw(GetAngleByAxisFromAngleAxis(angleaxis_msg, Eigen::Vector3d({0.0, 0.0, 1.0})));
      Eigen::Quaterniond est_q = CalculateOrientationFromYawAndGravity(result.est_yaw(), client_request.gravity_s);
      variable::Orientation est_orientation;
      est_orientation.q(est_q);
      result.est_orientation(est_orientation);
    }
  } else {
    // set est_yaw to yaw_m
    Eigen::AngleAxisd angleaxis_wsg(est_state.yaw(), Eigen::Vector3d({0.0, 0.0, 1.0}));
    Eigen::AngleAxisd angleaxis_msg(CalculateRzFromOrientation(Eigen::Quaterniond(R_mw_ * angleaxis_wsg)));
    result.est_yaw(GetAngleByAxisFromAngleAxis(angleaxis_msg, Eigen::Vector3d({0.0, 0.0, 1.0})));
    Eigen::Quaterniond est_q = CalculateOrientationFromYawAndGravity(result.est_yaw(), client_request.gravity_s);
    variable::Orientation est_orientation;
    est_orientation.q(est_q);
    result.est_orientation(est_orientation);
  }
  result.distance_variance(particle_filter_.EstimateDistanceVariance(&est_state));
  result.est_bluetooth_offset(bluetooth_offset);
  result.est_wifi_offset(wifi_offset);
  result.est_geomagnetism_bias_3d(geomag_bias);
  result.log_probability_est(est_state.state_log_probability());

  if (config_paras_.particle_statistics) {
    std::vector<prediction_model::CompoundPredictionModelState> current_particle_states = particle_filter_.filter_state()->particle_states();
    int n_current_particles = current_particle_states.size();
    std::sort(current_particle_states.begin(), current_particle_states.end(),
              [&](prediction_model::CompoundPredictionModelState c_state_0, prediction_model::CompoundPredictionModelState c_state_1) {
                return (c_state_0.state_log_probability() > c_state_1.state_log_probability());
              });
    result.log_probability_max((current_particle_states.begin())->state_log_probability());
    result.log_probability_min((current_particle_states.end() - 1)->state_log_probability());
    result.log_probability_top_20_percent(current_particle_states.at(std::floor(n_current_particles * 0.2)).state_log_probability());
    result.log_probability_top_50_percent(current_particle_states.at(std::floor(n_current_particles * 0.5)).state_log_probability());
    result.log_probability_top_80_percent(current_particle_states.at(std::floor(n_current_particles * 0.8)).state_log_probability());
  }

  // update predict_filter_
  ST_Predict::VIOState vio_state;
  vio_state.frame_id = this->step_count_;
  vio_state.b_state_update = false;
  vio_state.timestamp = result.timestamp();
  vio_state.t_wi(0) = result.est_position().x();
  vio_state.t_wi(1) = result.est_position().y();
  vio_state.t_wi(2) = result.est_position().z();
  vio_state.r_wi = result.est_orientation().q().cast<float>();
  // vio_state.v = vio_state.r_wi * client_request.INS_v_local.cast<float>();
  vio_state.v = vio_state.r_wi * control_input_0->v_local().cast<float>();
  vio_state.bw = Eigen::Vector3f::Zero();
  vio_state.ba = Eigen::Vector3f::Zero();
  vio_state.imu_data.acc = client_request.accelerometer_data.back().block(1, 0, 3, 1).cast<float>();
  vio_state.imu_data.gyr = client_request.gyroscope_data.back().block(1, 0, 3, 1).cast<float>();
  {
    std::lock_guard<std::mutex> lock(this->predict_filter_mutex_);
    this->predict_filter_ptr_->UpdateVIOState(vio_state);
  }

  return result;
}

double WirelessPositioningAgent::GetBluetoothOffsetFromMap(util::ClientRequest client_request, variable::Position gt_position) {
  observation_model::FusionObservation fusion_observation;
  fusion_observation.Init(config_paras_.client_buffer_duration, client_request.timestamp, 1.0, config_paras_.R_mw_vector, observation_model::GeomagnetismFeatureVectorType::kThreeDimensionalVector);
  fusion_observation.GetObservationFromLines(client_request.bluetooth_lines, client_request.wifi_lines, client_request.geomagnetism_lines, client_request.gravity_lines, config_paras_.time_unit_in_second);

  observation_model::FusionObservationState gt_state;
  observation_model::BluetoothObservationState gt_bluetooth_state;
  gt_bluetooth_state.position(gt_position);
  gt_state.bluetooth_observation_state(gt_bluetooth_state);

  return this->observation_model_.CalculateBluetoothOffset(fusion_observation, gt_state);
}

double WirelessPositioningAgent::GetWifiOffsetFromMap(util::ClientRequest client_request, variable::Position gt_position) {
  observation_model::FusionObservation fusion_observation;
  fusion_observation.Init(config_paras_.client_buffer_duration, client_request.timestamp, 1.0, config_paras_.R_mw_vector, observation_model::GeomagnetismFeatureVectorType::kThreeDimensionalVector);
  fusion_observation.GetObservationFromLines(client_request.bluetooth_lines, client_request.wifi_lines, client_request.geomagnetism_lines, client_request.gravity_lines, config_paras_.time_unit_in_second);

  observation_model::FusionObservationState gt_state;
  observation_model::WifiObservationState gt_wifi_state;
  gt_wifi_state.position(gt_position);
  gt_state.wifi_observation_state(gt_wifi_state);

  return this->observation_model_.CalculateWifiOffset(fusion_observation, gt_state);
}

Eigen::Vector3d WirelessPositioningAgent::GetGeomagnetismBiasFromMap(util::ClientRequest client_request, variable::Position gt_position, Eigen::Quaterniond gt_q_ms) {
  observation_model::FusionObservation fusion_observation;
  fusion_observation.Init(config_paras_.client_buffer_duration, client_request.timestamp, 1.0, config_paras_.R_mw_vector, observation_model::GeomagnetismFeatureVectorType::kThreeDimensionalVector);
  fusion_observation.GetObservationFromLines(client_request.bluetooth_lines, client_request.wifi_lines, client_request.geomagnetism_lines, client_request.gravity_lines, config_paras_.time_unit_in_second);

  observation_model::FusionObservationState gt_state;
  observation_model::GeomagnetismObservationYawState gt_geomagnetism_state;
  gt_geomagnetism_state.position(gt_position);
  gt_state.geomagnetism_observation_state(gt_geomagnetism_state);

  return this->observation_model_.CalculateGeomagnetismBias(fusion_observation, gt_state, gt_q_ms);
}

SmoothInfo WirelessPositioningAgent::ParticleFilterSmooth(int smooth_start_index) {
  int smoothed_steps = particle_filter_.BackwardSimulationResample(resolutions_.discretization_resolutions(), smooth_start_index);
  int all_stored_steps = particle_filter_.GetFilterStateMemoryCurrentSize();
  SmoothInfo smooth_info;
  smooth_info.smoothed_steps = smoothed_steps;
  smooth_info.all_stored_steps = all_stored_steps;
  return smooth_info;
}

EvaluationResult WirelessPositioningAgent::Evaluate(std::vector<prediction_model::CompoundPredictionModelState>& true_states, int smooth_start_index) {
  SmoothInfo smooth_info = this->ParticleFilterSmooth(smooth_start_index);

  std::cout << "WirelessPositioningAgent::Evaluate: " << smooth_info.smoothed_steps << "/" << smooth_info.all_stored_steps << std::endl;

  EvaluationResult evaluation_result;
  evaluation_result.gt_steps = true_states.size();
  evaluation_result.est_steps = smooth_info.smoothed_steps;

  std::vector<double> timestamps;
  std::vector<prediction_model::CompoundPredictionModelState> gt_states;
  std::vector<prediction_model::CompoundPredictionModelState> est_states;
  std::vector<prediction_model::CompoundPredictionModelControlInput> control_inputs;
  std::vector<observation_model::FusionObservation> observations;
  std::vector<bool> need_updates;

  if (true_states.size() < smooth_info.smoothed_steps) {
    evaluation_result.evaluated_steps = 0;
    return evaluation_result;
  }

  evaluation_result.evaluated_steps = smooth_info.smoothed_steps;

  particle_filter_.PushState();
  int counter = 0;
  prediction_model::CompoundPredictionModelControlInput control_input_tminus;
  observation_model::FusionObservation observation_tminus;
  bool need_update_tminus = false;
  while (particle_filter_.StepSimulationSmoothed()) {
    timestamps.push_back((particle_filter_.filter_state())->timestamp());
    gt_states.push_back(true_states.at(true_states.size() - smooth_info.smoothed_steps + counter));
    prediction_model::CompoundPredictionModelState est_state;
    particle_filter_.EstimateState(&est_state);
    est_states.push_back(est_state);
    control_inputs.push_back(control_input_tminus);
    control_input_tminus = particle_filter_.filter_state()->filter_control_input().control_input();
    observations.push_back(observation_tminus);
    observation_tminus = particle_filter_.filter_state()->observation();
    need_updates.push_back(need_update_tminus);
    need_update_tminus = particle_filter_.filter_state()->need_update();
    counter++;

    std::vector<std::pair<std::string, double>> state_named_values;
    est_state.GetAllNamedValues(&state_named_values);
    double bluetooth_offset = 0.0;
    double wifi_offset = 0.0;
    Eigen::Vector3d geomag_bias = Eigen::Vector3d::Zero();
    for (int i = 0; i < state_named_values.size(); i++) {
      if (state_named_values.at(i).first == state_estimation::util::kNameBluetoothDynamicOffset) {
        bluetooth_offset = state_named_values.at(i).second;
      }
      if (state_named_values.at(i).first == state_estimation::util::kNameWifiDynamicOffset) {
        wifi_offset = state_named_values.at(i).second;
      }
      if (state_named_values.at(i).first == state_estimation::util::kNameGeomagnetismBiasX) {
        geomag_bias(0) = state_named_values.at(i).second;
      }
      if (state_named_values.at(i).first == state_estimation::util::kNameGeomagnetismBiasY) {
        geomag_bias(1) = state_named_values.at(i).second;
      }
      if (state_named_values.at(i).first == state_estimation::util::kNameGeomagnetismBiasZ) {
        geomag_bias(2) = state_named_values.at(i).second;
      }
    }

    util::Result temp_result;
    temp_result.timestamp(timestamps.back());
    temp_result.est_position(est_state.position());
    temp_result.est_yaw(est_state.yaw());
    temp_result.distance_variance(particle_filter_.EstimateDistanceVariance(&est_state));
    temp_result.gt_position(gt_states.back().position());
    temp_result.gt_yaw(gt_states.back().yaw());
    temp_result.est_bluetooth_offset(bluetooth_offset);
    temp_result.est_wifi_offset(wifi_offset);
    temp_result.est_geomagnetism_bias_3d(geomag_bias);

    double position_error = std::pow(std::pow((temp_result.est_position().x() - temp_result.gt_position().x()), 2.0) +
                                     std::pow((temp_result.est_position().y() - temp_result.gt_position().y()), 2.0), 0.5);
    evaluation_result.smoothed_position_errors.push_back(position_error);
    double yaw_error = std::abs(temp_result.est_yaw() - temp_result.gt_yaw());
    yaw_error = yaw_error - std::floor(yaw_error / (2 * M_PI)) * (2 * M_PI);
    if (2 * M_PI - yaw_error > yaw_error) {
      evaluation_result.smoothed_yaw_errors.push_back(yaw_error);
    } else {
      evaluation_result.smoothed_yaw_errors.push_back(2 * M_PI - yaw_error);
    }

    evaluation_result.smoothed_results.push_back(temp_result);
  }

  filter_evaluator_.SetData(timestamps,
                            gt_states,
                            est_states,
                            control_inputs,
                            observations,
                            need_updates);

  evaluation_result.gt_overall_log_likelihood = filter_evaluator_.CalculateOverallProbabilityLog(evaluation::EvaluationType::kGT);
  evaluation_result.est_overall_log_likelihood = filter_evaluator_.CalculateOverallProbabilityLog(evaluation::EvaluationType::kEst);
  evaluation_result.gt_prediction_log_likelihood = filter_evaluator_.CalculatePredictionProbabilityLog(evaluation::EvaluationType::kGT);
  evaluation_result.est_prediction_log_likelihood = filter_evaluator_.CalculatePredictionProbabilityLog(evaluation::EvaluationType::kEst);
  evaluation_result.gt_observation_log_likelihood = filter_evaluator_.CalculateObservationProbabilityLog(evaluation::EvaluationType::kGT);
  evaluation_result.est_observation_log_likelihood = filter_evaluator_.CalculateObservationProbabilityLog(evaluation::EvaluationType::kEst);

  // filter_evaluator_.OutputObservationProbabilityDetail();
  // filter_evaluator_.OutputPredictionProbabilityDetail();

  double average_update_step_time = 0.0;
  for (int i = 0; i < update_step_times_.size(); i++) {
    average_update_step_time += update_step_times_.at(i) / update_step_times_.size();
  }
  evaluation_result.average_update_step_time = average_update_step_time;

  double average_dead_reckoning_step_time = 0.0;
  for (int i = 0; i < dead_reckoning_step_times_.size(); i++) {
    average_dead_reckoning_step_time += dead_reckoning_step_times_.at(i) / dead_reckoning_step_times_.size();
  }
  evaluation_result.average_dead_reckoning_step_time = average_dead_reckoning_step_time;

  double average_smoothed_position_error = 0.0;
  for (int i = 0; i < evaluation_result.smoothed_position_errors.size(); i++) {
    average_smoothed_position_error += evaluation_result.smoothed_position_errors.at(i) / evaluation_result.smoothed_position_errors.size();
  }
  evaluation_result.average_smoothed_position_error = average_smoothed_position_error;

  double average_smoothed_yaw_error = 0.0;
  for (int i = 0; i < evaluation_result.smoothed_yaw_errors.size(); i++) {
    average_smoothed_yaw_error += evaluation_result.smoothed_yaw_errors.at(i) / evaluation_result.smoothed_yaw_errors.size();
  }
  evaluation_result.average_smoothed_yaw_error = average_smoothed_yaw_error;

  evaluation_result.smooth_info = smooth_info;

  return evaluation_result;
}

int WirelessPositioningAgent::InjectSpecifiedState(prediction_model::CompoundPredictionModelState specified_state, int particle_index) {
  return this->particle_filter_.InjectSpecifiedState(specified_state, particle_index);
}

prediction_model::CompoundPredictionModelState WirelessPositioningAgent::GetSpecifiedParticleState(int particle_index) {
  return this->particle_filter_.GetSpecifiedParticleState(particle_index);
}

int WirelessPositioningAgent::DumpParticleFilterState(std::ofstream &dump_file) {
  return this->particle_filter_.DumpFilterState(dump_file);
}

int WirelessPositioningAgent::DumpParticleFilterStateMemory(std::string dump_filepath) {
  return this->particle_filter_.DumpFilterStateMemory(dump_filepath);
}

int WirelessPositioningAgent::GetParticleFilterOnlineMemorySize(void) {
  return this->particle_filter_.GetFilterStateMemoryCurrentSize();
}

void WirelessPositioningAgent::Reset(void) {
  is_initialized_ = false;
  particle_filter_.Reset();
  dead_reckoning_step_times_.clear();
  update_step_times_.clear();
}

}  // namespace state_estimation
