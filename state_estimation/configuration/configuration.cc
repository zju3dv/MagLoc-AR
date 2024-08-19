/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-07-07 11:20:00
 * @LastEditTime: 2023-02-14 15:48:15
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#include "configuration/configuration.h"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <sstream>
#include <utility>

#include "util/misc.h"

namespace state_estimation {

namespace configuration {

static const int kOutputDoublePrecision = 3;

int WirelessPositioningConfigurator::Init(const std::string& config_filepath) {
  this->config_filepath_ = config_filepath;
  YAML::Node config;
  try {
    config = YAML::LoadFile(config_filepath);
  } catch (YAML::BadFile) {
    printf("WirelessPositioningConfigurator: cannot open config_filepath: %s\n", config_filepath.c_str());
    return 0;
  }

  if (config["offline"]) {
    this->offline_ = config["offline"].as<bool>();
  }

  if (config["offline_request_path"]) {
    this->offline_request_path_ = config["offline_request_path"].as<std::string>();
  }

  if (config["dump_data"]) {
    this->dump_data_ = config["dump_data"].as<bool>();
  }

  if (config["dump_data_path"]) {
    this->dump_data_path_ = config["dump_data_path"].as<std::string>();
  }

  if (config["update_interval"]) {
    this->update_interval_ = config["update_interval"].as<double>();
  }

  if (config["excution_interval"]) {
    this->excution_interval_ = config["excution_interval"].as<double>();
  }

  if (config["running_prediction"]) {
    this->running_prediction_ = config["running_prediction"].as<bool>();
  }

  if (config["running_INS"]) {
    this->running_INS_ = config["running_INS"].as<bool>();
  }

  if (config["using_gravity_sensor"]) {
    this->using_gravity_sensor_ = config["using_gravity_sensor"].as<bool>();
  }

  if (config["running_IMU_rotation"]) {
    this->running_IMU_rotation_ = config["running_IMU_rotation"].as<bool>();
  }

  return 1;
}

WirelessPositioningConfigurator::WirelessPositioningConfigurator(void) {
  this->config_filepath_ = "";
  this->offline_ = false;
  this->offline_request_path_ = "";
  this->dump_data_ = false;
  this->dump_data_path_ = "";
  this->update_interval_ = 1.0;
  this->excution_interval_ = 0.1;
  this->running_prediction_ = true;
  this->running_INS_ = true;
  this->using_gravity_sensor_ = false;
  this->running_IMU_rotation_ = true;
}

WirelessPositioningConfigurator::~WirelessPositioningConfigurator() {}

std::string CameraIntrinsics::Printf(void) {
  std::stringstream ss;
  ss << "CameraIntrinsics:" << std::endl
     << "  width: " << DoubleToString(width, kOutputDoublePrecision) << std::endl
     << "  height: " << DoubleToString(height, kOutputDoublePrecision) << std::endl
     << "  fx: " << DoubleToString(fx, kOutputDoublePrecision) << std::endl
     << "  fy: " << DoubleToString(fy, kOutputDoublePrecision) << std::endl
     << "  cx: " << DoubleToString(cx, kOutputDoublePrecision) << std::endl
     << "  cy: " << DoubleToString(cy, kOutputDoublePrecision) << std::endl;
  return ss.str();
}

int CameraCalibrationConfigurator::Init(const std::string& calib_filepath, int number_of_cameras) {
  this->calib_filepath_ = calib_filepath;
  this->number_of_cameras_ = number_of_cameras;
  YAML::Node calib;
  try {
    calib = YAML::LoadFile(calib_filepath);
  } catch (YAML::BadFile) {
    std::cout << "CalibrationConfigurator: cannot open calib_filepath " << calib_filepath << std::endl
              << "CalibrationConfigurator: use default calibration parameters." << std::endl;
    return 0;
  }

  for (int i = 0; i < number_of_cameras; i++) {
    std::string camera_id = "cam" + std::to_string(i);

    Eigen::Matrix4d T_cam_imu = Eigen::Matrix4d::Zero();
    CameraIntrinsics my_camera_intrinsics;
    std::string distortion_model = "";
    Eigen::Matrix<double, 8, 1> distortion_coefficients = Eigen::Matrix<double, 8, 1>::Zero();
    if (calib[camera_id]) {
      if (calib[camera_id]["T_cam_imu"]) {
        assert(calib[camera_id]["T_cam_imu"].size() == 4);
        for (int i = 0; i < calib[camera_id]["T_cam_imu"].size(); i++) {
          std::vector<double> T_cam_imu_row_vector = calib[camera_id]["T_cam_imu"][i].as<std::vector<double>>();
          T_cam_imu(i, 0) = T_cam_imu_row_vector.at(0);
          T_cam_imu(i, 1) = T_cam_imu_row_vector.at(1);
          T_cam_imu(i, 2) = T_cam_imu_row_vector.at(2);
          T_cam_imu(i, 3) = T_cam_imu_row_vector.at(3);
        }
      }

      if (calib[camera_id]["intrinsics"]) {
        std::vector<double> camera_intrinsics = calib[camera_id]["intrinsics"].as<std::vector<double>>();
        assert(camera_intrinsics.size() == 4);
        my_camera_intrinsics.fx = camera_intrinsics.at(0);
        my_camera_intrinsics.fy = camera_intrinsics.at(1);
        my_camera_intrinsics.cx = camera_intrinsics.at(2);
        my_camera_intrinsics.cy = camera_intrinsics.at(3);
      }

      if (calib[camera_id]["resolution"]) {
        std::vector<double> camera_resolution = calib[camera_id]["resolution"].as<std::vector<double>>();
        assert(camera_resolution.size() == 2);
        my_camera_intrinsics.width = camera_resolution.at(0);
        my_camera_intrinsics.height = camera_resolution.at(1);
      }

      if (calib[camera_id]["distortion_model"]) {
        distortion_model = calib[camera_id]["distortion_model"].as<std::string>();
      }

      if (calib[camera_id]["distortion_coeffs"]) {
        std::vector<double> distortion_coeffs = calib[camera_id]["distortion_coeffs"].as<std::vector<double>>();
        for (int j = 0; j < distortion_coeffs.size(); j++) {
          distortion_coefficients(j) = distortion_coeffs.at(j);
        }
      }
    }
    this->T_cam_imus_.emplace_back(T_cam_imu);
    this->camera_intrinsicses_.emplace_back(my_camera_intrinsics);
    this->distortion_models_.emplace_back(distortion_model);
    this->distortion_coefficientses_.emplace_back(distortion_coefficients);
  }

  return 1;
}

CameraCalibrationConfigurator::CameraCalibrationConfigurator() {
  this->calib_filepath_ = "";
  this->number_of_cameras_ = 0;
}

CameraCalibrationConfigurator::~CameraCalibrationConfigurator() {}

void ConfigParas::Justify(void) {}

std::string ConfigParas::Printf(void) {
  std::stringstream ss;
  ss << "ConfigParas:" << std::endl;
  ss << "  map_spatial_interval: " << DoubleToString(map_spatial_interval, kOutputDoublePrecision) << std::endl;
  ss << "  bluetooth_map_spatial_interval: " << DoubleToString(bluetooth_map_spatial_interval, kOutputDoublePrecision) << std::endl;
  ss << "  wifi_map_spatial_interval: " << DoubleToString(wifi_map_spatial_interval, kOutputDoublePrecision) << std::endl;
  ss << "  geomagnetism_map_spatial_interval: " << DoubleToString(geomagnetism_map_spatial_interval, kOutputDoublePrecision) << std::endl;
  ss << "  heatmap_smooth_factor: " << DoubleToString(heatmap_smooth_factor, kOutputDoublePrecision) << std::endl;
  ss << "  bluetooth_uncertainty_scale_factor: " << DoubleToString(bluetooth_uncertainty_scale_factor, kOutputDoublePrecision) << std::endl;
  ss << "  wifi_uncertainty_scale_factor: " << DoubleToString(wifi_uncertainty_scale_factor, kOutputDoublePrecision) << std::endl;
  ss << "  geomagnetism_uncertainty_scale_factor: " << DoubleToString(geomagnetism_uncertainty_scale_factor, kOutputDoublePrecision) << std::endl;
  ss << "  geomagnetism_additive_noise_std: " << DoubleToString(geomagnetism_additive_noise_std, kOutputDoublePrecision) << std::endl;
  ss << "  zero_centering: " << zero_centering << std::endl;
  ss << "  acc_std: " << DoubleToString(acc_std, kOutputDoublePrecision) << std::endl;
  ss << "  yaw_std_degree: " << DoubleToString(yaw_std_degree, kOutputDoublePrecision) << std::endl;
  ss << "  v_max: " << DoubleToString(v_max, kOutputDoublePrecision) << std::endl;
  ss << "  v_min: " << DoubleToString(v_min, kOutputDoublePrecision) << std::endl;
  ss << "  effective_population_ratio: " << DoubleToString(effective_population_ratio, kOutputDoublePrecision) << std::endl;
  ss << "  population_expansion_ratio: " << DoubleToString(population_expansion_ratio, kOutputDoublePrecision) << std::endl;
  ss << "  n_init_particles: " << n_init_particles << std::endl;
  ss << "  filter_state_memory_size: " << filter_state_memory_size << std::endl;
  ss << "  max_number_of_particles: " << max_number_of_particles << std::endl;
  ss << "  effective_number_of_feature_threshold: " << effective_number_of_features_threshold << std::endl;
  ss << "  threshold_of_effective_feature_number_for_zero_centering: " << threshold_of_effective_feature_number_for_zero_centering << std::endl;
  ss << "  nearest_k: " << nearest_k << std::endl;
  ss << "  distribution_map_path: " << distribution_map_path << std::endl;
  ss << "  number_of_map_label_fields: " << number_of_map_label_fields << std::endl;
  ss << "  number_of_map_feature_fields: " << number_of_map_feature_fields << std::endl;
  ss << "  number_of_map_features: " << number_of_map_features << std::endl;
  ss << "  bluetooth_zero_centering: " << bluetooth_zero_centering << std::endl;
  ss << "  wifi_zero_centering: " << wifi_zero_centering << std::endl;
  ss << "  bluetooth_distribution_map_path: " << bluetooth_distribution_map_path << std::endl;
  ss << "  wifi_distribution_map_path: " << wifi_distribution_map_path << std::endl;
  ss << "  geomagnetism_distribution_map_path: " << geomagnetism_distribution_map_path << std::endl;
  ss << "  geomagnetism_distribution_map_covariance_path: " << geomagnetism_distribution_map_covariance_path << std::endl;
  ss << "  geomagnetism_coarse_distribution_map_path: " << geomagnetism_coarse_distribution_map_path << std::endl;
  ss << "  geomagnetism_coarse_distribution_map_covariance_path: " << geomagnetism_coarse_distribution_map_covariance_path << std::endl;
  ss << "  v_heading_yaw_min: " << v_heading_yaw_min << std::endl;
  ss << "  v_heading_yaw_max: " << v_heading_yaw_max << std::endl;
  ss << "  v_orientation_angle_min: " << v_orientation_angle_min << std::endl;
  ss << "  v_orientation_angle_max: " << v_orientation_angle_max << std::endl;
  ss << "  bias_min: " << bias_min << std::endl;
  ss << "  bias_max: " << bias_max << std::endl;
  ss << "  v_bias_min: " << v_bias_min << std::endl;
  ss << "  v_bias_max: " << v_bias_max << std::endl;
  ss << "  control_input_means_vector: ";
  for (int i = 0; i < control_input_means_vector.size(); i++) {
    ss << DoubleToString(control_input_means_vector.at(i), kOutputDoublePrecision) << " ";
  }
  ss << std::endl;
  ss << "  control_input_covariances_vector: ";
  for (int i = 0; i < control_input_covariances_vector.size(); i++) {
    ss << DoubleToString(control_input_covariances_vector.at(i), kOutputDoublePrecision) << " ";
  }
  ss << std::endl;
  ss << "  client_trace_path: " << client_trace_path << std::endl;
  ss << "  R_mw_path: " << R_mw_path << std::endl;
  ss << "  R_mw_vector: ";
  for (int i = 0; i < R_mw_vector.size(); i++) {
    ss << DoubleToString(R_mw_vector.at(i), kOutputDoublePrecision) << " ";
  }
  ss << std::endl;
  ss << "  model_trajectory_path: " << model_trajectory_path << std::endl;
  ss << "  output_path: " << output_path << std::endl;
  ss << "  client_buffer_duration: " << DoubleToString(client_buffer_duration, kOutputDoublePrecision) << std::endl;
  ss << "  request_wifi_fingerprints_path: " << request_wifi_fingerprints_path << std::endl;
  ss << "  request_num_effective_wifi_fingerprints_path: " << request_num_effective_wifi_fingerprints_path << std::endl;
  ss << "  request_bluetooth_fingerprints_path: " << request_bluetooth_fingerprints_path << std::endl;
  ss << "  request_num_effective_bluetooth_fingerprints_path: " << request_num_effective_bluetooth_fingerprints_path << std::endl;
  ss << "  request_geomag_fingerprints_path: " << request_geomag_fingerprints_path << std::endl;
  ss << "  request_num_effective_geomag_fingerprints_path: " << request_num_effective_geomag_fingerprints_path << std::endl;
  ss << "  sensor_weights_path: " << sensor_weights_path << std::endl;
  ss << "  sensor_weights: ";
  for (auto p = sensor_weights.begin(); p != sensor_weights.end(); p++) {
    ss << p->first << ":" << p->second << " ";
  }
  ss << std::endl;
  ss << "  results_path: " << results_path << std::endl;
  ss << "  geomagnetism_feature_type: " << geomagnetism_feature_type << std::endl;
  ss << "  time_unit_in_second: " << DoubleToString(time_unit_in_second, kOutputDoublePrecision) << std::endl;
  ss << "  orientation_sensor_yaw_variance_in_rad: " << DoubleToString(orientation_sensor_yaw_variance_in_rad, 8) << std::endl;
  ss << "  bluetooth_max_absolute_offset: " << DoubleToString(bluetooth_max_absolute_offset, kOutputDoublePrecision) << std::endl;
  ss << "  wifi_max_absolute_offset: " << DoubleToString(wifi_max_absolute_offset, kOutputDoublePrecision) << std::endl;
  ss << "  geomagnetism_max_absolute_bias: " << DoubleToString(geomagnetism_max_absolute_bias, kOutputDoublePrecision) << std::endl;
  ss << "  bluetooth_offset_variance: " << DoubleToString(bluetooth_offset_variance, 8) << std::endl;
  ss << "  wifi_offset_variance: " << DoubleToString(wifi_offset_variance, 8) << std::endl;
  ss << "  geomagnetism_bias_variance: " << DoubleToString(geomagnetism_bias_variance, 8) << std::endl;
  ss << "  rotation_covariance: " << std::endl;
  ss << rotation_covariance << std::endl;
  ss << "  INS_v_local_covariance: " << std::endl;
  ss << INS_v_local_covariance << std::endl;
  ss << "  geomagnetism_s_covariance: " << std::endl;
  ss << geomagnetism_s_covariance << std::endl;
  ss << "  gravity_s_covariance: " << std::endl;
  ss << gravity_s_covariance << std::endl;
  ss << "  resample_jitter_state: " << resample_jitter_state << std::endl;
  ss << "  resample_position_2d_jitter: " << resample_position_2d_jitter << std::endl;
  ss << "  position_2d_jitter_covariance: " << std::endl;
  ss << position_2d_jitter_covariance << std::endl;
  ss << "  resample_yaw_jitter: " << resample_yaw_jitter << std::endl;
  ss << "  yaw_jitter_std_rad: " << yaw_jitter_std_rad << std::endl;
  ss << "  geomagnetism_bias_jitter_std: " << geomagnetism_bias_jitter_std << std::endl;
  ss << "  random_seed: " << random_seed << std::endl;
  ss << "  use_gt_gravity: " << use_gt_gravity << std::endl;
  ss << "  use_gt_local_velocity: " << use_gt_local_velocity << std::endl;
  ss << "  sample_local_velocity: " << sample_local_velocity << std::endl;
  ss << "  use_gt_local_rotation: " << use_gt_local_rotation << std::endl;
  ss << "  sample_local_rotation: " << sample_local_rotation << std::endl;
  ss << "  use_gt_global_orientation: " << use_gt_global_orientation << std::endl;
  ss << "  sample_global_orientation: " << sample_global_orientation << std::endl;
  ss << "  translation_module_load_gt_orientation_instead_of_orientation_sensor: " << translation_module_load_gt_orientation_instead_of_oriention_sensor << std::endl;
  ss << "  predict_translation_with_estimated_orientation: " << predict_translation_with_estimated_orientation << std::endl;
  ss << "  return_orientation_sensor_orientation: " << return_orientation_sensor_orientation << std::endl;
  ss << "  bluetooth_offset_estimation: " << bluetooth_offset_estimation << std::endl;
  ss << "  wifi_offset_estimation: " << wifi_offset_estimation << std::endl;
  ss << "  geomagnetism_bias_estimation: " << geomagnetism_bias_estimation << std::endl;
  ss << "  geomagnetism_bias_use_map_prior: " << geomagnetism_bias_use_map_prior << std::endl;
  ss << "  geomagnetism_bias_use_exponential_averaging: " << geomagnetism_bias_use_exponential_averaging << std::endl;
  ss << "  use_orientation_sensor_constraint: " << use_orientation_sensor_constraint << std::endl;
  ss << "  orientation_sensor_constraint_abs_yaw_diff_rad: " << orientation_sensor_constraint_abs_yaw_diff_rad << std::endl;
  ss << "  use_orientation_geomagnetism_bias_correlated_jittering: " << use_orientation_geomagnetism_bias_correlated_jittering << std::endl;
  ss << "  synthesize_gt_local_velocity: " << synthesize_gt_local_velocity << std::endl;
  ss << "  synthesize_gt_local_rotation: " << synthesize_gt_local_rotation << std::endl;
  ss << "  synthesize_gt_global_orientation: " << synthesize_gt_global_orientation << std::endl;
  ss << "  use_relative_observation: " << use_relative_observation << std::endl;
  ss << "  use_dense_relative_observation: " << use_dense_relative_observation << std::endl;
  ss << "  dense_relative_observation_step_length: " << dense_relative_observation_step_length << std::endl;
  ss << "  relative_observation_window_size: " << relative_observation_window_size << std::endl;
  ss << "  time_profile_window_size: " << time_profile_window_size << std::endl;
  ss << "  output_folderpath: " << output_folderpath << std::endl;
  ss << "  smooth_start_index: " << smooth_start_index << std::endl;
  ss << "  stability_evaluation: " << stability_evaluation << std::endl;
  ss << "  number_of_retries: " << number_of_retries << std::endl;
  ss << "  particle_statistics: " << particle_statistics << std::endl;
  ss << "  number_of_samples_per_grid_for_traversing: " << number_of_samples_per_grid_for_traversing << std::endl;
  ss << "  number_of_localization_steps: " << number_of_localization_steps << std::endl;
  ss << "  use_map_covariance: " << use_map_covariance << std::endl;
  ss << "  number_of_initial_coarse_map_update_steps: " << number_of_initial_coarse_map_update_steps << std::endl;
  ss << "  coarse_map_uncertainty_scale_factor: " << coarse_map_uncertainty_scale_factor << std::endl;
  ss << "  particle_filter_local_resampling: " << particle_filter_local_resampling << std::endl;
  ss << "  particle_filter_local_resampling_region_size_in_meters: " << particle_filter_local_resampling_region_size_in_meters << std::endl;
  ss << "  use_separate_geomagnetism_coarse_map: " << use_separate_geomagnetism_coarse_map << std::endl;
  ss << "  use_prediction_filter: " << use_prediction_filter << std::endl;
  ss << "  imu_sampling_frequency: " << imu_sampling_frequency << std::endl;
  ss << "  prediction_frequency: " << prediction_frequency << std::endl;
  ss << "  prediction_imu_down_sampling_divider: " << prediction_imu_down_sampling_divider << std::endl;
  ss << "  use_static_detection: " << use_static_detection << std::endl;
  ss << "  use_constant_heading_velocity: " << use_constant_heading_velocity << std::endl;
  ss << "  use_second_order_constant_heading_velocity: " << use_second_order_constant_heading_velocity << std::endl;
  ss << "  use_pdr_heading_velocity: " << use_pdr_heading_velocity << std::endl;
  ss << "  knn_localization_map_folderpath: " << knn_localization_map_folderpath << std::endl;
  ss << "  knn_localization_sensor: " << knn_localization_sensor << std::endl;

  return ss.str();
}

int Configurator::Init(std::string yaml_config_path) {
  this->yaml_config_path_ = yaml_config_path;
  YAML::Node config;
  try {
    config = YAML::LoadFile(yaml_config_path);
  } catch (YAML::BadFile) {
    std::cout << "Configurator: cannot open config_path, use default configurations." << std::endl;
    return 0;
  }

  ConfigParas config_paras;
  if (config["map_spatial_interval"]) {
    config_paras.map_spatial_interval = config["map_spatial_interval"].as<double>();
  }
  if (config["bluetooth_map_spatial_interval"]) {
    config_paras.bluetooth_map_spatial_interval = config["bluetooth_map_spatial_interval"].as<double>();
  }
  if (config["wifi_map_spatial_interval"]) {
    config_paras.wifi_map_spatial_interval = config["wifi_map_spatial_interval"].as<double>();
  }
  if (config["geomagnetism_map_spatial_interval"]) {
    config_paras.geomagnetism_map_spatial_interval = config["geomagnetism_map_spatial_interval"].as<double>();
  }
  if (config["heatmap_smooth_factor"]) {
    config_paras.heatmap_smooth_factor = config["heatmap_smooth_factor"].as<double>();
  }
  if (config["bluetooth_uncertainty_scale_factor"]) {
    config_paras.bluetooth_uncertainty_scale_factor = config["bluetooth_uncertainty_scale_factor"].as<double>();
  }
  if (config["wifi_uncertainty_scale_factor"]) {
    config_paras.wifi_uncertainty_scale_factor = config["wifi_uncertainty_scale_factor"].as<double>();
  }
  if (config["geomagnetism_uncertainty_scale_factor"]) {
    config_paras.geomagnetism_uncertainty_scale_factor = config["geomagnetism_uncertainty_scale_factor"].as<double>();
  }
  if (config["geomagnetism_additive_noise_std"]) {
    config_paras.geomagnetism_additive_noise_std = config["geomagnetism_additive_noise_std"].as<double>();
  }
  if (config["zero_centering"]) {
    config_paras.zero_centering = config["zero_centering"].as<int>();
  }
  if (config["acc_std"]) {
    config_paras.acc_std = config["acc_std"].as<double>();
  }
  if (config["yaw_std_degree"]) {
    config_paras.yaw_std_degree = config["yaw_std_degree"].as<double>();
  }
  if (config["v_max"]) {
    config_paras.v_max = config["v_max"].as<double>();
  }
  if (config["v_min"]) {
    config_paras.v_min = config["v_min"].as<double>();
  }
  if (config["effective_population_ratio"]) {
    config_paras.effective_population_ratio = config["effective_population_ratio"].as<double>();
  }
  if (config["population_expansion_ratio"]) {
    config_paras.population_expansion_ratio = config["population_expansion_ratio"].as<double>();
  }
  if (config["n_init_particles"]) {
    config_paras.n_init_particles = config["n_init_particles"].as<int>();
  }
  if (config["filter_state_memory_size"]) {
    config_paras.filter_state_memory_size = config["filter_state_memory_size"].as<int>();
  }
  if (config["max_number_of_particles"]) {
    config_paras.max_number_of_particles = config["max_number_of_particles"].as<int>();
  }
  if (config["effective_number_of_features_threshold"]) {
    config_paras.effective_number_of_features_threshold = config["effective_number_of_features_threshold"].as<int>();
  }
  if (config["threshold_of_effective_feature_number_for_zero_centering"]) {
    config_paras.threshold_of_effective_feature_number_for_zero_centering = config["threshold_of_effective_feature_number_for_zero_centering"].as<int>();
  }
  if (config["nearest_k"]) {
    config_paras.nearest_k = config["nearest_k"].as<int>();
  }
  if (config["distribution_map_path"]) {
    std::string distribution_map_path = config["distribution_map_path"].as<std::string>();
    config_paras.distribution_map_path = distribution_map_path;
  }
  if (config["number_of_map_label_fields"]) {
    config_paras.number_of_map_label_fields = config["number_of_map_label_fields"].as<int>();
  }
  if (config["number_of_map_feature_fields"]) {
    config_paras.number_of_map_feature_fields = config["number_of_map_feature_fields"].as<int>();
  }
  if (config["number_of_map_features"]) {
    config_paras.number_of_map_features = config["number_of_map_features"].as<int>();
  }
  if (config["bluetooth_zero_centering"]) {
    config_paras.bluetooth_zero_centering = config["bluetooth_zero_centering"].as<int>();
  }
  if (config["wifi_zero_centering"]) {
    config_paras.wifi_zero_centering = config["wifi_zero_centering"].as<int>();
  }
  if (config["bluetooth_distribution_map_path"]) {
    std::string bluetooth_distribution_map_path = config["bluetooth_distribution_map_path"].as<std::string>();
    config_paras.bluetooth_distribution_map_path = bluetooth_distribution_map_path;
  }
  if (config["wifi_distribution_map_path"]) {
    std::string wifi_distribution_map_path = config["wifi_distribution_map_path"].as<std::string>();
    config_paras.wifi_distribution_map_path = wifi_distribution_map_path;
  }
  if (config["geomagnetism_distribution_map_path"]) {
    std::string geomagnetism_distribution_map_path = config["geomagnetism_distribution_map_path"].as<std::string>();
    config_paras.geomagnetism_distribution_map_path = geomagnetism_distribution_map_path;
  }
  if (config["geomagnetism_distribution_map_covariance_path"]) {
    std::string geomagnetism_distribution_map_covariance_path = config["geomagnetism_distribution_map_covariance_path"].as<std::string>();
    config_paras.geomagnetism_distribution_map_covariance_path = geomagnetism_distribution_map_covariance_path;
  }
  if (config["geomagnetism_coarse_distribution_map_path"]) {
    std::string geomagnetism_coarse_distribution_map_path = config["geomagnetism_coarse_distribution_map_path"].as<std::string>();
    config_paras.geomagnetism_coarse_distribution_map_path = geomagnetism_coarse_distribution_map_path;
  }
  if (config["geomagnetism_coarse_distribution_map_covariance_path"]) {
    std::string geomagnetism_coarse_distribution_map_covariance_path = config["geomagnetism_coarse_distribution_map_covariance_path"].as<std::string>();
    config_paras.geomagnetism_coarse_distribution_map_covariance_path = geomagnetism_coarse_distribution_map_covariance_path;
  }

  if (config["v_heading_yaw_min"]) {
    config_paras.v_heading_yaw_min = config["v_heading_yaw_min"].as<double>();
  }
  if (config["v_heading_yaw_max"]) {
    config_paras.v_heading_yaw_max = config["v_heading_yaw_max"].as<double>();
  }
  if (config["v_orientation_angle_min"]) {
    config_paras.v_orientation_angle_min = config["v_orientation_angle_min"].as<double>();
  }
  if (config["v_orientation_angle_max"]) {
    config_paras.v_orientation_angle_max = config["v_orientation_angle_max"].as<double>();
  }

  if (config["bias_min"]) {
    config_paras.bias_min = config["bias_min"].as<double>();
  }
  if (config["bias_max"]) {
    config_paras.bias_max = config["bias_max"].as<double>();
  }
  if (config["v_bias_min"]) {
    config_paras.v_bias_min = config["v_bias_min"].as<double>();
  }
  if (config["v_bias_max"]) {
    config_paras.v_bias_max = config["v_bias_max"].as<double>();
  }

  if (config["control_input_means_vector"]) {
    config_paras.control_input_means_vector = config["control_input_means_vector"].as<std::vector<double>>();
  }
  if (config["control_input_covariances_vector"]) {
    config_paras.control_input_covariances_vector = config["control_input_covariances_vector"].as<std::vector<double>>();
  }

  if (config["client_trace_path"]) {
    std::string client_trace_path = config["client_trace_path"].as<std::string>();
    config_paras.client_trace_path = client_trace_path;
  }
  if (config["R_mw_path"]) {
    std::string R_mw_path = config["R_mw_path"].as<std::string>();
    config_paras.R_mw_path = R_mw_path;
    YAML::Node R_mw_node;
    try {
      R_mw_node = YAML::LoadFile(R_mw_path);
    } catch (YAML::BadFile) {
      std::cout << "Configurator: cannot open R_mw_path, use default configurations." << std::endl;
    }
    if (R_mw_node["transMatrix"]["data"]) {
      config_paras.R_mw_vector = R_mw_node["transMatrix"]["data"].as<std::vector<double>>();
    } else {
      std::cout << "Configurator: no R_mw provided, use the default one." << std::endl;
    }
  }

  if (config["model_trajectory_path"]) {
    std::string model_trajectory_path = config["model_trajectory_path"].as<std::string>();
    config_paras.model_trajectory_path = model_trajectory_path;
  }
  if (config["output_path"]) {
    std::string output_path = config["output_path"].as<std::string>();
    config_paras.output_path = output_path;
  }

  if (config["client_buffer_duration"]) {
    config_paras.client_buffer_duration = config["client_buffer_duration"].as<double>();
  }

  if (config["request_wifi_fingerprints_path"]) {
    std::string request_wifi_fingerprints_path = config["request_wifi_fingerprints_path"].as<std::string>();
    config_paras.request_wifi_fingerprints_path = request_wifi_fingerprints_path;
  }
  if (config["request_num_effective_wifi_fingerprints_path"]) {
    std::string request_num_effective_wifi_fingerprints_path = config["request_num_effective_wifi_fingerprints_path"].as<std::string>();
    config_paras.request_num_effective_wifi_fingerprints_path = request_num_effective_wifi_fingerprints_path;
  }
  if (config["request_bluetooth_fingerprints_path"]) {
    std::string request_bluetooth_fingerprints_path = config["request_bluetooth_fingerprints_path"].as<std::string>();
    config_paras.request_bluetooth_fingerprints_path = request_bluetooth_fingerprints_path;
  }
  if (config["request_num_effective_bluetooth_fingerprints_path"]) {
    std::string request_num_effective_bluetooth_fingerprints_path = config["request_num_effective_bluetooth_fingerprints_path"].as<std::string>();
    config_paras.request_num_effective_bluetooth_fingerprints_path = request_num_effective_bluetooth_fingerprints_path;
  }
  if (config["request_geomag_fingerprints_path"]) {
    std::string request_geomag_fingerprints_path = config["request_geomag_fingerprints_path"].as<std::string>();
    config_paras.request_geomag_fingerprints_path = request_geomag_fingerprints_path;
  }
  if (config["request_num_effective_geomag_fingerprints_path"]) {
    std::string request_num_effective_geomag_fingerprints_path = config["request_num_effective_geomag_fingerprints_path"].as<std::string>();
    config_paras.request_num_effective_geomag_fingerprints_path = request_num_effective_geomag_fingerprints_path;
  }

  if (config["sensor_weights_path"]) {
    std::string sensor_weights_path = config["sensor_weights_path"].as<std::string>();
    config_paras.sensor_weights_path = sensor_weights_path;
    YAML::Node sensor_weights_node;
    try {
      sensor_weights_node = YAML::LoadFile(sensor_weights_path);
    } catch (YAML::BadFile) {
      std::cout << "Configurator: cannot open sensor_weights_path. use the default one." << std::endl;
    }
    if (sensor_weights_node.IsNull()) {
      std::cout << "Configurator: cannot open sensor_weights_path. use the default one." << std::endl;
    } else {
      for (YAML::const_iterator yaml_iter = sensor_weights_node.begin(); yaml_iter != sensor_weights_node.end(); yaml_iter++) {
        std::string attr_name = yaml_iter->first.as<std::string>();
        double attr_value = yaml_iter->second.as<double>();
        if (config_paras.sensor_weights.find(attr_name) != config_paras.sensor_weights.end()) {
          config_paras.sensor_weights.at(attr_name) = attr_value;
        } else {
          config_paras.sensor_weights.insert(std::pair<std::string, double>(attr_name, attr_value));
        }
      }
    }
  }

  if (config["results_path"]) {
    std::string results_path = config["results_path"].as<std::string>();
    config_paras.results_path = results_path;
  }

  if (config["geomagnetism_feature_type"]) {
    config_paras.geomagnetism_feature_type = config["geomagnetism_feature_type"].as<int>();
  }

  if (config["time_unit_in_second"]) {
    config_paras.time_unit_in_second = config["time_unit_in_second"].as<double>();
  }

  if (config["orientation_sensor_yaw_variance_in_rad"]) {
    config_paras.orientation_sensor_yaw_variance_in_rad = config["orientation_sensor_yaw_variance_in_rad"].as<double>();
  }

  if (config["bluetooth_max_absolute_offset"]) {
    config_paras.bluetooth_max_absolute_offset = config["bluetooth_max_absolute_offset"].as<double>();
  }

  if (config["wifi_max_absolute_offset"]) {
    config_paras.wifi_max_absolute_offset = config["wifi_max_absolute_offset"].as<double>();
  }

  if (config["geomagnetism_max_absolute_bias"]) {
    config_paras.geomagnetism_max_absolute_bias = config["geomagnetism_max_absolute_bias"].as<double>();
  }

  if (config["bluetooth_offset_variance"]) {
    config_paras.bluetooth_offset_variance = config["bluetooth_offset_variance"].as<double>();
  }

  if (config["wifi_offset_variance"]) {
    config_paras.wifi_offset_variance = config["wifi_offset_variance"].as<double>();
  }

  if (config["geomagnetism_bias_variance"]) {
    config_paras.geomagnetism_bias_variance = config["geomagnetism_bias_variance"].as<double>();
  }

  if (config["rotation_covariance"]) {
    config_paras.rotation_covariance = CompactVectorToCovarianceMatrix(config["rotation_covariance"].as<std::vector<double>>(), 3);
  }

  if (config["INS_v_local_covariance"]) {
    config_paras.INS_v_local_covariance = CompactVectorToCovarianceMatrix(config["INS_v_local_covariance"].as<std::vector<double>>(), 3);
  }

  if (config["geomagnetism_s_covariance"]) {
    config_paras.geomagnetism_s_covariance = CompactVectorToCovarianceMatrix(config["geomagnetism_s_covariance"].as<std::vector<double>>(), 3);
  }

  if (config["gravity_s_covariance"]) {
    config_paras.gravity_s_covariance = CompactVectorToCovarianceMatrix(config["gravity_s_covariance"].as<std::vector<double>>(), 3);
  }

  if (config["resample_jitter_state"]) {
    config_paras.resample_jitter_state = config["resample_jitter_state"].as<bool>();
  }

  if (config["resample_position_2d_jitter"]) {
    config_paras.resample_position_2d_jitter = config["resample_position_2d_jitter"].as<bool>();
  }

  if (config["position_2d_jitter_covariance"]) {
    config_paras.position_2d_jitter_covariance = CompactVectorToCovarianceMatrix(config["position_2d_jitter_covariance"].as<std::vector<double>>(), 2);
  }

  if (config["resample_yaw_jitter"]) {
    config_paras.resample_yaw_jitter = config["resample_yaw_jitter"].as<bool>();
  }

  if (config["yaw_jitter_std_rad"]) {
    config_paras.yaw_jitter_std_rad = config["yaw_jitter_std_rad"].as<double>();
  }

  if (config["geomagnetism_bias_jitter_std"]) {
    config_paras.geomagnetism_bias_jitter_std = config["geomagnetism_bias_jitter_std"].as<double>();
  }

  if (config["random_seed"]) {
    config_paras.random_seed = config["random_seed"].as<int>();
  }

  if (config["use_gt_gravity"]) {
    config_paras.use_gt_gravity = config["use_gt_gravity"].as<bool>();
  }

  if (config["use_gt_local_velocity"]) {
    config_paras.use_gt_local_velocity = config["use_gt_local_velocity"].as<bool>();
  }

  if (config["sample_local_velocity"]) {
    config_paras.sample_local_velocity = config["sample_local_velocity"].as<bool>();
  }

  if (config["use_gt_local_rotation"]) {
    config_paras.use_gt_local_rotation = config["use_gt_local_rotation"].as<bool>();
  }

  if (config["sample_local_rotation"]) {
    config_paras.sample_local_rotation = config["sample_local_rotation"].as<bool>();
  }

  if (config["use_gt_global_orientation"]) {
    config_paras.use_gt_global_orientation = config["use_gt_global_orientation"].as<bool>();
  }

  if (config["sample_global_orientation"]) {
    config_paras.sample_global_orientation = config["sample_global_orientation"].as<bool>();
  }

  if (config["translation_module_load_gt_orientation_instead_of_orientation_sensor"]) {
    config_paras.translation_module_load_gt_orientation_instead_of_oriention_sensor = config["translation_module_load_gt_orientation_instead_of_orientation_sensor"].as<bool>();
  }

  if (config["predict_translation_with_estimated_orientation"]) {
    config_paras.predict_translation_with_estimated_orientation = config["predict_translation_with_estimated_orientation"].as<bool>();
  }

  if (config["return_orientation_sensor_orientation"]) {
    config_paras.return_orientation_sensor_orientation = config["return_orientation_sensor_orientation"].as<bool>();
  }

  if (config["bluetooth_offset_estimation"]) {
    config_paras.bluetooth_offset_estimation = config["bluetooth_offset_estimation"].as<bool>();
  }

  if (config["wifi_offset_estimation"]) {
    config_paras.wifi_offset_estimation = config["wifi_offset_estimation"].as<bool>();
  }

  if (config["geomagnetism_bias_estimation"]) {
    config_paras.geomagnetism_bias_estimation = config["geomagnetism_bias_estimation"].as<bool>();
  }

  if (config["geomagnetism_bias_use_map_prior"]) {
    config_paras.geomagnetism_bias_use_map_prior = config["geomagnetism_bias_use_map_prior"].as<bool>();
  }

  if (config["geomagnetism_bias_use_exponential_averaging"]) {
    config_paras.geomagnetism_bias_use_exponential_averaging = config["geomagnetism_bias_use_exponential_averaging"].as<bool>();
  }

  if (config["use_orientation_sensor_constraint"]) {
    config_paras.use_orientation_sensor_constraint = config["use_orientation_sensor_constraint"].as<bool>();
  }

  if (config["orientation_sensor_constraint_abs_yaw_diff_rad"]) {
    config_paras.orientation_sensor_constraint_abs_yaw_diff_rad = config["orientation_sensor_constraint_abs_yaw_diff_rad"].as<double>();
  }

  if (config["use_orientation_geomagnetism_bias_correlated_jittering"]) {
    config_paras.use_orientation_geomagnetism_bias_correlated_jittering = config["use_orientation_geomagnetism_bias_correlated_jittering"].as<bool>();
  }

  if (config["synthesize_gt_local_velocity"]) {
    config_paras.synthesize_gt_local_velocity = config["synthesize_gt_local_velocity"].as<bool>();
  }

  if (config["synthesize_gt_local_rotation"]) {
    config_paras.synthesize_gt_local_rotation = config["synthesize_gt_local_rotation"].as<bool>();
  }

  if (config["synthesize_gt_global_orientation"]) {
    config_paras.synthesize_gt_global_orientation = config["synthesize_gt_global_orientation"].as<bool>();
  }

  if (config["use_relative_observation"]) {
    config_paras.use_relative_observation = config["use_relative_observation"].as<bool>();
  }

  if (config["use_dense_relative_observation"]) {
    config_paras.use_dense_relative_observation = config["use_dense_relative_observation"].as<bool>();
  }

  if (config["relative_observation_window_size"]) {
    config_paras.relative_observation_window_size = config["relative_observation_window_size"].as<int>();
  }

  if (config["dense_relative_observation_step_length"]) {
    config_paras.dense_relative_observation_step_length = config["dense_relative_observation_step_length"].as<int>();
  }

  if (config["time_profile_window_size"]) {
    config_paras.time_profile_window_size = config["time_profile_window_size"].as<int>();
  }

  if (config["output_folderpath"]) {
    config_paras.output_folderpath = config["output_folderpath"].as<std::string>();
  }

  if (config["smooth_start_index"]) {
    config_paras.smooth_start_index = config["smooth_start_index"].as<int>();
  }

  if (config["stability_evaluation"]) {
    config_paras.stability_evaluation = config["stability_evaluation"].as<bool>();
  }

  if (config["number_of_retries"]) {
    config_paras.number_of_retries = config["number_of_retries"].as<int>();
  }

  if (config["particle_statistics"]) {
    config_paras.particle_statistics = config["particle_statistics"].as<bool>();
  }

  if (config["number_of_samples_per_grid_for_traversing"]) {
    config_paras.number_of_samples_per_grid_for_traversing = config["number_of_samples_per_grid_for_traversing"].as<int>();
  }

  if (config["number_of_localization_steps"]) {
    config_paras.number_of_localization_steps = config["number_of_localization_steps"].as<int>();
  }

  if (config["use_map_covariance"]) {
    config_paras.use_map_covariance = config["use_map_covariance"].as<bool>();
  }

  if (config["number_of_initial_coarse_map_update_steps"]) {
    config_paras.number_of_initial_coarse_map_update_steps = config["number_of_initial_coarse_map_update_steps"].as<int>();
  }

  if (config["coarse_map_uncertainty_scale_factor"]) {
    config_paras.coarse_map_uncertainty_scale_factor = config["coarse_map_uncertainty_scale_factor"].as<double>();
  }

  if (config["particle_filter_local_resampling"]) {
    config_paras.particle_filter_local_resampling = config["particle_filter_local_resampling"].as<bool>();
  }

  if (config["particle_filter_local_resampling_region_size_in_meters"]) {
    config_paras.particle_filter_local_resampling_region_size_in_meters = config["particle_filter_local_resampling_region_size_in_meters"].as<double>();
  }

  if (config["use_separate_geomagnetism_coarse_map"]) {
    config_paras.use_separate_geomagnetism_coarse_map = config["use_separate_geomagnetism_coarse_map"].as<bool>();
  }

  if (config["use_prediction_filter"]) {
    config_paras.use_prediction_filter = config["use_prediction_filter"].as<bool>();
  }

  if (config["imu_sampling_frequency"]) {
    config_paras.imu_sampling_frequency = config["imu_sampling_frequency"].as<double>();
  }

  if (config["prediction_frequency"]) {
    config_paras.prediction_frequency = config["prediction_frequency"].as<double>();
  }

  if (config["prediction_imu_down_sampling_divider"]) {
    config_paras.prediction_imu_down_sampling_divider = config["prediction_imu_down_sampling_divider"].as<int>();
  }

  if (config["use_static_detection"]) {
    config_paras.use_static_detection = config["use_static_detection"].as<bool>();
  }

  if (config["use_constant_heading_velocity"]) {
    config_paras.use_constant_heading_velocity = config["use_constant_heading_velocity"].as<bool>();
  }

  if (config["use_second_order_constant_heading_velocity"]) {
    config_paras.use_second_order_constant_heading_velocity = config["use_second_order_constant_heading_velocity"].as<bool>();
  }

  if (config["use_pdr_heading_velocity"]) {
    config_paras.use_pdr_heading_velocity = config["use_pdr_heading_velocity"].as<bool>();
  }

  if (config["knn_localization_map_folderpath"]) {
    config_paras.knn_localization_map_folderpath = config["knn_localization_map_folderpath"].as<std::string>();
  }

  if (config["knn_localization_sensor"]) {
    config_paras.knn_localization_sensor = config["knn_localization_sensor"].as<std::string>();
  }

  this->config_paras_ = config_paras;

  return 1;
}

Configurator::Configurator(void) {
  this->yaml_config_path_ = "";
  ConfigParas config_paras;
  this->config_paras_ = config_paras;
}

Configurator::~Configurator() {}

}  // namespace configuration

}  // namespace state_estimation
