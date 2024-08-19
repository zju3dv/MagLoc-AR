/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-06-08 16:36:20
 * @LastEditTime: 2023-02-14 15:47:35
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#ifndef STATE_ESTIMATION_CONFIGURATION_CONFIGURATION_H_
#define STATE_ESTIMATION_CONFIGURATION_CONFIGURATION_H_

#include <Eigen/Eigen>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace state_estimation {

namespace configuration {

struct CameraIntrinsics {
    double width = 0.0;
    double height = 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    std::string Printf(void);
};

class WirelessPositioningConfigurator {
 public:
  int Init(const std::string& config_filepath);

  std::string config_filepath(void) {
    return this->config_filepath_;
  }

  bool offline(void) {
    return this->offline_;
  }

  std::string offline_request_path(void) {
    return this->offline_request_path_;
  }

  bool dump_data(void) {
    return this->dump_data_;
  }

  std::string dump_data_path(void) {
    return this->dump_data_path_;
  }

  double update_interval(void) {
    return this->update_interval_;
  }

  double excution_interval(void) {
    return this->excution_interval_;
  }

  bool running_prediction(void) {
    return this->running_prediction_;
  }

  bool running_INS(void) {
    return this->running_INS_;
  }

  bool using_gravity_sensor(void) {
    return this->using_gravity_sensor_;
  }

  bool running_IMU_rotation(void) {
    return this->running_IMU_rotation_;
  }

  WirelessPositioningConfigurator(void);
  ~WirelessPositioningConfigurator();

 private:
  std::string config_filepath_;
  bool offline_;
  std::string offline_request_path_;
  bool dump_data_;
  std::string dump_data_path_;
  double update_interval_;
  double excution_interval_;
  bool running_prediction_;
  bool running_INS_;
  bool using_gravity_sensor_;
  bool running_IMU_rotation_;
};

class CameraCalibrationConfigurator {
 public:
  int Init(const std::string& calib_filepath, int number_of_cameras = 1);

  std::string calib_filepath(void) {
    return this->calib_filepath_;
  }

  int number_of_cameras(void) {
    return this->number_of_cameras_;
  }

  CameraIntrinsics camera_intrinsics(void) {
    if (this->number_of_cameras_ > 0) {
      return this->camera_intrinsicses_.at(0);
    } else {
      CameraIntrinsics camera_intrinsics;
      return camera_intrinsics;
    }
  }

  Eigen::Matrix4d T_cam_imu(void) {
    if (this->number_of_cameras_ > 0) {
      return this->T_cam_imus_.at(0);
    } else {
      return Eigen::Matrix4d::Identity();
    }
  }

  Eigen::Matrix4d T_imu_cam(void) {
    if (this->number_of_cameras_ > 0) {
      return this->T_cam_imus_.at(0).inverse();
    } else {
      return Eigen::Matrix4d::Identity();
    }
  }

  std::string distortion_model(void) {
    if (this->number_of_cameras_ > 0) {
      return this->distortion_models_.at(0);
    } else {
      return "";
    }
  }

  Eigen::Matrix<double, 8, 1> distortion_coefficients(void) {
    if (this->number_of_cameras_ > 0) {
      return this->distortion_coefficientses_.at(0);
    } else {
      return Eigen::Matrix<double, 8, 1>::Zero();
    }
  }

  std::vector<CameraIntrinsics> camera_intrinsicses(void) {
    return this->camera_intrinsicses_;
  }

  std::vector<std::string> distortion_models(void) {
    return this->distortion_models_;
  }

  std::vector<Eigen::Matrix<double, 8, 1>> distortion_coefficientses(void) {
    return this->distortion_coefficientses_;
  }

  std::vector<Eigen::Matrix4d> T_cam_imus(void) {
    return this->T_cam_imus_;
  }

  std::vector<Eigen::Matrix4d> T_imu_cams(void) {
    std::vector<Eigen::Matrix4d> T_imu_cams;
    for (int i = 0; i < this->T_cam_imus_.size(); i++) {
      T_imu_cams.emplace_back(this->T_cam_imus_.at(i).inverse());
    }
    return T_imu_cams;
  }

  CameraCalibrationConfigurator(void);
  ~CameraCalibrationConfigurator();

 private:
  std::string calib_filepath_;
  int number_of_cameras_;
  std::vector<CameraIntrinsics> camera_intrinsicses_;
  std::vector<std::string> distortion_models_;
  std::vector<Eigen::Matrix<double, 8, 1>> distortion_coefficientses_;
  std::vector<Eigen::Matrix4d> T_cam_imus_;
};

struct ConfigParas {
  double map_spatial_interval = 1.0;
  double bluetooth_map_spatial_interval = 1.0;
  double wifi_map_spatial_interval = 1.0;
  double geomagnetism_map_spatial_interval = 1.0;
  double heatmap_smooth_factor = 1.0;
  double bluetooth_uncertainty_scale_factor = 1.0;
  double wifi_uncertainty_scale_factor = 1.0;
  double geomagnetism_uncertainty_scale_factor = 1.0;
  double geomagnetism_additive_noise_std = 0.0;
  int zero_centering = 0;
  double acc_std = 0.1;
  double yaw_std_degree = 30.0;
  double v_max = 2.0;
  double v_min = 0.0;
  double effective_population_ratio = 0.5;
  double population_expansion_ratio = 2.0;
  int n_init_particles = 100;
  int filter_state_memory_size = 10;
  int max_number_of_particles = 100000;
  int effective_number_of_features_threshold = 1;
  int threshold_of_effective_feature_number_for_zero_centering = 5;
  int nearest_k = 5;
  std::string distribution_map_path = "";
  int number_of_map_label_fields = 1;
  int number_of_map_feature_fields = 1;
  int number_of_map_features = 1;
  int bluetooth_zero_centering = 0;
  int wifi_zero_centering = 0;

  std::string bluetooth_distribution_map_path = "";
  std::string wifi_distribution_map_path = "";
  std::string geomagnetism_distribution_map_path = "";
  std::string geomagnetism_distribution_map_covariance_path = "";
  std::string geomagnetism_coarse_distribution_map_path = "";
  std::string geomagnetism_coarse_distribution_map_covariance_path = "";

  double v_heading_yaw_min = -2 * M_PI;
  double v_heading_yaw_max = 2 * M_PI;
  double v_orientation_angle_min = -2 * M_PI;
  double v_orientation_angle_max = 2 * M_PI;

  double bias_min = -10;
  double bias_max = 10;
  double v_bias_min = -5;
  double v_bias_max = 5;

  std::vector<double> control_input_means_vector;
  std::vector<double> control_input_covariances_vector;

  std::string client_trace_path = "";
  std::string R_mw_path = "";
  std::vector<double> R_mw_vector{1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0};
  std::string model_trajectory_path = "";
  std::string output_path = "";

  double client_buffer_duration = 3.0;

  std::string request_wifi_fingerprints_path = "";
  std::string request_num_effective_wifi_fingerprints_path = "";
  std::string request_bluetooth_fingerprints_path = "";
  std::string request_num_effective_bluetooth_fingerprints_path = "";
  std::string request_geomag_fingerprints_path = "";
  std::string request_num_effective_geomag_fingerprints_path = "";

  std::string sensor_weights_path = "";
  std::unordered_map<std::string, double> sensor_weights = {std::pair<std::string, double>{"bluetooth", 1.0},
                                                            std::pair<std::string, double>{"wifi", 1.0},
                                                            std::pair<std::string, double>{"mf", 1.0}};

  std::string results_path = "";

  int geomagnetism_feature_type = 2;

  double time_unit_in_second = 1.0;

  double orientation_sensor_yaw_variance_in_rad = 0.0;
  double bluetooth_max_absolute_offset = 0.0;
  double wifi_max_absolute_offset = 0.0;
  double geomagnetism_max_absolute_bias = 0.0;

  double bluetooth_offset_variance = 0.0;
  double wifi_offset_variance = 0.0;
  double geomagnetism_bias_variance = 0.0;

  Eigen::Matrix3d rotation_covariance = Eigen::Matrix3d::Zero();

  Eigen::Matrix3d INS_v_local_covariance = Eigen::Matrix3d::Zero();

  Eigen::Matrix3d geomagnetism_s_covariance = Eigen::Matrix3d::Zero();

  Eigen::Matrix3d gravity_s_covariance = Eigen::Matrix3d::Zero();

  bool resample_jitter_state = false;
  bool resample_position_2d_jitter = false;
  Eigen::Matrix2d position_2d_jitter_covariance = Eigen::Matrix2d::Zero();
  bool resample_yaw_jitter = false;
  double yaw_jitter_std_rad = 0.0;

  double geomagnetism_bias_jitter_std = 0.0;

  int random_seed = 2020;

  bool use_gt_gravity = false;
  bool use_gt_local_velocity = false;
  bool sample_local_velocity = true;
  bool use_gt_local_rotation = false;
  bool sample_local_rotation = true;
  bool use_gt_global_orientation = false;
  bool sample_global_orientation = true;
  bool translation_module_load_gt_orientation_instead_of_oriention_sensor = false;
  bool predict_translation_with_estimated_orientation = true;
  bool return_orientation_sensor_orientation = false;
  bool bluetooth_offset_estimation = false;
  bool wifi_offset_estimation = false;
  bool geomagnetism_bias_estimation = false;

  bool geomagnetism_bias_use_map_prior = false;

  bool geomagnetism_bias_use_exponential_averaging = false;

  bool use_orientation_sensor_constraint = false;

  double orientation_sensor_constraint_abs_yaw_diff_rad = 1.0;

  bool use_orientation_geomagnetism_bias_correlated_jittering = false;

  bool synthesize_gt_local_velocity = false;
  bool synthesize_gt_local_rotation = false;
  bool synthesize_gt_global_orientation = false;

  bool use_relative_observation = true;
  bool use_dense_relative_observation = false;
  int dense_relative_observation_step_length = 1;
  int relative_observation_window_size = 20;

  int time_profile_window_size = 1;

  std::string output_folderpath = "";

  int smooth_start_index = 0;

  bool stability_evaluation = false;

  int number_of_retries = 1;

  bool particle_statistics = false;

  int number_of_samples_per_grid_for_traversing = 5;

  int number_of_localization_steps = 0;

  bool use_map_covariance = false;

  int number_of_initial_coarse_map_update_steps = 0;

  double coarse_map_uncertainty_scale_factor = 8.0;

  bool particle_filter_local_resampling = false;

  double particle_filter_local_resampling_region_size_in_meters = 10.0;

  bool use_separate_geomagnetism_coarse_map = false;

  bool use_prediction_filter = false;

  double imu_sampling_frequency = 500.0;

  double prediction_frequency = 30.0;

  int prediction_imu_down_sampling_divider = 1;

  bool use_static_detection = false;

  bool use_constant_heading_velocity = false;

  bool use_second_order_constant_heading_velocity = false;

  bool use_pdr_heading_velocity = false;

  std::string knn_localization_map_folderpath = "";
  std::string knn_localization_sensor = "";

  void Justify(void);
  std::string Printf(void);
};

class Configurator {
 public:
  int Init(std::string yaml_config_path);

  ConfigParas config_paras(void) {
    return this->config_paras_;
  }

  std::string config_path(void) {
    return this->yaml_config_path_;
  }

  Configurator(void);
  ~Configurator();

 private:
  std::string yaml_config_path_;
  ConfigParas config_paras_;
};

}  // namespace configuration

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_CONFIGURATION_CONFIGURATION_H_
