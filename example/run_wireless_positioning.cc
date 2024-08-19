/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2022-01-10 17:10:13
 * @LastEditTime: 2023-04-10 15:26:18
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <filesystem>
#include <fstream>

#include "configuration/configuration.h"
#include "observation_model/imu_observation_model.h"
#include "observation_model/orientation_observation_model.h"
#include "offline/client_data_reader.h"
#include "util/client_request.h"
#include "util/result_format.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_range_sampler.h"
#include "WirelessPositioningMultiplexingAgentWoCovMapPrior.h"
#include "offline/results_reader.h"

namespace state_estimation {

// add definition of two structures -- 1. result structure  2. gt structure
struct TestSummaryResult {
    util::Result wpa_result;
    Eigen::Vector3d ins_position_under_gt_orientation = Eigen::Vector3d::Zero();
    Eigen::Vector3d ins_position_under_est_orientation = Eigen::Vector3d::Zero();
    Eigen::Vector3d ins_position_under_orientation_sensor = Eigen::Vector3d::Zero();
    Eigen::Vector3d gt_position_under_gt_orientation_and_gt_velocity = Eigen::Vector3d::Zero();
    double yaw_m_imu = 0.0;
    double yaw_m_orientation = 0.0;
    configuration::ConfigParas config_paras;
    double gt_bluetooth_offset = 0.0;
    double gt_wifi_offset = 0.0;
    Eigen::Vector3d gt_geomagnetism_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d dq_log_error = Eigen::Vector3d::Zero();
    Eigen::Vector3d INS_v_local = Eigen::Vector3d::Zero();
    Eigen::Vector3d gt_local_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d global_R_orientation_sensor_log_error = Eigen::Vector3d::Zero();
    Eigen::Vector3d global_R_geomag_log_error = Eigen::Vector3d::Zero();
    double gt_state_log_probability = 0.0;

    std::string Output(void) {
      std::stringstream ss;
      ss.setf(std::ios::scientific, std::ios::floatfield);
      ss.precision(6);
      ss << wpa_result.OutputGTEstFormatTUM() << ","
         << wpa_result.distance_variance() << ","
         << ins_position_under_gt_orientation(0) << "," << ins_position_under_gt_orientation(1) << ","
         << ins_position_under_orientation_sensor(0) << "," << ins_position_under_orientation_sensor(1) << ","
         << gt_position_under_gt_orientation_and_gt_velocity(0) << "," << gt_position_under_gt_orientation_and_gt_velocity(1) << ","
         << yaw_m_imu << ","
         << yaw_m_orientation << ","
         << config_paras.orientation_sensor_yaw_variance_in_rad << ","
         << gt_bluetooth_offset << ","
         << wpa_result.est_bluetooth_offset() << ","
         << config_paras.bluetooth_offset_variance << ","
         << gt_wifi_offset << ","
         << wpa_result.est_wifi_offset() << ","
         << config_paras.wifi_offset_variance << ","
         << gt_geomagnetism_bias(0) << "," << gt_geomagnetism_bias(1) << "," << gt_geomagnetism_bias(2) << ","
         << wpa_result.est_geomagnetism_bias_3d()(0) << ","
         << wpa_result.est_geomagnetism_bias_3d()(1) << ","
         << wpa_result.est_geomagnetism_bias_3d()(2) << ","
         << config_paras.geomagnetism_bias_variance << ","
         << dq_log_error(0) << "," << dq_log_error(1) << "," << dq_log_error(2) << ","
         << config_paras.rotation_covariance(0, 0) << ","
         << config_paras.rotation_covariance(0, 1) << ","
         << config_paras.rotation_covariance(0, 2) << ","
         << config_paras.rotation_covariance(1, 0) << ","
         << config_paras.rotation_covariance(1, 1) << ","
         << config_paras.rotation_covariance(1, 2) << ","
         << config_paras.rotation_covariance(2, 0) << ","
         << config_paras.rotation_covariance(2, 1) << ","
         << config_paras.rotation_covariance(2, 2) << ","
         << INS_v_local(0) << "," << INS_v_local(1) << "," << INS_v_local(2) << ","
         << gt_local_velocity(0) << "," << gt_local_velocity(1) << "," << gt_local_velocity(2) << ","
         << config_paras.INS_v_local_covariance(0, 0) << ","
         << config_paras.INS_v_local_covariance(0, 1) << ","
         << config_paras.INS_v_local_covariance(0, 2) << ","
         << config_paras.INS_v_local_covariance(1, 0) << ","
         << config_paras.INS_v_local_covariance(1, 1) << ","
         << config_paras.INS_v_local_covariance(1, 2) << ","
         << config_paras.INS_v_local_covariance(2, 0) << ","
         << config_paras.INS_v_local_covariance(2, 1) << ","
         << config_paras.INS_v_local_covariance(2, 2) << ","
         << ins_position_under_est_orientation(0) << "," << ins_position_under_est_orientation(1) << ","
         << global_R_orientation_sensor_log_error(0) << "," << global_R_orientation_sensor_log_error(1) << "," << global_R_orientation_sensor_log_error(2) << ","
         << global_R_geomag_log_error(0) << "," << global_R_geomag_log_error(1) << "," << global_R_geomag_log_error(2) << ","
         << config_paras.geomagnetism_bias_variance << ","
         << gt_state_log_probability << ","
         << wpa_result.log_probability_max() << ","
         << wpa_result.log_probability_min() << ","
         << wpa_result.log_probability_top_20_percent() << ","
         << wpa_result.log_probability_top_50_percent() << ","
         << wpa_result.log_probability_top_80_percent() << ","
         << static_cast<int>(wpa_result.activity_type());
      return ss.str();
    }
};

Eigen::Matrix3d jitter_gt_orientation(util::ClientRequest &client_request, double jitter_yaw) {
  Eigen::AngleAxisd angleaxis_wsg(jitter_yaw, Eigen::Vector3d({0.0, 0.0, 1.0}));
  Eigen::Quaterniond q_wsg(angleaxis_wsg);
  Eigen::Matrix3d geomagnetism_s_trans = (client_request.gt_orientation.conjugate() * q_wsg.conjugate() * client_request.gt_orientation).toRotationMatrix();
  client_request.gt_orientation = q_wsg.normalized() * client_request.gt_orientation.normalized();
  return geomagnetism_s_trans;
}

void test_wireless_positioning_multiplexing_agent(configuration::ConfigParas config_paras, bool need_evaluation, bool print_verbose, DiscretizationResolutions resolutions, bool need_estimation, std::string client_trace_path, std::string output_folderpath, std::vector<util::Result>* sequential_results_ptr = nullptr, bool track_gt_particle = false, bool dump_online_filter_states = false, int skip_request_num = 0, int remain_request_num = 0, std::string gt_euroc_filepath = "", std::string pose_prior_euroc_filepath = "") {
  config_paras.Justify();
  if (need_evaluation) {
    config_paras.filter_state_memory_size = 3000;
  } else {
    config_paras.filter_state_memory_size = 1;
  }

  std::filesystem::create_directories(output_folderpath);
  std::string online_result_filepath = std::filesystem::path(output_folderpath) / "online_result.csv";
  std::string online_result_euroc_filepath = std::filesystem::path(output_folderpath) / "online_result_euroc.csv";
  // online_prediction_result: generate predicted estimates with time aligned with the gt pose.
  std::string online_prediction_result_euroc_filepath = std::filesystem::path(output_folderpath) / "online_prediction_result_euroc.csv";
  // online_highrate_result: generate estimates only have time aligned with the gt pose but update in the pace of client requests.
  std::string online_highrate_result_euroc_filepath = std::filesystem::path(output_folderpath) / "online_highrate_result_euroc.csv";
  std::string smooth_result_filepath = std::filesystem::path(output_folderpath) / "smooth_result.csv";
  std::string config_filepath = std::filesystem::path(output_folderpath) / "config.txt";
  std::string evaluation_result_filepath = std::filesystem::path(output_folderpath) / "evaluation_result.txt";
  std::string online_filter_states_dump_filepath = std::filesystem::path(output_folderpath) / "online_filter_states.dat";
  std::ofstream online_result_stream;
  std::ofstream online_result_euroc_stream;
  std::ofstream online_prediction_result_euroc_stream;
  std::ofstream online_highrate_result_euroc_stream;
  std::ofstream smooth_result_stream;
  std::ofstream config_stream;
  std::ofstream evaluation_result_stream;
  std::ofstream online_filter_states_dump_stream;
  online_result_stream.open(online_result_filepath);
  online_result_euroc_stream.open(online_result_euroc_filepath);
  online_prediction_result_euroc_stream.open(online_prediction_result_euroc_filepath);
  online_highrate_result_euroc_stream.open(online_highrate_result_euroc_filepath);
  smooth_result_stream.open(smooth_result_filepath);
  config_stream.open(config_filepath);
  evaluation_result_stream.open(evaluation_result_filepath);
  online_filter_states_dump_stream.open(online_filter_states_dump_filepath, std::ofstream::binary | std::ofstream::app);

  std::string config_string = config_paras.Printf();
  config_stream << config_string;
  config_stream.close();

  WirelessPositioningAgent wireless_positioning_agent;
  wireless_positioning_agent.Init(config_paras);

  wireless_positioning_agent.resolutions(resolutions);

  if (track_gt_particle) {
    wireless_positioning_agent.include_ideal_prediction(true);
  } else {
    wireless_positioning_agent.include_ideal_prediction(false);
  }

  offline::ClientDataReader client_data_reader;
  client_data_reader.Init(client_trace_path);
  client_data_reader.SortRequestFolderPaths();

  offline::EuRoCResultReader gt_data_reader;
  if (gt_euroc_filepath.size() > 0) {
    gt_data_reader.Init(gt_euroc_filepath);
  }

  offline::EuRoCResultReader pose_prior_reader;
  if (pose_prior_euroc_filepath.size() > 0) {
    pose_prior_reader.Init(pose_prior_euroc_filepath);
  }

  Eigen::Matrix3d R_mw = VectorTo2DMatrixC(config_paras.R_mw_vector, 3);

  sampler::MultivariateGaussianSampler mvg_sampler;
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double> uniform_range_sampler;
#ifdef RANDOMNESS_OFF
  mvg_sampler.Seed(config_paras.random_seed);
  uniform_range_sampler.Seed(config_paras.random_seed);
#endif

  int request_counter = 0;
  double last_update_time = 0.0;
  std::vector<prediction_model::CompoundPredictionModelState> gt_states;
  double last_time = 0.0;
  double ins_x_gt_orientation = 1e9;
  double ins_y_gt_orientation = 1e9;
  double ins_x_est_orientation = 1e9;
  double ins_y_est_orientation = 1e9;
  double ins_x_orientation_sensor = 1e9;
  double ins_y_orientation_sensor = 1e9;
  double gt_x_gt_orientation = 1e9;
  double gt_y_gt_orientation = 1e9;
  std::vector<double> yaw_m_imus;
  std::vector<double> yaw_m_orientations;
  std::vector<double> ins_x_gt_orientations;
  std::vector<double> ins_y_gt_orientations;
  std::vector<double> ins_x_orientation_sensors;
  std::vector<double> ins_y_orientation_sensors;
  std::vector<double> ins_x_est_orientations;
  std::vector<double> ins_y_est_orientations;
  std::vector<double> gt_x_gt_orientations;
  std::vector<double> gt_y_gt_orientations;
  std::vector<Eigen::Vector3d> dq_log_errors;
  std::vector<Eigen::Vector3d> ins_local_vs;
  std::vector<Eigen::Vector3d> gt_local_vs;
  std::vector<Eigen::Vector3d> global_R_orientation_sensor_log_errors;
  std::vector<Eigen::Vector3d> global_R_geomag_log_errors;
  std::vector<double> gt_bluetooth_offsets;
  std::vector<double> gt_wifi_offsets;
  std::vector<Eigen::Vector3d> gt_geomagnetism_biases;
  Eigen::Quaterniond q_ms_gt_pre = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond q_ws_imu_pre = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond q_ms_gt_synthesize_pre = Eigen::Quaterniond::Identity();
  if (!print_verbose) {
    std::cout << "Filtering forward:" << std::endl;
  }

  util::ClientRequest next_client_request;
  while (!client_data_reader.IsEmpty()) {
    request_counter += 1;
    util::ClientRequest temp_client_request = client_data_reader.GetNextRequest();
    if (request_counter <= skip_request_num) {
      continue;
    }
    next_client_request = temp_client_request;
    break;
  }
  offline::EuRoCResult next_gt_result;
  if (!gt_data_reader.IsEmpty()) {
    next_gt_result = gt_data_reader.GetNextResult();
  }
  offline::EuRoCResult next_pose_prior_result_before;
  offline::EuRoCResult next_pose_prior_result_after;
  if (!pose_prior_reader.IsEmpty()) {
    next_pose_prior_result_before = pose_prior_reader.GetNextResult();
  }
  if (!pose_prior_reader.IsEmpty()) {
    next_pose_prior_result_after = pose_prior_reader.GetNextResult();
  }

  Eigen::Vector3d z_vector = {0.0, 0.0, 1.0};

  util::Result previous_request_result;
  while (next_client_request.timestamp > 0.0 || next_gt_result.timestamp > 0.0) {
    util::Result result;
    if (next_client_request.timestamp < 0.0 || (next_client_request.timestamp > next_gt_result.timestamp && next_gt_result.timestamp > 0.0)) {
      // the next timestamp is next_gt_result.timestamp
      result = wireless_positioning_agent.GetPredictionResultInTheMiddle(next_gt_result.timestamp, 1, next_client_request.gyroscope_data, next_client_request.accelerometer_data);
      result = wireless_positioning_agent.GetPredictionResultInTheMiddle(next_gt_result.timestamp, 2, next_client_request.gyroscope_data, next_client_request.accelerometer_data);

      variable::Position temp_gt_position;
      temp_gt_position.x(next_gt_result.p(0));
      temp_gt_position.y(next_gt_result.p(1));
      result.gt_position(temp_gt_position);

      Eigen::AngleAxisd temp_angle_axis_msg_gt(CalculateRzFromOrientation(next_gt_result.q));
      double temp_yaw_m_gt = GetAngleByAxisFromAngleAxis(temp_angle_axis_msg_gt, z_vector);
      result.gt_yaw(temp_yaw_m_gt);

      online_prediction_result_euroc_stream << result.OutputEstFormatEuRoC() << std::endl;

      // online high-rate result
      result = previous_request_result;
      result.gt_position(temp_gt_position);
      result.gt_yaw(temp_yaw_m_gt);

      online_highrate_result_euroc_stream << result.OutputEstFormatEuRoC() << std::endl;

      if (!gt_data_reader.IsEmpty()) {
        next_gt_result = gt_data_reader.GetNextResult();
      } else {
        next_gt_result.timestamp = -1.0;
      }
      continue;
    }

    // the next timestamp is next_client_request.timestamp

    // get the most recent pose prior result
    while (next_pose_prior_result_after.timestamp < next_client_request.timestamp && !pose_prior_reader.IsEmpty()) {
      next_pose_prior_result_before = next_pose_prior_result_after;
      next_pose_prior_result_after = pose_prior_reader.GetNextResult();
    }
    offline::EuRoCResult most_recent_pose_prior_result;
    if (next_client_request.timestamp <= next_pose_prior_result_before.timestamp) {
      most_recent_pose_prior_result = next_pose_prior_result_before;
    } else if (next_client_request.timestamp <= next_pose_prior_result_after.timestamp) {
      double time_diff_before = next_client_request.timestamp - next_pose_prior_result_before.timestamp;
      double time_diff_after = next_pose_prior_result_after.timestamp - next_client_request.timestamp;
      if (time_diff_before >= time_diff_after) {
        most_recent_pose_prior_result = next_pose_prior_result_after;
      } else {
        most_recent_pose_prior_result = next_pose_prior_result_before;
      }
    } else {
      most_recent_pose_prior_result = next_pose_prior_result_after;
    }

    if (std::abs(next_client_request.timestamp - most_recent_pose_prior_result.timestamp) < 0.05) {
      next_client_request.position_prior = most_recent_pose_prior_result.p;
      next_client_request.orientation_prior = most_recent_pose_prior_result.q;
      next_client_request.position_prior_valid = true;
      next_client_request.orientation_prior_valid = true;
    }

    // get gt state
    Eigen::AngleAxisd angle_axis_msg_gt(CalculateRzFromOrientation(next_client_request.gt_orientation));
    double yaw_m_gt = GetAngleByAxisFromAngleAxis(angle_axis_msg_gt, z_vector);
    Eigen::AngleAxisd angle_axis_wsg_gt(CalculateRzFromOrientation(Eigen::Quaterniond(R_mw.transpose() * next_client_request.gt_orientation)));
    double yaw_w_gt = GetAngleByAxisFromAngleAxis(angle_axis_wsg_gt, z_vector);

    // get dR from imu and from gt, and calculate the log error
    Eigen::Quaterniond dq_imu  = next_client_request.imu_pose_ws * q_ws_imu_pre.conjugate();
    // std::cout << "main: dR: " << std::endl;
    // std::cout << dq_imu.toRotationMatrix() << std::endl;
    Eigen::Quaterniond q_mw(R_mw);
    q_mw.normalize();
    Eigen::Quaterniond dq_gt = (q_mw.conjugate() * next_client_request.gt_orientation) * (q_mw.conjugate() * q_ms_gt_pre).conjugate();
    Eigen::AngleAxisd dq_log_error_angleaxis(dq_gt * dq_imu.conjugate());
    Eigen::Vector3d dq_log_error = dq_log_error_angleaxis.angle() * dq_log_error_angleaxis.axis();

    // get global orientation error from gt and orientation sensor, and calculate the log error
    Eigen::Matrix3d global_R_orientation_sensor_error = next_client_request.gt_orientation * (R_mw * next_client_request.orientation_sensor_pose_ws).transpose();
    Eigen::AngleAxisd global_R_orientation_sensor_error_angleaxis(global_R_orientation_sensor_error);
    Eigen::Vector3d global_R_orientation_sensor_log_error = global_R_orientation_sensor_error_angleaxis.angle() * global_R_orientation_sensor_error_angleaxis.axis();

    // get global orientation error from gravity and geomagnetism, and calculate the log error
    observation_model::OrientationObservation orientation_observation;
    orientation_observation.GetObservationFromGravityAndGeomagnetismLines(next_client_request.gravity_lines, next_client_request.geomagnetism_lines, next_client_request.timestamp / config_paras.time_unit_in_second);
    Eigen::Matrix3d global_R_geomag_error = next_client_request.gt_orientation * (R_mw * orientation_observation.q()).transpose();
    Eigen::AngleAxisd global_R_geomag_error_angleaxis(global_R_geomag_error);
    Eigen::Vector3d global_R_geomag_log_error = global_R_geomag_error_angleaxis.angle() * global_R_geomag_error_angleaxis.axis();

    prediction_model::CompoundPredictionModelState gt_state;
    gt_state.position(next_client_request.gt_position);
    gt_state.yaw(yaw_w_gt);
    gt_states.push_back(gt_state);

    // get yaw_m_imu
    Eigen::AngleAxisd angle_axis_msg_imu(CalculateRzFromOrientation(Eigen::Quaterniond(R_mw * next_client_request.imu_pose_ws)));
    double yaw_m_imu = GetAngleByAxisFromAngleAxis(angle_axis_msg_imu, z_vector);

    // get yaw_m_orientation
    Eigen::AngleAxisd angle_axis_msg_orientation(CalculateRzFromOrientation(Eigen::Quaterniond(R_mw * next_client_request.orientation_sensor_pose_ws)));
    double yaw_m_orientation = GetAngleByAxisFromAngleAxis(angle_axis_msg_orientation, z_vector);

    // create synthesized client_request
    util::ClientRequest synthesized_client_request = next_client_request;
    if (config_paras.synthesize_gt_local_velocity) {
      Eigen::Vector3d gt_velocity_local = next_client_request.gt_orientation.conjugate() * next_client_request.gt_velocity;
      mvg_sampler.SetParams(Eigen::Vector3d::Zero(), config_paras.INS_v_local_covariance);
      Eigen::Vector3d gt_velocity_local_jittered = gt_velocity_local + mvg_sampler.Sample();
      synthesized_client_request.gt_velocity = next_client_request.gt_orientation * gt_velocity_local_jittered;
    }
    if (config_paras.synthesize_gt_local_rotation) {
      if (wireless_positioning_agent.is_initialized()) {
        mvg_sampler.SetParams(Eigen::Vector3d::Zero(), config_paras.rotation_covariance);
        Eigen::Vector3d dq_gt_jitter_sample = mvg_sampler.Sample();
        double dq_gt_jitter_angle = dq_gt_jitter_sample.norm();
        Eigen::Vector3d dq_gt_jitter_axis;
        if (std::abs(dq_gt_jitter_angle) >= 1e-10 ) {
          dq_gt_jitter_axis = dq_gt_jitter_sample / dq_gt_jitter_angle;
        } else {
          dq_gt_jitter_axis = Eigen::Vector3d(0.0, 0.0, 1.0);
        }
        Eigen::Quaterniond dq_gt_jitter = Eigen::Quaterniond(Eigen::AngleAxisd(dq_gt_jitter_angle, dq_gt_jitter_axis));
        dq_gt.normalize();
        synthesized_client_request.gt_orientation = dq_gt_jitter * dq_gt * q_ms_gt_synthesize_pre;
      }
    }
    if (config_paras.synthesize_gt_global_orientation) {
      if (!wireless_positioning_agent.is_initialized()) {
        uniform_range_sampler.Init(-config_paras.orientation_sensor_yaw_variance_in_rad, config_paras.orientation_sensor_yaw_variance_in_rad);
        double sample_yaw_jitter = uniform_range_sampler.Sample();
        Eigen::Quaterniond q_global_orientation_jitter(Eigen::AngleAxisd(sample_yaw_jitter, Eigen::Vector3d(0.0, 0.0, 1.0)));
        synthesized_client_request.gt_orientation = q_global_orientation_jitter * next_client_request.gt_orientation;
      }
    }

    q_ws_imu_pre = next_client_request.imu_pose_ws.normalized();
    q_ms_gt_pre = next_client_request.gt_orientation.normalized();
    q_ms_gt_synthesize_pre = synthesized_client_request.gt_orientation.normalized();

    if (need_estimation) {
      if (next_client_request.timestamp - last_update_time > 0.5) {
        result = wireless_positioning_agent.GetResult(synthesized_client_request, true, synthesized_client_request.position_prior_valid, synthesized_client_request.position_prior);
        last_update_time = next_client_request.timestamp;
      } else {
        result = wireless_positioning_agent.GetResult(synthesized_client_request, false);
      }
    }

    if ((request_counter == 1) && track_gt_particle) {
      prediction_model::CompoundPredictionModelState specified_state = wireless_positioning_agent.GetSpecifiedParticleState(0);
      specified_state.position(next_client_request.gt_position);
      specified_state.yaw(yaw_w_gt);
      wireless_positioning_agent.InjectSpecifiedState(specified_state, 0);
    }

    if (ins_x_gt_orientation > 1e5) {
      ins_x_gt_orientation = next_client_request.gt_position.x();
      ins_y_gt_orientation = next_client_request.gt_position.y();
      ins_x_orientation_sensor = next_client_request.gt_position.x();
      ins_y_orientation_sensor = next_client_request.gt_position.y();
      gt_x_gt_orientation = next_client_request.gt_position.x();
      gt_y_gt_orientation = next_client_request.gt_position.y();
      ins_x_est_orientation = next_client_request.gt_position.x();
      ins_y_est_orientation = next_client_request.gt_position.y();
    } else {
      Eigen::Vector3d INS_v_m_gt = next_client_request.gt_orientation * next_client_request.INS_v_local;
      Eigen::Vector3d INS_v_m_orientation = R_mw * next_client_request.orientation_sensor_pose_ws * next_client_request.INS_v_local;
      ins_x_gt_orientation += INS_v_m_gt(0) * (next_client_request.timestamp - last_time);
      ins_y_gt_orientation += INS_v_m_gt(1) * (next_client_request.timestamp - last_time);
      ins_x_orientation_sensor += INS_v_m_orientation(0) * (next_client_request.timestamp - last_time);
      ins_y_orientation_sensor += INS_v_m_orientation(1) * (next_client_request.timestamp - last_time);
      gt_x_gt_orientation += next_client_request.gt_velocity(0) * (next_client_request.timestamp - last_time);
      gt_y_gt_orientation += next_client_request.gt_velocity(1) * (next_client_request.timestamp - last_time);
      Eigen::Matrix3d R_sgs_sensor = Eigen::Quaterniond::FromTwoVectors(next_client_request.gravity_s, z_vector).toRotationMatrix();
      Eigen::AngleAxisd angleaxis_msg_est(result.est_yaw(), z_vector);
      Eigen::Matrix3d R_ms_est = angleaxis_msg_est * R_sgs_sensor;
      ins_x_est_orientation += (R_ms_est * next_client_request.INS_v_local)(0) * (next_client_request.timestamp - last_time);
      ins_y_est_orientation += (R_ms_est * next_client_request.INS_v_local)(1) * (next_client_request.timestamp - last_time);
    }
    last_time = next_client_request.timestamp;

    result.timestamp(next_client_request.timestamp);
    result.gt_position(next_client_request.gt_position);
    variable::Orientation gt_orientation;
    gt_orientation.q(next_client_request.gt_orientation);
    result.gt_orientation(gt_orientation);
    result.gt_yaw(yaw_m_gt);

    Eigen::Vector3d gt_local_velocity = next_client_request.gt_orientation.conjugate() * next_client_request.gt_velocity;

    double gt_bluetooth_offset = wireless_positioning_agent.GetBluetoothOffsetFromMap(next_client_request, next_client_request.gt_position);
    double gt_wifi_offset = wireless_positioning_agent.GetWifiOffsetFromMap(next_client_request, next_client_request.gt_position);
    Eigen::Vector3d gt_geomagnetism_bias = wireless_positioning_agent.GetGeomagnetismBiasFromMap(next_client_request, next_client_request.gt_position, next_client_request.gt_orientation);

    if (sequential_results_ptr) {
      sequential_results_ptr->push_back(result);
    }

    TestSummaryResult test_summary_result;
    test_summary_result.wpa_result = result;
    test_summary_result.ins_position_under_gt_orientation(0) = ins_x_gt_orientation;
    test_summary_result.ins_position_under_gt_orientation(1) = ins_y_gt_orientation;
    test_summary_result.ins_position_under_orientation_sensor(0) = ins_x_orientation_sensor;
    test_summary_result.ins_position_under_orientation_sensor(1) = ins_y_orientation_sensor;
    test_summary_result.gt_position_under_gt_orientation_and_gt_velocity(0) = gt_x_gt_orientation;
    test_summary_result.gt_position_under_gt_orientation_and_gt_velocity(1) = gt_y_gt_orientation;
    test_summary_result.yaw_m_imu = yaw_m_imu;
    test_summary_result.yaw_m_orientation = yaw_m_orientation;
    test_summary_result.config_paras = config_paras;
    test_summary_result.gt_bluetooth_offset = gt_bluetooth_offset;
    test_summary_result.gt_wifi_offset = gt_wifi_offset;
    test_summary_result.gt_geomagnetism_bias = gt_geomagnetism_bias;
    test_summary_result.dq_log_error = dq_log_error;
    test_summary_result.INS_v_local = next_client_request.INS_v_local;
    test_summary_result.gt_local_velocity = gt_local_velocity;
    test_summary_result.ins_position_under_est_orientation(0) = ins_x_est_orientation;
    test_summary_result.ins_position_under_est_orientation(1) = ins_y_est_orientation;
    test_summary_result.global_R_orientation_sensor_log_error = global_R_orientation_sensor_log_error;
    test_summary_result.global_R_geomag_log_error = global_R_geomag_log_error;
    if (track_gt_particle) {
      prediction_model::CompoundPredictionModelState gt_state_in_pf = wireless_positioning_agent.GetSpecifiedParticleState(0);
      test_summary_result.gt_state_log_probability = gt_state_in_pf.state_log_probability();
    } else {
      test_summary_result.gt_state_log_probability = 0.0;
    }

    online_result_stream << test_summary_result.Output() << std::endl;
    online_result_euroc_stream << result.OutputEstFormatEuRoC() << std::endl;

    previous_request_result = result;

    if (!print_verbose) {
      std::cout << ".";
      std::cout.flush();
    } else {
      std::cout << test_summary_result.Output() << std::endl;
    }

    if (dump_online_filter_states) {
      wireless_positioning_agent.DumpParticleFilterState(online_filter_states_dump_stream);
    }

    yaw_m_imus.push_back(yaw_m_imu);
    yaw_m_orientations.push_back(yaw_m_orientation);
    ins_x_gt_orientations.push_back(ins_x_gt_orientation);
    ins_y_gt_orientations.push_back(ins_y_gt_orientation);
    ins_x_orientation_sensors.push_back(ins_x_orientation_sensor);
    ins_y_orientation_sensors.push_back(ins_y_orientation_sensor);
    gt_x_gt_orientations.push_back(gt_x_gt_orientation);
    gt_y_gt_orientations.push_back(gt_y_gt_orientation);
    dq_log_errors.push_back(dq_log_error);
    ins_local_vs.push_back(next_client_request.INS_v_local);
    gt_local_vs.push_back(gt_local_velocity);
    gt_bluetooth_offsets.push_back(gt_bluetooth_offset);
    gt_wifi_offsets.push_back(gt_wifi_offset);
    gt_geomagnetism_biases.push_back(gt_geomagnetism_bias);
    ins_x_est_orientations.push_back(ins_x_est_orientation);
    ins_y_est_orientations.push_back(ins_y_est_orientation);
    global_R_orientation_sensor_log_errors.push_back(global_R_orientation_sensor_log_error);
    global_R_geomag_log_errors.push_back(global_R_geomag_log_error);

    if (!client_data_reader.IsEmpty() && (client_data_reader.GetCurrentSize() > remain_request_num)) {
      request_counter += 1;
      next_client_request = client_data_reader.GetNextRequest();
    } else {
      next_client_request.timestamp = -1.0;
      next_client_request.accelerometer_data.clear();
      next_client_request.gyroscope_data.clear();
    }
  }
  if (!print_verbose) {
    std::cout << std::endl;
  }

  online_result_stream.close();
  online_result_euroc_stream.close();
  online_prediction_result_euroc_stream.close();
  online_highrate_result_euroc_stream.close();
  online_filter_states_dump_stream.close();

  if (need_evaluation) {
    EvaluationResult evaluation_result = wireless_positioning_agent.Evaluate(gt_states, config_paras.smooth_start_index);

    evaluation_result_stream << "gt_prediction_log_likelihood: " << evaluation_result.gt_prediction_log_likelihood << std::endl;
    evaluation_result_stream << "est_prediction_log_likelihood: " << evaluation_result.est_prediction_log_likelihood << std::endl;
    evaluation_result_stream << "gt_observation_log_likelihood: " << evaluation_result.gt_observation_log_likelihood << std::endl;
    evaluation_result_stream << "est_observation_log_likelihood: " << evaluation_result.est_observation_log_likelihood << std::endl;
    evaluation_result_stream << "evaluated_steps: " << evaluation_result.evaluated_steps << std::endl;
    evaluation_result_stream << "gt_steps: " << evaluation_result.gt_steps << std::endl;
    evaluation_result_stream << "est_steps: " << evaluation_result.est_steps << std::endl;
    evaluation_result_stream << "smoothed_position_errors size: " << evaluation_result.smoothed_position_errors.size() << std::endl;
    evaluation_result_stream << "smoothed_yaw_errors size: " << evaluation_result.smoothed_yaw_errors.size() << std::endl;
    evaluation_result_stream << "smoothed_results size: " << evaluation_result.smoothed_results.size() << std::endl;
    evaluation_result_stream << "average_dead_reckoning_step_time: " << evaluation_result.average_dead_reckoning_step_time << std::endl;
    evaluation_result_stream << "average_update_step_time: " << evaluation_result.average_update_step_time << std::endl;
    evaluation_result_stream << "average_smoothed_position_error: " << evaluation_result.average_smoothed_position_error << std::endl;
    evaluation_result_stream << "average_smoothed_yaw_error: " << evaluation_result.average_smoothed_yaw_error << std::endl;

    evaluation_result_stream.close();

    int smooth_step_start_index = evaluation_result.gt_steps - evaluation_result.evaluated_steps;
    for (int i = 0; i < evaluation_result.smoothed_results.size(); i++) {
      util::Result result = evaluation_result.smoothed_results.at(i);
      TestSummaryResult test_summary_result_smooth;
      test_summary_result_smooth.wpa_result = result;
      test_summary_result_smooth.ins_position_under_gt_orientation(0) = ins_x_gt_orientations.at(smooth_step_start_index + i);
      test_summary_result_smooth.ins_position_under_gt_orientation(1) = ins_y_gt_orientations.at(smooth_step_start_index + i);
      test_summary_result_smooth.ins_position_under_orientation_sensor(0) = ins_x_orientation_sensors.at(smooth_step_start_index + i);
      test_summary_result_smooth.ins_position_under_orientation_sensor(1) = ins_y_orientation_sensors.at(smooth_step_start_index + i);
      test_summary_result_smooth.gt_position_under_gt_orientation_and_gt_velocity(0) = gt_x_gt_orientations.at(smooth_step_start_index + i);
      test_summary_result_smooth.gt_position_under_gt_orientation_and_gt_velocity(1) = gt_y_gt_orientations.at(smooth_step_start_index + i);
      test_summary_result_smooth.yaw_m_imu = yaw_m_imus.at(smooth_step_start_index + i);
      test_summary_result_smooth.yaw_m_orientation = yaw_m_orientations.at(smooth_step_start_index + i);
      test_summary_result_smooth.config_paras = config_paras;
      test_summary_result_smooth.gt_bluetooth_offset = gt_bluetooth_offsets.at(smooth_step_start_index + i);
      test_summary_result_smooth.gt_wifi_offset = gt_wifi_offsets.at(smooth_step_start_index + i);
      test_summary_result_smooth.gt_geomagnetism_bias = gt_geomagnetism_biases.at(smooth_step_start_index + i);
      test_summary_result_smooth.dq_log_error = dq_log_errors.at(smooth_step_start_index + i);
      test_summary_result_smooth.INS_v_local = ins_local_vs.at(smooth_step_start_index + i);
      test_summary_result_smooth.gt_local_velocity = gt_local_vs.at(smooth_step_start_index + i);
      test_summary_result_smooth.ins_position_under_est_orientation(0) = ins_x_est_orientations.at(smooth_step_start_index + i);
      test_summary_result_smooth.ins_position_under_est_orientation(1) = ins_y_est_orientations.at(smooth_step_start_index + i);
      test_summary_result_smooth.global_R_orientation_sensor_log_error = global_R_orientation_sensor_log_errors.at(smooth_step_start_index + i);
      test_summary_result_smooth.global_R_geomag_log_error = global_R_geomag_log_errors.at(smooth_step_start_index + i);
      test_summary_result_smooth.gt_state_log_probability = 0.0;

      smooth_result_stream << test_summary_result_smooth.Output() << std::endl;
    }
    smooth_result_stream.close();
  }
}

void test_wireless_positioning_multiplexing_agent_stability_evaluation(configuration::ConfigParas config_paras, bool need_evaluation, bool print_verbose, DiscretizationResolutions resolutions, bool need_estimation, std::string client_trace_path, std::string output_folderpath, bool track_gt_particle, bool dump_online_filter_states, int skip_request_num = 0, int remain_request_num = 0, std::string gt_euroc_filepath = "", std::string pose_prior_euroc_filepath = "") {
#ifdef RANDOMNESS_OFF
  std::default_random_engine random_seed_generator;
  random_seed_generator.seed(static_cast<uint64_t>(2022));
  std::uniform_int_distribution<int> random_seed_distribution(0, 1000);
  std::set<int> random_seeds;
  while (random_seeds.size() < config_paras.number_of_retries) {
    random_seeds.insert(random_seed_distribution(random_seed_generator));
  }
  std::vector<int> random_seeds_vector;
  for (auto it = random_seeds.begin(); it != random_seeds.end(); it++) {
    random_seeds_vector.push_back(*it);
  }
#endif
  if (config_paras.stability_evaluation) {
    std::filesystem::create_directories(output_folderpath);
    std::string retry_folderpath;
    std::vector<std::vector<util::Result>> all_results;
    for (int i = 0; i < config_paras.number_of_retries; i++) {
      std::vector<util::Result> retry_sequential_results;
      retry_folderpath = std::filesystem::path(output_folderpath) / (std::string("retry_") + std::to_string(i));
      std::filesystem::create_directory(retry_folderpath);
      config_paras.output_folderpath = retry_folderpath;
#ifdef RANDOMNESS_OFF
      config_paras.random_seed = random_seeds_vector.at(i);
#endif
      test_wireless_positioning_multiplexing_agent(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, retry_folderpath, &retry_sequential_results, track_gt_particle, dump_online_filter_states, skip_request_num, remain_request_num, gt_euroc_filepath, pose_prior_euroc_filepath);
      all_results.push_back(retry_sequential_results);
    }

    // evaluate stability
    int trajectory_length = all_results.at(0).size();
    for (int i = 0; i < all_results.size(); i++) {
      assert(all_results.at(i).size() == trajectory_length);
    }

    std::string stability_result_path = std::filesystem::path(output_folderpath) / "stability_statistics.csv";
    std::ofstream stability_result_stream;
    stability_result_stream.open(stability_result_path);
    stability_result_stream.setf(std::ios::scientific, std::ios::floatfield);
    stability_result_stream.precision(6);

    for (int i = 0; i < trajectory_length; i++) {
      double mean_x = 0.0;
      double mean_y = 0.0;
      double mse = 0.0;
      for (int j = 0; j < all_results.size(); j++) {
        mean_x += all_results.at(j).at(i).est_position().x();
        mean_y += all_results.at(j).at(i).est_position().y();
        mse += std::pow(all_results.at(j).at(i).est_position().x() - all_results.at(j).at(i).gt_position().x(), 2.0) +
               std::pow(all_results.at(j).at(i).est_position().y() - all_results.at(j).at(i).gt_position().y(), 2.0);
      }
      mean_x = mean_x / all_results.size();
      mean_y = mean_y / all_results.size();
      mse = mse / all_results.size();
      double distance_variance = 0.0;
      for (int j = 0; j < all_results.size(); j++) {
        distance_variance += std::pow(all_results.at(j).at(i).est_position().x() - mean_x, 2.0) + std::pow(all_results.at(j).at(i).est_position().y() - mean_y, 2.0) ;
      }
      distance_variance = distance_variance / all_results.size();

      stability_result_stream << std::pow(distance_variance, 0.5) << "," << std::pow(mse, 0.5);

      for (int j = 0; j < all_results.size(); j++) {
        stability_result_stream << "," << std::pow(std::pow(all_results.at(j).at(i).est_position().x() - all_results.at(j).at(i).gt_position().x(), 2.0) +
                                                    std::pow(all_results.at(j).at(i).est_position().y() - all_results.at(j).at(i).gt_position().y(), 2.0), 0.5);
      }
      stability_result_stream << std::endl;
    }

    stability_result_stream.close();

  } else {
    test_wireless_positioning_multiplexing_agent(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, output_folderpath, nullptr, track_gt_particle, dump_online_filter_states, skip_request_num, remain_request_num, gt_euroc_filepath, pose_prior_euroc_filepath);
  }
}

void experiment_wireless_positioning_multiplexing_agent_different_initial_position(configuration::ConfigParas config_paras, bool need_evaluation, bool print_verbose, DiscretizationResolutions resolutions, bool need_estimation, std::string client_trace_path, std::string output_folderpath, bool track_gt_particle, bool dump_online_filter_states, int num_skip_requests, int num_remain_requests, int n_sample_trajectories = 10, std::string gt_euroc_filepath = "") {
  offline::ClientDataReader client_data_reader;
  client_data_reader.Init(client_trace_path);
  int num_requests = client_data_reader.GetSize();

  int actual_num_requests = num_requests - num_skip_requests - num_remain_requests;

  const int kMinTrajectorySize = 200;

  assert(actual_num_requests > kMinTrajectorySize);

  std::uniform_int_distribution<int> uniform_distribution(0, actual_num_requests - kMinTrajectorySize - 1);
  std::default_random_engine random_generator;

  // struct timespec tn;
  // clock_gettime(CLOCK_REALTIME, &tn);
  // random_generator.seed(static_cast<uint64_t>(tn.tv_nsec));
  random_generator.seed(2022);

  // // randomly sample starting points
  // for (int i = 0; i < n_sample_trajectories; i++) {
  //   int n_skip_requests = uniform_distribution(random_generator) + num_skip_requests;
  //   assert(n_skip_requests >= 0);
  //   assert(n_skip_requests <= (num_requests - num_remain_requests - kMinTrajectorySize));
  //   std::string sub_output_folderpath = std::filesystem::path(output_folderpath) / ("skip_" + std::to_string(n_skip_requests) + "_requests");
  //   std::filesystem::create_directories(sub_output_folderpath);
  //   test_wireless_positioning_multiplexing_agent_stability_evaluation(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, sub_output_folderpath, track_gt_particle, dump_online_filter_states, n_skip_requests, num_remain_requests, gt_euroc_filepath);
  // }

  // sample starting points with fixed step-length
  const int kStepLength = 10;
  const int kTrajectoryLength = 200;
  int n_skip_requests = num_skip_requests;
  while (n_skip_requests < num_requests - num_remain_requests - kMinTrajectorySize) {
    assert(n_skip_requests >= 0);
    int n_remaining_requests = num_requests - n_skip_requests - kTrajectoryLength - 1;
    std::string sub_output_folderpath = std::filesystem::path(output_folderpath) / ("skip_" + std::to_string(n_skip_requests) + "_requests");
    std::filesystem::create_directories(sub_output_folderpath);
    test_wireless_positioning_multiplexing_agent_stability_evaluation(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, sub_output_folderpath, track_gt_particle, dump_online_filter_states, n_skip_requests, n_remaining_requests, gt_euroc_filepath);
    n_skip_requests += kStepLength;
  }
}

void experiment_different_walking_distance(configuration::ConfigParas config_paras, bool need_evaluation, bool print_verbose, DiscretizationResolutions resolutions, bool need_estimation, std::string client_trace_path, std::string output_folderpath, bool track_gt_particle, bool dump_online_filter_states, int num_skip_requests, int num_remain_requests) {
  offline::ClientDataReader client_data_reader;
  client_data_reader.Init(client_trace_path);
  client_data_reader.SortRequestFolderPaths();
  int num_requests = client_data_reader.GetSize();

  int request_counter = 0;
  while (request_counter < num_skip_requests) {
    util::ClientRequest temp_request = client_data_reader.GetNextRequest();
    request_counter += 1;
  }

  std::vector<double> evaluate_convergence_distances = {5.0, 10.0, 15.0, 20.0};
  std::deque<double> distance_to_current_request;

  util::ClientRequest previous_client_request;
  util::ClientRequest current_client_request;
  while (request_counter < num_requests - num_remain_requests) {
    previous_client_request = current_client_request;
    current_client_request = client_data_reader.GetNextRequest();
    request_counter += 1;
    distance_to_current_request.push_back(0.0);
    if (distance_to_current_request.size() <= 1) {
      continue;
    }

    // update window_distance and window_start_request_id
    variable::Position previous_position = previous_client_request.gt_position;
    variable::Position current_position = current_client_request.gt_position;
    double last_step_distance = std::pow(std::pow(current_position.x() - previous_position.x(), 2.0) + std::pow(current_position.y() - previous_position.y(), 2.0), 0.5);
    for (int i = 0; i < distance_to_current_request.size() - 1; i++) {
      distance_to_current_request.at(i) += last_step_distance;
    }

    int evaluate_remain_requests = num_requests - request_counter;

    // make dir
    std::string sub_output_folderpath = std::filesystem::path(output_folderpath) / std::to_string(request_counter);
    std::filesystem::create_directories(sub_output_folderpath);

    // evaluate
    for (int i = 0; i < evaluate_convergence_distances.size(); i++) {
      double evaluate_distance = evaluate_convergence_distances.at(i);
      if (evaluate_distance > distance_to_current_request.at(0)) {
        continue;
      }
      int evaluate_skip_requests = num_skip_requests;
      for (int j = 0; j < distance_to_current_request.size(); j++) {
        if (distance_to_current_request.at(j) <= evaluate_distance) {
          evaluate_skip_requests += j;
          break;
        }
      }
      std::string sub_sub_output_folderpath = std::filesystem::path(sub_output_folderpath) / std::to_string(evaluate_distance);
      std::filesystem::create_directories(sub_sub_output_folderpath);
      test_wireless_positioning_multiplexing_agent_stability_evaluation(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, sub_sub_output_folderpath, track_gt_particle, dump_online_filter_states, evaluate_skip_requests, evaluate_remain_requests);
    }
  }
}

void experiment_wireless_positioning_multiplexing_agent_observability(configuration::ConfigParas config_paras, bool need_evaluation, bool print_verbose, DiscretizationResolutions resolutions, bool need_estimation, std::string client_trace_path, std::string output_folderpath, bool track_gt_particle, bool dump_online_filter_states, std::string gt_euroc_filepath = "") {
  // std::vector<double> global_yaw_half_ranges = {0.263, 0.523};
  // std::vector<std::string> global_yaw_range_strs = {"30", "60"};
  std::vector<double> global_yaw_half_ranges = {0.263};
  std::vector<std::string> global_yaw_range_strs = {"30"};

  for (int i = 0; i < global_yaw_half_ranges.size(); i++) {
    config_paras.orientation_sensor_yaw_variance_in_rad = global_yaw_half_ranges.at(i);
    std::string sub_output_folderpath = std::filesystem::path(output_folderpath) / ("global_yaw_range_" + global_yaw_range_strs.at(i) + "_degree");
    std::filesystem::create_directories(sub_output_folderpath);

    config_paras.use_gt_gravity = false;
    config_paras.use_gt_local_velocity = true;
    config_paras.sample_local_velocity = false;
    config_paras.use_gt_local_rotation = true;
    config_paras.sample_local_rotation = false;
    config_paras.use_gt_global_orientation = true;
    config_paras.sample_global_orientation = false;
    config_paras.translation_module_load_gt_orientation_instead_of_oriention_sensor = true;
    config_paras.predict_translation_with_estimated_orientation = true;
    config_paras.return_orientation_sensor_orientation = false;
    config_paras.synthesize_gt_local_velocity = false;
    config_paras.synthesize_gt_local_rotation = false;
    config_paras.synthesize_gt_global_orientation = false;

    configuration::ConfigParas temp_config_paras;
    std::string ssub_output_folderpath;
    std::string sssub_output_folderpath;
    int n_steps;

    // no jittering, only use gt measurements
    temp_config_paras = config_paras;
    ssub_output_folderpath = std::filesystem::path(sub_output_folderpath) / "no_uncertainty";
    std::filesystem::create_directories(ssub_output_folderpath);
    test_wireless_positioning_multiplexing_agent_stability_evaluation(temp_config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, ssub_output_folderpath, track_gt_particle, dump_online_filter_states, 0, 0, gt_euroc_filepath);

    // jitter on gt_local_velocity
    temp_config_paras = config_paras;
    temp_config_paras.sample_local_velocity = true;
    ssub_output_folderpath = std::filesystem::path(sub_output_folderpath) / "sample_local_velocity";
    std::filesystem::create_directories(ssub_output_folderpath);
    test_wireless_positioning_multiplexing_agent_stability_evaluation(temp_config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, ssub_output_folderpath, track_gt_particle, dump_online_filter_states, 0, 0, gt_euroc_filepath);

    // jitter on gt_local_rotation
    temp_config_paras = config_paras;
    temp_config_paras.sample_local_rotation = true;
    ssub_output_folderpath = std::filesystem::path(sub_output_folderpath) / "sample_local_rotation";
    std::filesystem::create_directories(ssub_output_folderpath);
    test_wireless_positioning_multiplexing_agent_stability_evaluation(temp_config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, ssub_output_folderpath, track_gt_particle, dump_online_filter_states, 0, 0, gt_euroc_filepath);

    // change jitters on gt_local_velocity
    temp_config_paras = config_paras;
    temp_config_paras.sample_local_velocity = true;
    ssub_output_folderpath = std::filesystem::path(sub_output_folderpath) / "sample_local_velocity_variate_jitter";
    std::filesystem::create_directories(ssub_output_folderpath);
    double velocity_covariance_multiplicative_step = 10.0;
    n_steps = 5;
    for (int i = 0; i < n_steps; i++) {
      temp_config_paras.INS_v_local_covariance = Eigen::Matrix3d::Identity() * 1e-3 * std::pow(velocity_covariance_multiplicative_step, i);
      sssub_output_folderpath = std::filesystem::path(ssub_output_folderpath) / (std::string("jitter_") + std::to_string(i));
      std::filesystem::create_directories(sssub_output_folderpath);
      test_wireless_positioning_multiplexing_agent_stability_evaluation(temp_config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, sssub_output_folderpath, track_gt_particle, dump_online_filter_states, 0, 0, gt_euroc_filepath);
    }

    // change jitters on gt_local_rotation
    temp_config_paras = config_paras;
    temp_config_paras.sample_local_rotation = true;
    ssub_output_folderpath = std::filesystem::path(sub_output_folderpath) / "sample_local_rotation_variate_jitter";
    std::filesystem::create_directories(ssub_output_folderpath);
    double rotation_covariance_multiplicative_step = 10.0;
    n_steps = 7;
    for (int i = 0; i < n_steps; i++) {
      temp_config_paras.rotation_covariance = Eigen::Matrix3d::Identity() * 1e-8 * std::pow(rotation_covariance_multiplicative_step, i);
      sssub_output_folderpath = std::filesystem::path(ssub_output_folderpath) / (std::string("jitter_") + std::to_string(i));
      std::filesystem::create_directories(sssub_output_folderpath);
      test_wireless_positioning_multiplexing_agent_stability_evaluation(temp_config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, sssub_output_folderpath, track_gt_particle, dump_online_filter_states, 0, 0, gt_euroc_filepath);
    }
  }
}

}  // namespace state_estimation

int main(int argc, char** argv) {
  state_estimation::DiscretizationResolutions resolutions;
  resolutions.position_resolution = 0.5;
  resolutions.yaw_resolution = 0.2;
  resolutions.bluetooth_offset_resolution = 1.0;
  resolutions.wifi_offset_resolution = 1.0;
  resolutions.geomagnetism_bias_resolution_x = 1.0;
  resolutions.geomagnetism_bias_resolution_y = 1.0;
  resolutions.geomagnetism_bias_resolution_z = 1.0;

  bool print_verbose = false;
  bool need_estimation = true;
  bool need_evaluation = false;
  bool track_gt_particle = false;
  bool dump_online_filter_states = false;

  if (argc < 2) {
    std::cout << "Please at least provide the yaml-config_filepath." << std::endl;
    return 1;
  }

  std::string config_path = argv[1];
  state_estimation::configuration::Configurator configurator;
  configurator.Init(config_path);
  state_estimation::configuration::ConfigParas config_paras = configurator.config_paras();

  if (argc < 3) {
    // single data mode
    state_estimation::test_wireless_positioning_multiplexing_agent_stability_evaluation(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, config_paras.client_trace_path, config_paras.output_folderpath, track_gt_particle, dump_online_filter_states);
  } else if (argc < 6) {
    std::cout << "For batch mode, please provide the request folderpath, the output folderpath and the num_skip_requests." << std::endl;
    return 1;
  } else if (argc < 7) {
    std::string client_trace_path = argv[2];
    std::string output_folderpath = argv[3];
    std::string num_skip_requests_str = argv[4];
    int num_skip_requests = std::stoi(num_skip_requests_str);
    std::string num_remain_requests_str = argv[5];
    int num_remain_requests = std::stoi(num_remain_requests_str);
    // state_estimation::test_wireless_positioning_multiplexing_agent_stability_evaluation(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, output_folderpath, false, dump_online_filter_states, num_skip_requests, num_remain_requests);
    state_estimation::experiment_different_walking_distance(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, output_folderpath, false, dump_online_filter_states, num_skip_requests, num_remain_requests);
  } else if (argc < 8) {
    std::string client_trace_path = argv[2];
    std::string output_folderpath = argv[3];
    std::string num_skip_requests_str = argv[4];
    std::string num_remain_requests_str = argv[5];
    int num_skip_requests = std::stoi(num_skip_requests_str);
    int num_remain_requests = std::stoi(num_remain_requests_str);
    std::string gt_euroc_filepath = argv[6];
    state_estimation::test_wireless_positioning_multiplexing_agent_stability_evaluation(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, output_folderpath, false, dump_online_filter_states, num_skip_requests, num_remain_requests, gt_euroc_filepath);
    // state_estimation::experiment_wireless_positioning_multiplexing_agent_different_initial_position(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, output_folderpath, false, dump_online_filter_states, num_skip_requests, num_remain_requests, 10, gt_euroc_filepath);
  } else if (argc < 9) {
    std::string client_trace_path = argv[2];
    std::string output_folderpath = argv[3];
    std::string num_skip_requests_str = argv[4];
    std::string num_remain_requests_str = argv[5];
    int num_skip_requests = std::stoi(num_skip_requests_str);
    int num_remain_requests = std::stoi(num_remain_requests_str);
    std::string gt_euroc_filepath = argv[6];
    std::string pose_prior_euroc_filepath = argv[7];
    state_estimation::test_wireless_positioning_multiplexing_agent_stability_evaluation(config_paras, need_evaluation, print_verbose, resolutions, need_estimation, client_trace_path, output_folderpath, false, dump_online_filter_states, num_skip_requests, num_remain_requests, gt_euroc_filepath, pose_prior_euroc_filepath);
  }

  return 0;
}

