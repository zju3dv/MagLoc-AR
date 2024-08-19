/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-11-29 19:54:05
 * @LastEditTime: 2023-03-02 16:50:21
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#ifndef STATE_ESTIMATION_UTIL_CLIENT_REQUEST_H_
#define STATE_ESTIMATION_UTIL_CLIENT_REQUEST_H_

#include <sys/stat.h>

#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace util {

static const int kOutputDoublePrecision = 9;

struct ClientRequest {
  double timestamp = -1.0;
  double gt_timestamp = -1.0;
  variable::Position gt_position;
  Eigen::Quaterniond gt_orientation = Eigen::Quaterniond::Identity();
  Eigen::Matrix<double, 3, 1> gt_velocity = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> gt_bias_w = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 1> gt_bias_a = Eigen::Matrix<double, 3, 1>::Zero();
  std::vector<std::string> bluetooth_lines;
  std::vector<std::string> wifi_lines;
  std::vector<std::string> orientation_lines;
  std::vector<std::string> gravity_lines;
  std::vector<std::string> geomagnetism_lines;
  std::vector<std::string> gyroscope_lines;
  std::vector<Eigen::Matrix<double, 4, 1>> gyroscope_data;
  std::vector<std::string> accelerometer_lines;
  std::vector<Eigen::Matrix<double, 4, 1>> accelerometer_data;
  Eigen::Vector2d pdr_dp_trivial{0.0, 0.0};
  Eigen::Vector2d pdr_dp{0.0, 0.0};
  Eigen::Vector2d pdr_cov{0.0, 0.0};
  double pdr_heading_speed = 0.0;
  double INS_timestamp = -1.0;
  Eigen::Vector3d INS_v_local{0.0, 0.0, 0.0};
  Eigen::Matrix3d INS_v_local_cov = Eigen::Matrix3d::Zero();
  Eigen::Quaterniond imu_pose_ws = Eigen::Quaterniond::Identity();
  double orientation_sensor_timestamp = -1.0;
  Eigen::Quaterniond orientation_sensor_pose_ws = Eigen::Quaterniond::Identity();
  double gravity_s_timestamp = -1.0;
  Eigen::Vector3d gravity_s = Eigen::Vector3d::Zero();
  Eigen::Vector3d position_prior = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation_prior = Eigen::Quaterniond::Identity();
  bool position_prior_valid = false;
  bool orientation_prior_valid = false;

  void OutputClientRequest(std::string base_path, int request_index) {
    std::string foldername = "request_" + std::to_string(request_index) + "_timestamp_" + DoubleToString(timestamp, 9);
    std::string folderpath = base_path + "/" + foldername;
    if (mkdir(folderpath.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == -1) {
      std::cout << "ClientRequest::OutputClientRequest: error creating directory " << folderpath << std::endl;
    } else {
      if (FILE *gt_csv = fopen((folderpath + "/gba_pose.csv").c_str(), "w")) {
        fprintf(gt_csv, "%s\n",
                (DoubleToString(timestamp * 1e9, 0) + "," +
                 DoubleToString(gt_position.x(), kOutputDoublePrecision) + "," +
                 DoubleToString(gt_position.y(), kOutputDoublePrecision) + "," +
                 "0.0" + "," +
                 DoubleToString(gt_orientation.w(), kOutputDoublePrecision) + "," +
                 DoubleToString(gt_orientation.x(), kOutputDoublePrecision) + "," +
                 DoubleToString(gt_orientation.y(), kOutputDoublePrecision) + "," +
                 DoubleToString(gt_orientation.z(), kOutputDoublePrecision) + "," +
                 "0.0" + "," +
                 "0.0" + "," +
                 "0.0" + "," +
                 "0.0" + "," +
                 "0.0" + "," +
                 "0.0" + "," +
                 "0.0" + "," +
                 "0.0" + "," +
                 "0.0" + "," +
                 DoubleToString(gt_timestamp * 1e9, 0))
                    .c_str());
        fclose(gt_csv);
      }
      if (FILE *bluetooth_csv = fopen((folderpath + "/bluetooth.csv").c_str(), "w")) {
        for (auto item : bluetooth_lines) {
          fprintf(bluetooth_csv, "%s\n", item.c_str());
        }
        fclose(bluetooth_csv);
      }
      if (FILE *wifi_csv = fopen((folderpath + "/wifi.csv").c_str(), "w")) {
        for (auto item : wifi_lines) {
          fprintf(wifi_csv, "%s\n", item.c_str());
        }
        fclose(wifi_csv);
      }
      if (FILE *orientation_csv = fopen((folderpath + "/rv.csv").c_str(), "w")) {
        for (auto item : orientation_lines) {
          fprintf(orientation_csv, "%s\n", item.c_str());
        }
        fclose(orientation_csv);
      }
      if (FILE *gravity_csv = fopen((folderpath + "/gravity.csv").c_str(), "w")) {
        for (auto item : gravity_lines) {
          fprintf(gravity_csv, "%s\n", item.c_str());
        }
        fclose(gravity_csv);
      }
      if (FILE *accelerometer_csv = fopen((folderpath + "/accelerometer.csv").c_str(), "w")) {
        for (auto item : accelerometer_lines) {
          fprintf(accelerometer_csv, "%s\n", item.c_str());
        }
        fclose(accelerometer_csv);
      }
      if (FILE *gyroscope_csv = fopen((folderpath + "/gyroscope.csv").c_str(), "w")) {
        for (auto item : gyroscope_lines) {
          fprintf(gyroscope_csv, "%s\n", item.c_str());
        }
        fclose(gyroscope_csv);
      }
      if (FILE *geomagnetism_csv = fopen((folderpath + "/magnetic_field.csv").c_str(), "w")) {
        for (auto item : geomagnetism_lines) {
          fprintf(geomagnetism_csv, "%s\n", item.c_str());
        }
        fclose(geomagnetism_csv);
      }
      if (FILE *pdr_csv = fopen((folderpath + "/pdr.csv").c_str(), "w")) {
        fprintf(pdr_csv, "%s\n", (DoubleToString(timestamp * 1e9, 0) + "," + DoubleToString(pdr_dp_trivial(0), kOutputDoublePrecision) + "," + DoubleToString(pdr_dp_trivial(1), kOutputDoublePrecision)).c_str());
        fclose(pdr_csv);
      }
      if (FILE *pdr_cov_csv = fopen((folderpath + "/pdr_cov.csv").c_str(), "w")) {
        fprintf(pdr_cov_csv, "%s\n",
                (DoubleToString(timestamp * 1e9, 0) + "," +
                 DoubleToString(pdr_dp(0), kOutputDoublePrecision) + "," +
                 DoubleToString(pdr_dp(1), kOutputDoublePrecision) + "," +
                 DoubleToString(pdr_cov(0), kOutputDoublePrecision) + "," +
                 DoubleToString(pdr_cov(1), kOutputDoublePrecision))
                    .c_str());
        fclose(pdr_cov_csv);
      }
      if (FILE *pdr_heading_speed_csv = fopen((folderpath + "/pdr_heading_speed.csv").c_str(), "w")) {
        fprintf(pdr_heading_speed_csv, "%s\n",
                (DoubleToString(timestamp * 1e9, 0) + "," +
                 DoubleToString(pdr_heading_speed, kOutputDoublePrecision))
                    .c_str());
        fclose(pdr_heading_speed_csv);
      }
      if (FILE *INS_v_local_csv = fopen((folderpath + "/INS_v_local.csv").c_str(), "w")) {
        fprintf(INS_v_local_csv, "%s\n",
                (DoubleToString(timestamp * 1e9, 0) + "," +
                 DoubleToString(INS_v_local(0), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_v_local(1), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_v_local(2), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_v_local_cov(0, 0), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_v_local_cov(0, 1), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_v_local_cov(0, 2), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_v_local_cov(1, 1), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_v_local_cov(1, 2), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_v_local_cov(2, 2), kOutputDoublePrecision) + "," +
                 DoubleToString(INS_timestamp * 1e9, 0))
                    .c_str());
        fclose(INS_v_local_csv);
      }
      if (FILE *imu_pose_ws_csv = fopen((folderpath + "/imu_pose_ws.csv").c_str(), "w")) {
        fprintf(imu_pose_ws_csv, "%s\n",
                (DoubleToString(timestamp * 1e9, 0) + "," +
                 DoubleToString(imu_pose_ws.x(), kOutputDoublePrecision) + "," +
                 DoubleToString(imu_pose_ws.y(), kOutputDoublePrecision) + "," +
                 DoubleToString(imu_pose_ws.z(), kOutputDoublePrecision) + "," +
                 DoubleToString(imu_pose_ws.w(), kOutputDoublePrecision))
                    .c_str());
        fclose(imu_pose_ws_csv);
      }
      if (FILE *orientation_sensor_pose_ws_csv = fopen((folderpath + "/orientation_sensor_pose_ws.csv").c_str(), "w")) {
        fprintf(orientation_sensor_pose_ws_csv, "%s\n",
                (DoubleToString(timestamp * 1e9, 0) + "," +
                 DoubleToString(orientation_sensor_pose_ws.x(), kOutputDoublePrecision) + "," +
                 DoubleToString(orientation_sensor_pose_ws.y(), kOutputDoublePrecision) + "," +
                 DoubleToString(orientation_sensor_pose_ws.z(), kOutputDoublePrecision) + "," +
                 DoubleToString(orientation_sensor_pose_ws.w(), kOutputDoublePrecision) + "," +
                 DoubleToString(orientation_sensor_timestamp * 1e9, 0))
                    .c_str());
        fclose(orientation_sensor_pose_ws_csv);
      }
      if (FILE *gravity_s_csv = fopen((folderpath + "/gravity_s.csv").c_str(), "w")) {
        fprintf(gravity_s_csv, "%s\n",
                (DoubleToString(timestamp * 1e9, 0) + "," +
                 DoubleToString(gravity_s(0), kOutputDoublePrecision) + "," +
                 DoubleToString(gravity_s(1), kOutputDoublePrecision) + "," +
                 DoubleToString(gravity_s(2), kOutputDoublePrecision) + "," +
                 DoubleToString(gravity_s_timestamp * 1e9, 0))
                    .c_str());
        fclose(gravity_s_csv);
      }
    }
  }
};

}  // namespace util

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_UTIL_CLIENT_REQUEST_H_
