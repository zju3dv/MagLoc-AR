/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:17:44
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-21 13:01:08
 */
#include "offline/client_data_reader.h"

#include <Eigen/Dense>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "util/client_request.h"
#include "util/misc.h"

namespace state_estimation {

namespace offline {

bool CompareRequestFolderNamesAscending(std::string folder_name_0,
                                        std::string folder_name_1) {
  std::vector<std::string> folder_name_split_0;
  std::vector<std::string> folder_name_split_1;
  SplitString(folder_name_0, folder_name_split_0, "_");
  SplitString(folder_name_1, folder_name_split_1, "_");
  int request_id_0 = std::stoi(folder_name_split_0[1]);
  int request_id_1 = std::stoi(folder_name_split_1[1]);
  return request_id_0 < request_id_1;
}

bool CompareRequestFolderNamesDescending(std::string folder_name_0,
                                         std::string folder_name_1) {
  std::vector<std::string> folder_name_split_0;
  std::vector<std::string> folder_name_split_1;
  SplitString(folder_name_0, folder_name_split_0, "_");
  SplitString(folder_name_1, folder_name_split_1, "_");
  int request_id_0 = std::stoi(folder_name_split_0[1]);
  int request_id_1 = std::stoi(folder_name_split_1[1]);
  return request_id_0 > request_id_1;
}

int ClientDataReader::Init(std::string client_folderpath) {
  this->client_folder_path_ = client_folderpath;
  this->client_request_folder_names_ =
      GetFolderNamesInFolder(client_folderpath);
  this->timestamp_ms_str_to_folder_name_.clear();
  this->index_to_folder_name_.clear();
  std::string folder_name;
  std::vector<std::string> folder_name_split;
  std::vector<std::string> timestamp_split;
  for (int i = 0; i < this->client_request_folder_names_.size(); i++) {
    folder_name = this->client_request_folder_names_.at(i);
    SplitString(folder_name, folder_name_split, "_");
    SplitString(folder_name_split[3], timestamp_split, ".");
    this->timestamp_ms_str_to_folder_name_.insert(std::pair<std::string, std::string>(timestamp_split[0] + timestamp_split[1].substr(0, 3), folder_name));
    this->index_to_folder_name_.insert(std::pair<int, std::string>(std::stoi(folder_name_split[1]), folder_name));
  }
  return this->client_request_folder_names_.size();
}

void ClientDataReader::SortRequestFolderPaths(bool reverse) {
  std::vector<std::string> client_request_folder_names =
      this->client_request_folder_names();
  if (reverse) {
    sort(client_request_folder_names.begin(),
         client_request_folder_names.end(),
         CompareRequestFolderNamesDescending);
  } else {
    sort(client_request_folder_names.begin(),
         client_request_folder_names.end(),
         CompareRequestFolderNamesAscending);
  }
  this->client_request_folder_names_ = client_request_folder_names;
}

util::ClientRequest ClientDataReader::GetRequestByRequestFoldername(std::string request_folder_name) {
  std::string request_folder_path =
      this->client_folder_path_ + "/" + request_folder_name;

  // initialize the client_request
  util::ClientRequest client_request;

  // read and fill information contained in the foldername
  std::vector<std::string> folder_name_split;
  SplitString(request_folder_name, folder_name_split, "_");
  double timestamp = std::stod(folder_name_split[3]);
  client_request.timestamp = timestamp;

  // read and fill information contained in the groundtruth file.
  // the groundtruth file is in the format of EuRoC.
  std::string gt_path = request_folder_path + "/gba_pose.csv";
  std::vector<std::string> gt_lines = GetLinesInFile(gt_path);
  if (gt_lines.size() == 0) {
    client_request.gt_position.x(0.0);
    client_request.gt_position.y(0.0);
    client_request.gt_position.z(0.0);
    client_request.gt_orientation = Eigen::Quaterniond::Identity();
    client_request.gt_velocity = Eigen::Matrix<double, 3, 1>::Zero();
    client_request.gt_bias_w = Eigen::Matrix<double, 3, 1>::Zero();
    client_request.gt_bias_a = Eigen::Matrix<double, 3, 1>::Zero();
    client_request.gt_timestamp = -1.0;
  } else {
    std::vector<std::string> gt_line_split;
    SplitString(gt_lines[0], gt_line_split, ",");
    variable::Position gt_position;
    gt_position.x(std::stod(gt_line_split[1]));
    gt_position.y(std::stod(gt_line_split[2]));
    gt_position.z(std::stod(gt_line_split[3]));
    Eigen::Quaterniond gt_orientation(
        std::stod(gt_line_split[4]),
        std::stod(gt_line_split[5]),
        std::stod(gt_line_split[6]),
        std::stod(gt_line_split[7]));
    Eigen::Matrix<double, 3, 1> gt_velocity;
    gt_velocity(0) = std::stod(gt_line_split[8]);
    gt_velocity(1) = std::stod(gt_line_split[9]);
    gt_velocity(2) = std::stod(gt_line_split[10]);
    Eigen::Matrix<double, 3, 1> gt_bias_w;
    gt_bias_w(0) = std::stod(gt_line_split[11]);
    gt_bias_w(1) = std::stod(gt_line_split[12]);
    gt_bias_w(2) = std::stod(gt_line_split[13]);
    Eigen::Matrix<double, 3, 1> gt_bias_a;
    gt_bias_a(0) = std::stod(gt_line_split[14]);
    gt_bias_a(1) = std::stod(gt_line_split[15]);
    gt_bias_a(2) = std::stod(gt_line_split[16]);
    client_request.gt_position = gt_position;
    client_request.gt_orientation = gt_orientation;
    client_request.gt_velocity = gt_velocity;
    client_request.gt_bias_w = gt_bias_w;
    client_request.gt_bias_a = gt_bias_a;
    client_request.gt_timestamp = client_request.timestamp;
  }

  // read and fill information contained in the pdr file.
  std::string pdr_path = request_folder_path + "/pdr.csv";
  std::vector<std::string> pdr_lines = GetLinesInFile(pdr_path);
  Eigen::Vector2d pdr_dp_trivial;
  if (pdr_lines.size() == 0) {
    pdr_dp_trivial(0) = 0.0;
    pdr_dp_trivial(0) = 0.0;
  } else {
    std::vector<std::string> pdr_line_split;
    SplitString(pdr_lines[0], pdr_line_split, ",");
    pdr_dp_trivial(0) = std::stod(pdr_line_split[1]);
    pdr_dp_trivial(1) = std::stod(pdr_line_split[2]);
  }
  client_request.pdr_dp_trivial = pdr_dp_trivial;

  // read and fill information contained in the pdr_cov file.
  std::string pdr_cov_path = request_folder_path + "/pdr_cov.csv";
  std::vector<std::string> pdr_cov_lines = GetLinesInFile(pdr_cov_path);
  if (pdr_cov_lines.size() > 0) {
    std::vector<std::string> pdr_cov_line_split;
    SplitString(pdr_cov_lines[0], pdr_cov_line_split, ",");
    client_request.pdr_dp(0) = std::stod(pdr_cov_line_split[1]);
    client_request.pdr_dp(1) = std::stod(pdr_cov_line_split[2]);
    client_request.pdr_cov(0) = std::stod(pdr_cov_line_split[3]);
    client_request.pdr_cov(1) = std::stod(pdr_cov_line_split[4]);
  }

  // read and fill information contained in the pdr_heading_speed file.
  std::string pdr_heading_speed_path = request_folder_path + "/pdr_heading_speed.csv";
  std::vector<std::string> pdr_heading_speed_lines = GetLinesInFile(pdr_heading_speed_path);
  if (pdr_heading_speed_lines.size() > 0) {
    std::vector<std::string> pdr_heading_speed_line_split;
    SplitString(pdr_heading_speed_lines[0], pdr_heading_speed_line_split, ",");
    client_request.pdr_heading_speed = std::stod(pdr_heading_speed_line_split[1]);
  }

  // read and fill information contained in the INS_v_local.csv file.
  std::string INS_v_local_path = request_folder_path + "/INS_v_local.csv";
  std::vector<std::string> INS_v_local_lines = GetLinesInFile(INS_v_local_path);
  if (INS_v_local_lines.size() > 0) {
    std::vector<std::string> INS_v_local_line_split;
    SplitString(INS_v_local_lines[0], INS_v_local_line_split, ",");
    client_request.INS_v_local(0) = std::stod(INS_v_local_line_split[1]);
    client_request.INS_v_local(1) = std::stod(INS_v_local_line_split[2]);
    client_request.INS_v_local(2) = std::stod(INS_v_local_line_split[3]);
    client_request.INS_v_local_cov(0, 0) = std::stod(INS_v_local_line_split[4]);
    client_request.INS_v_local_cov(0, 1) = std::stod(INS_v_local_line_split[5]);
    client_request.INS_v_local_cov(1, 0) = std::stod(INS_v_local_line_split[5]);
    client_request.INS_v_local_cov(0, 2) = std::stod(INS_v_local_line_split[6]);
    client_request.INS_v_local_cov(2, 0) = std::stod(INS_v_local_line_split[6]);
    client_request.INS_v_local_cov(1, 1) = std::stod(INS_v_local_line_split[7]);
    client_request.INS_v_local_cov(1, 2) = std::stod(INS_v_local_line_split[8]);
    client_request.INS_v_local_cov(2, 1) = std::stod(INS_v_local_line_split[8]);
    client_request.INS_v_local_cov(2, 2) = std::stod(INS_v_local_line_split[9]);
  }

  // read and fill bluetooth_lines
  std::string bluetooth_path = request_folder_path + "/bluetooth.csv";
  std::vector<std::string> bluetooth_lines = GetLinesInFile(bluetooth_path);
  client_request.bluetooth_lines = bluetooth_lines;

  // read and fill wifi_lines
  std::string wifi_path = request_folder_path + "/wifi.csv";
  std::vector<std::string> wifi_lines = GetLinesInFile(wifi_path);
  client_request.wifi_lines = wifi_lines;

  // read and fill orientation_lines
  std::string orientation_path = request_folder_path + "/rv.csv";
  std::vector<std::string> orientation_lines = GetLinesInFile(orientation_path);
  client_request.orientation_lines = orientation_lines;

  // read and fill gravity_lines
  std::string gravity_path = request_folder_path + "/gravity.csv";
  std::vector<std::string> gravity_lines = GetLinesInFile(gravity_path);
  client_request.gravity_lines = gravity_lines;

  // read and fill geomagnetism_lines
  std::string geomagnetism_path =
      request_folder_path + "/magnetic_field.csv";
  std::vector<std::string> geomagnetism_lines =
      GetLinesInFile(geomagnetism_path);
  client_request.geomagnetism_lines = geomagnetism_lines;

  // read and fill gyroscope_lines
  std::string gyroscope_path = request_folder_path + "/gyroscope.csv";
  std::vector<std::string> gyroscope_lines = GetLinesInFile(gyroscope_path);
  client_request.gyroscope_lines = gyroscope_lines;
  client_request.gyroscope_data.clear();
  for (int i = 0; i < gyroscope_lines.size(); i++) {
    std::vector<std::string> gyroscope_line_split;
    SplitString(gyroscope_lines[i], gyroscope_line_split, ",");
    client_request.gyroscope_data.emplace_back(Eigen::Matrix<double, 4, 1>::Zero());
    client_request.gyroscope_data.back()(0) = std::stod(gyroscope_line_split[0]);
    client_request.gyroscope_data.back()(1) = std::stod(gyroscope_line_split[1]);
    client_request.gyroscope_data.back()(2) = std::stod(gyroscope_line_split[2]);
    client_request.gyroscope_data.back()(3) = std::stod(gyroscope_line_split[3]);
  }

  // read and fill accelerometer_lines
  std::string accelerometer_path = request_folder_path + "/accelerometer.csv";
  std::vector<std::string> accelerometer_lines = GetLinesInFile(accelerometer_path);
  client_request.accelerometer_lines = accelerometer_lines;
  client_request.accelerometer_data.clear();
  for (int i = 0; i < accelerometer_lines.size(); i++) {
    std::vector<std::string> accelerometer_line_split;
    SplitString(accelerometer_lines[i], accelerometer_line_split, ",");
    client_request.accelerometer_data.emplace_back(Eigen::Matrix<double, 4, 1>::Zero());
    client_request.accelerometer_data.back()(0) = std::stod(accelerometer_line_split[0]);
    client_request.accelerometer_data.back()(1) = std::stod(accelerometer_line_split[1]);
    client_request.accelerometer_data.back()(2) = std::stod(accelerometer_line_split[2]);
    client_request.accelerometer_data.back()(3) = std::stod(accelerometer_line_split[3]);
  }

  // read and fill gravity_s
  std::string gravity_s_path = request_folder_path + "/gravity_s.csv";
  std::vector<std::string> gravity_s_lines = GetLinesInFile(gravity_s_path);
  if (gravity_s_lines.size() > 0) {
    std::vector<std::string> gravity_s_line_split;
    SplitString(gravity_s_lines[0], gravity_s_line_split, ",");
    client_request.gravity_s_timestamp = std::stod(gravity_s_line_split[4]) * 1e-9;
    client_request.gravity_s(0) = std::stod(gravity_s_line_split[1]);
    client_request.gravity_s(1) = std::stod(gravity_s_line_split[2]);
    client_request.gravity_s(2) = std::stod(gravity_s_line_split[3]);
  }

  // read and fill imu_pose_ws
  std::string imu_pose_ws_path = request_folder_path + "/imu_pose_ws.csv";
  std::vector<std::string> imu_pose_ws_lines = GetLinesInFile(imu_pose_ws_path);
  if (imu_pose_ws_lines.size() > 0) {
    std::vector<std::string> imu_pose_ws_line_split;
    SplitString(imu_pose_ws_lines[0], imu_pose_ws_line_split, ",");
    client_request.imu_pose_ws = Eigen::Quaterniond(stod(imu_pose_ws_line_split[4]),
                                                    stod(imu_pose_ws_line_split[1]),
                                                    stod(imu_pose_ws_line_split[2]),
                                                    stod(imu_pose_ws_line_split[3]));
  }

  // read and fill orientation_sensor_pose_ws
  std::string orientation_sensor_pose_ws_path = request_folder_path + "/orientation_sensor_pose_ws.csv";
  std::vector<std::string> orientation_sensor_pose_ws_lines = GetLinesInFile(orientation_sensor_pose_ws_path);
  if (orientation_sensor_pose_ws_lines.size() > 0) {
    std::vector<std::string> orientation_sensor_pose_line_split;
    SplitString(orientation_sensor_pose_ws_lines[0], orientation_sensor_pose_line_split, ",");
    client_request.orientation_sensor_timestamp = stod(orientation_sensor_pose_line_split[5]) * 1e-9;
    client_request.orientation_sensor_pose_ws = Eigen::Quaterniond(stod(orientation_sensor_pose_line_split[4]),
                                                                   stod(orientation_sensor_pose_line_split[1]),
                                                                   stod(orientation_sensor_pose_line_split[2]),
                                                                   stod(orientation_sensor_pose_line_split[3]));
  }

  return client_request;
}

util::ClientRequest ClientDataReader::GetNextRequest(void) {
  std::string request_data_folder_name = this->client_request_folder_names_[this->cur_];
  this->cur_++;
  return this->GetRequestByRequestFoldername(request_data_folder_name);
}

util::ClientRequest ClientDataReader::GetRequestByTimestampMsStr(std::string timestamp_ms_str) {
  util::ClientRequest client_request;
  if (this->timestamp_ms_str_to_folder_name_.find(timestamp_ms_str) == this->timestamp_ms_str_to_folder_name_.end()) {
    std::cout << "ClientDataReader::GetRequestByTimestampMs: invalid timestamp_ms_str: " << timestamp_ms_str << std::endl;
  } else {
    std::string request_folder_name = this->timestamp_ms_str_to_folder_name_.at(timestamp_ms_str);
    client_request = this->GetRequestByRequestFoldername(request_folder_name);
  }
  return client_request;
}

util::ClientRequest ClientDataReader::GetRequestByIndex(int request_index) {
  util::ClientRequest client_request;
  if (this->index_to_folder_name_.find(request_index) == this->index_to_folder_name_.end()) {
    std::cout << "ClientDataReader::GetRequestByIndex: invalid index: " << request_index << std::endl;
  } else {
    std::string request_folder_name = this->index_to_folder_name_.at(request_index);
    client_request = this->GetRequestByRequestFoldername(request_folder_name);
  }
  return client_request;
}

}  // namespace offline

}  // namespace state_estimation
