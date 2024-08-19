/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-09-16 19:57:26
 * @LastEditTime: 2021-09-16 19:57:26
 * @LastEditors: xuehua
 */
#include "observation_model/imu_observation_model.h"

#include <string>
#include <utility>
#include <vector>

#include "util/misc.h"

namespace state_estimation {

namespace observation_model {

void GyroscopeObservation::Init(double buffer_duration) {
  this->buffer_duration_ = buffer_duration;
}

int GyroscopeObservation::GetObservationFromLines(const std::vector<std::string>& gyroscope_lines, double target_timestamp) {
  int timestamp_index = 0;
  int number_of_items_in_line = 4;
  int line_index;
  double time_difference =
      GetTimeClosestLineIndex(gyroscope_lines,
                              timestamp_index,
                              number_of_items_in_line,
                              target_timestamp,
                              &line_index);
  std::string gyroscope_line = gyroscope_lines[line_index];
  std::vector<std::string> gyroscope_line_split;
  SplitString(gyroscope_line, gyroscope_line_split, ",");
  this->omega_x_ = std::stod(gyroscope_line_split[1]);
  this->omega_y_ = std::stod(gyroscope_line_split[2]);
  this->omega_z_ = std::stod(gyroscope_line_split[3]);
  this->timestamp_ = target_timestamp;

  return gyroscope_lines.size();
}

int GyroscopeObservation::GetSequentialObservationFromLines(const std::vector<std::string>& gyroscope_lines) {
  this->sequential_timestamps_.clear();
  this->sequential_observations_.clear();
  std::vector<std::string> gyroscope_line_split;
  for (int i = 0; i < gyroscope_lines.size(); i++) {
    SplitString(gyroscope_lines.at(i), gyroscope_line_split, ",");
    if (gyroscope_line_split.size() != 4) {
      continue;
    }
    this->sequential_timestamps_.push_back(std::stod(gyroscope_line_split.at(0)));
    Eigen::Matrix<double, 3, 1> feature_vector = {std::stod(gyroscope_line_split.at(1)),
                                                  std::stod(gyroscope_line_split.at(2)),
                                                  std::stod(gyroscope_line_split.at(3))};
    this->sequential_observations_.push_back(feature_vector);
  }
  return gyroscope_lines.size();
}

std::vector<std::pair<std::string, double>> GyroscopeObservation::GetFeatureVector(void) {
  return std::vector<std::pair<std::string, double>>({std::pair<std::string, double>("omega_x", this->omega_x_),
                                                      std::pair<std::string, double>("omega_y", this->omega_y_),
                                                      std::pair<std::string, double>("omega_z", this->omega_z_)});
}

Eigen::Matrix<double, 3, 1> GyroscopeObservation::GetEigenVector(void) {
  return Eigen::Matrix<double, 3, 1>({this->omega_x_, this->omega_y_, this->omega_z_});
}

double GyroscopeObservation::timestamp(void) {
  return this->timestamp_;
}

double GyroscopeObservation::omega_x(void) {
  return this->omega_x_;
}

double GyroscopeObservation::omega_y(void) {
  return this->omega_y_;
}

double GyroscopeObservation::omega_z(void) {
  return this->omega_z_;
}

double GyroscopeObservation::buffer_duration(void) {
  return this->buffer_duration_;
}

std::vector<double> GyroscopeObservation::sequential_timestamps(void) {
  return this->sequential_timestamps_;
}

std::vector<Eigen::Matrix<double, 3, 1>> GyroscopeObservation::sequential_observations(void) {
  return this->sequential_observations_;
}

GyroscopeObservation::GyroscopeObservation(void) {
  this->timestamp_ = 0.0;
  this->omega_x_ = 0.0;
  this->omega_y_ = 0.0;
  this->omega_z_ = 0.0;
  this->buffer_duration_ = 0.0;
  this->sequential_timestamps_ = std::vector<double>();
  this->sequential_observations_ = std::vector<Eigen::Matrix<double, 3, 1>>();
}

GyroscopeObservation::~GyroscopeObservation(void) {}

void AccelerometerObservation::Init(double buffer_duration) {
  this->buffer_duration_ = buffer_duration;
}

int AccelerometerObservation::GetObservationFromLines(const std::vector<std::string>& accelerometer_lines, double target_timestamp) {
  int timestamp_index = 0;
  int number_of_items_in_line = 4;
  int line_index;
  double time_difference =
      GetTimeClosestLineIndex(accelerometer_lines,
                              timestamp_index,
                              number_of_items_in_line,
                              target_timestamp,
                              &line_index);
  std::string accelerometer_line = accelerometer_lines[line_index];
  std::vector<std::string> accelerometer_line_split;
  SplitString(accelerometer_line, accelerometer_line_split, ",");
  this->acc_x_ = std::stod(accelerometer_line_split[1]);
  this->acc_y_ = std::stod(accelerometer_line_split[2]);
  this->acc_z_ = std::stod(accelerometer_line_split[3]);
  this->timestamp_ = target_timestamp;

  return accelerometer_lines.size();
}

int AccelerometerObservation::GetSequentialObservationFromLines(const std::vector<std::string>& accelerometer_lines) {
  this->sequential_timestamps_.clear();
  this->sequential_observations_.clear();
  std::vector<std::string> accelerometer_line_split;
  for (int i = 0; i < accelerometer_lines.size(); i++) {
    SplitString(accelerometer_lines.at(i), accelerometer_line_split, ",");
    if (accelerometer_line_split.size() != 4) {
      continue;
    }
    this->sequential_timestamps_.push_back(std::stod(accelerometer_line_split.at(0)));
    Eigen::Matrix<double, 3, 1> feature_vector = {std::stod(accelerometer_line_split.at(1)),
                                                  std::stod(accelerometer_line_split.at(2)),
                                                  std::stod(accelerometer_line_split.at(3))};
    this->sequential_observations_.push_back(feature_vector);
  }
  return accelerometer_lines.size();
}

std::vector<std::pair<std::string, double>> AccelerometerObservation::GetFeatureVector(void) {
  return std::vector<std::pair<std::string, double>>({std::pair<std::string, double>("acc_x", this->acc_x_),
                                                      std::pair<std::string, double>("acc_y", this->acc_y_),
                                                      std::pair<std::string, double>("acc_z", this->acc_z_)});
}

Eigen::Matrix<double, 3, 1> AccelerometerObservation::GetEigenVector(void) {
  return Eigen::Matrix<double, 3, 1>({this->acc_x_, this->acc_y_, this->acc_z_});
}

double AccelerometerObservation::timestamp(void) {
  return this->timestamp_;
}

double AccelerometerObservation::acc_x(void) {
  return this->acc_x_;
}

double AccelerometerObservation::acc_y(void) {
  return this->acc_y_;
}

double AccelerometerObservation::acc_z(void) {
  return this->acc_z_;
}

double AccelerometerObservation::buffer_duration(void) {
  return this->buffer_duration_;
}

std::vector<double> AccelerometerObservation::sequential_timestamps(void) {
  return this->sequential_timestamps_;
}

std::vector<Eigen::Matrix<double, 3, 1>> AccelerometerObservation::sequential_observations(void) {
  return this->sequential_observations_;
}

AccelerometerObservation::AccelerometerObservation(void) {
  this->timestamp_ = 0.0;
  this->acc_x_ = 0.0;
  this->acc_y_ = 0.0;
  this->acc_z_ = 0.0;
  this->buffer_duration_ = 0.0;
  this->sequential_timestamps_ = std::vector<double>();
  this->sequential_observations_ = std::vector<Eigen::Matrix<double, 3, 1>>();
}

AccelerometerObservation::~AccelerometerObservation(void) {}

}  // namespace observation_model

}  // namespace state_estimation
