/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-10-11 16:32:51
 * @LastEditTime: 2023-01-29 21:28:14
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#ifndef STATE_ESTIMATION_UTIL_RESULT_FORMAT_H_
#define STATE_ESTIMATION_UTIL_RESULT_FORMAT_H_

#include <Eigen/Dense>

#include <string>

#include "variable/orientation.h"
#include "variable/position.h"

namespace state_estimation {

namespace util {

enum class ActivityType {
  kActivityStanding = 0,
  kActivityWalking,
  kActivityUnknown,
};

class Result {
 public:
  std::string OutputGTFormatEuRoC(void);
  std::string OutputEstFormatEuRoC(void);
  std::string OutputGTFormatTUM(void);
  std::string OutputEstFormatTUM(void);
  std::string OutputGTEstFormatX(void);
  std::string OutputGTEstFormatY(void);
  std::string OutputGTEstFormatTUM(void);
  Result(void);
  ~Result() {}

  double timestamp(void) const {
    return this->timestamp_;
  }

  void timestamp(double timestamp) {
    this->timestamp_ = timestamp;
  }

  variable::Position gt_position(void) const {
    return this->gt_position_;
  }

  void gt_position(variable::Position gt_position) {
    this->gt_position_ = gt_position;
  }

  variable::Position est_position(void) const {
    return this->est_position_;
  }

  void est_position(variable::Position est_position) {
    this->est_position_ = est_position;
  }

  double gt_heading_v(void) const {
    return this->gt_heading_v_;
  }

  void gt_heading_v(double gt_heading_v) {
    this->gt_heading_v_ = gt_heading_v;
  }

  double est_heading_v(void) const {
    return this->est_heading_v_;
  }

  void est_heading_v(double est_heading_v) {
    this->est_heading_v_ = est_heading_v;
  }

  Eigen::Matrix<double, 2, 1> gt_v_2d(void) const {
    return this->gt_v_2d_;
  }

  void gt_v_2d(Eigen::Matrix<double, 2, 1> gt_v_2d) {
    this->gt_v_2d_ = gt_v_2d;
  }

  Eigen::Matrix<double, 2, 1> est_v_2d(void) const {
    return this->est_v_2d_;
  }

  void est_v_2d(Eigen::Matrix<double, 2, 1> est_v_2d) {
    this->est_v_2d_ = est_v_2d;
  }

  variable::Orientation gt_orientation(void) const {
    return this->gt_orientation_;
  }

  void gt_orientation(variable::Orientation gt_orientation) {
    this->gt_orientation_ = gt_orientation;
  }

  variable::Orientation est_orientation(void) const {
    return this->est_orientation_;
  }

  void est_orientation(variable::Orientation est_orientation) {
    this->est_orientation_ = est_orientation;
  }

  double gt_yaw(void) const {
    return this->gt_yaw_;
  }

  void gt_yaw(double gt_yaw) {
    this->gt_yaw_ = gt_yaw;
  }

  double est_yaw(void) const {
    return this->est_yaw_;
  }

  void est_yaw(double est_yaw) {
    this->est_yaw_ = est_yaw;
  }

  double gt_omega_z(void) const {
    return this->gt_omega_z_;
  }

  void gt_omega_z(double gt_omega_z) {
    this->gt_omega_z_ = gt_omega_z;
  }

  double est_omega_z(void) const {
    return this->est_omega_z_;
  }

  void est_omega_z(double est_omega_z) {
    this->est_omega_z_ = est_omega_z;
  }

  Eigen::AngleAxisd gt_omega(void) const {
    return this->gt_omega_;
  }

  void gt_omega(Eigen::AngleAxisd gt_omega) {
    this->gt_omega_ = gt_omega;
  }

  Eigen::AngleAxisd est_omega(void) const {
    return this->est_omega_;
  }

  void est_omega(Eigen::AngleAxisd est_omega) {
    this->est_omega_ = est_omega;
  }

  double bias_1d(void) const {
    return this->bias_1d_;
  }

  void bias_1d(double bias_1d) {
    this->bias_1d_ = bias_1d;
  }

  Eigen::Matrix<double, 3, 1> bias_3d(void) const {
    return this->bias_3d_;
  }

  void bias_3d(Eigen::Matrix<double, 3, 1> bias_3d) {
    this->bias_3d_ = bias_3d;
  }

  Eigen::Matrix<double, 2, 1> gt_displacement_2d(void) const {
    return this->gt_displacement_2d_;
  }

  void gt_displacement_2d(Eigen::Matrix<double, 2, 1> gt_displacement_2d) {
    this->gt_displacement_2d_ = gt_displacement_2d;
  }

  Eigen::Matrix<double, 2, 1> est_displacement_2d(void) const {
    return this->est_displacement_2d_;
  }

  void est_displacement_2d(Eigen::Matrix<double, 2, 1> est_displacement_2d) {
    this->est_displacement_2d_ = est_displacement_2d;
  }

  Eigen::Matrix<double, 2, 1> control_displacement_2d(void) const {
    return this->control_displacement_2d_;
  }

  void control_displacement_2d(Eigen::Matrix<double, 2, 1> control_displacement_2d) {
    this->control_displacement_2d_ = control_displacement_2d;
  }

  double distance_variance(void) const {
    return this->distance_variance_;
  }

  void distance_variance(double distance_variance) {
    this->distance_variance_ = distance_variance;
  }

  double gt_bluetooth_offset(void) const {
    return this->gt_bluetooth_offset_;
  }

  void gt_bluetooth_offset(double gt_bluetooth_offset) {
    this->gt_bluetooth_offset_ = gt_bluetooth_offset;
  }

  double est_bluetooth_offset(void) const {
    return this->est_bluetooth_offset_;
  }

  void est_bluetooth_offset(double est_bluetooth_offset) {
    this->est_bluetooth_offset_ = est_bluetooth_offset;
  }

  double gt_wifi_offset(void) const {
    return this->gt_wifi_offset_;
  }

  void gt_wifi_offset(double gt_wifi_offset) {
    this->gt_wifi_offset_ = gt_wifi_offset;
  }

  double est_wifi_offset(void) const {
    return this->est_wifi_offset_;
  }

  void est_wifi_offset(double est_wifi_offset) {
    this->est_wifi_offset_ = est_wifi_offset;
  }

  Eigen::Vector3d gt_geomagnetism_bias_3d(void) const {
    return this->gt_geomagnetism_bias_3d_;
  }

  void gt_geomagnetism_bias_3d(Eigen::Vector3d gt_geomagnetism_bias_3d) {
    this->gt_geomagnetism_bias_3d_ = gt_geomagnetism_bias_3d;
  }

  Eigen::Vector3d est_geomagnetism_bias_3d(void) const {
    return this->est_geomagnetism_bias_3d_;
  }

  void est_geomagnetism_bias_3d(Eigen::Vector3d est_geomagnetism_bias_3d) {
    this->est_geomagnetism_bias_3d_ = est_geomagnetism_bias_3d;
  }

  double log_probability_est(void) const {
    return this->log_probability_est_;
  }

  void log_probability_est(double log_probability_est) {
    this->log_probability_est_ = log_probability_est;
  }

  double log_probability_max(void) const {
    return this->log_probability_max_;
  }

  void log_probability_max(double log_probability_max) {
    this->log_probability_max_ = log_probability_max;
  }

  double log_probability_min(void) const {
    return this->log_probability_min_;
  }

  void log_probability_min(double log_probability_min) {
    this->log_probability_min_ = log_probability_min;
  }

  double log_probability_top_20_percent(void) const {
    return this->log_probability_top_20_percent_;
  }

  void log_probability_top_20_percent(double log_probability_top_20_percent) {
    this->log_probability_top_20_percent_ = log_probability_top_20_percent;
  }

  double log_probability_top_50_percent(void) const {
    return this->log_probability_top_50_percent_;
  }

  void log_probability_top_50_percent(double log_probability_top_50_percent) {
    this->log_probability_top_50_percent_ = log_probability_top_50_percent;
  }

  double log_probability_top_80_percent(void) const {
    return this->log_probability_top_80_percent_;
  }

  void log_probability_top_80_percent(double log_probability_top_80_percent) {
    this->log_probability_top_80_percent_ = log_probability_top_80_percent;
  }

  ActivityType activity_type(void) const {
    return this->activity_type_;
  }

  void activity_type(ActivityType activity_type) {
    this->activity_type_ = activity_type;
  }

 private:
  double timestamp_;
  variable::Position gt_position_;
  variable::Position est_position_;
  double gt_heading_v_;
  double est_heading_v_;
  Eigen::Matrix<double, 2, 1> gt_v_2d_;
  Eigen::Matrix<double, 2, 1> est_v_2d_;
  variable::Orientation gt_orientation_;
  variable::Orientation est_orientation_;
  double gt_yaw_;
  double est_yaw_;
  double gt_omega_z_;
  double est_omega_z_;
  Eigen::AngleAxisd gt_omega_;
  Eigen::AngleAxisd est_omega_;
  double bias_1d_;
  Eigen::Matrix<double, 3, 1> bias_3d_;
  Eigen::Matrix<double, 2, 1> gt_displacement_2d_;
  Eigen::Matrix<double, 2, 1> est_displacement_2d_;
  Eigen::Matrix<double, 2, 1> control_displacement_2d_;
  double distance_variance_;
  double gt_bluetooth_offset_;
  double est_bluetooth_offset_;
  double gt_wifi_offset_;
  double est_wifi_offset_;
  Eigen::Vector3d gt_geomagnetism_bias_3d_;
  Eigen::Vector3d est_geomagnetism_bias_3d_;
  double log_probability_est_;
  double log_probability_max_;
  double log_probability_min_;
  double log_probability_top_20_percent_;
  double log_probability_top_50_percent_;
  double log_probability_top_80_percent_;
  ActivityType activity_type_;
};

}  // namespace util

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_UTIL_RESULT_FORMAT_H_
