/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-10-11 16:36:35
 * @LastEditTime: 2023-01-29 21:35:06
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#include "util/result_format.h"

#include <sstream>
#include <string>
#include <vector>

namespace state_estimation {

namespace util {

std::string Result::OutputGTFormatEuRoC(void) {
  std::stringstream ss;
  ss.setf(std::ios::scientific, std::ios::floatfield);
  ss.precision(6);
  ss << DoubleToString(this->timestamp_ * 1.0e9, 0) << ","    // timestamp_ns
     << this->gt_position_.x() << ","                         // p_x
     << this->gt_position_.y() << ","                         // p_y
     << "0.0" << ","                                          // p_z
     << this->gt_orientation_.q().w() << ","                  // q_w
     << this->gt_orientation_.q().x() << ","                  // q_x
     << this->gt_orientation_.q().y() << ","                  // q_y
     << this->gt_orientation_.q().z() << ","                  // q_z
     << this->gt_v_2d_(0) << ","                              // v_x
     << this->gt_v_2d_(1) << ","                              // v_y
     << "0.0" << ","                                          // v_z
     << "0.0" << ","                                          // b_w_x
     << "0.0" << ","                                          // b_w_y
     << "0.0" << ","                                          // b_w_z
     << "0.0" << ","                                          // b_a_x
     << "0.0" << ","                                          // b_a_y
     << "0.0";                                                // b_a_z
  return ss.str();
}

std::string Result::OutputEstFormatEuRoC(void) {
  std::stringstream ss;
  ss.setf(std::ios::scientific, std::ios::floatfield);
  ss.precision(6);
  ss << DoubleToString(this->timestamp_ * 1.0e9, 0) << ","    // timestamp
     << this->est_position_.x() << ","                        // p_x
     << this->est_position_.y() << ","                        // p_y
     << "1.8" << ","                                          // p_z
     << this->est_orientation_.q().w() << ","                 // q_w
     << this->est_orientation_.q().x() << ","                 // q_x
     << this->est_orientation_.q().y() << ","                 // q_y
     << this->est_orientation_.q().z() << ","                 // q_z
     << this->est_v_2d_(0) << ","                             // v_x
     << this->est_v_2d_(1) << ","                             // v_y
     << "0.0" << ","                                          // v_z
     << "0.0" << ","                                          // b_w_x
     << "0.0" << ","                                          // b_w_y
     << "0.0" << ","                                          // b_w_z
     << "0.0" << ","                                          // b_a_x
     << "0.0" << ","                                          // b_a_y
     << "0.0";                                                // b_a_z
  return ss.str();
}

std::string Result::OutputGTFormatTUM(void) {
  std::stringstream ss;
  ss.setf(std::ios::scientific, std::ios::floatfield);
  ss.precision(6);
  ss << DoubleToString(this->timestamp_, 9) << ","    // timestamp
     << this->gt_position_.x() << ","                 // p_x
     << this->gt_position_.y() << ","                 // p_y
     << "0.0" << ","                                  // p_z
     << this->gt_orientation_.q().x() << ","          // q_x
     << this->gt_orientation_.q().y() << ","          // q_y
     << this->gt_orientation_.q().z() << ","          // q_z
     << this->gt_orientation_.q().w();                // q_w
  return ss.str();
}

std::string Result::OutputEstFormatTUM(void) {
  std::stringstream ss;
  ss.setf(std::ios::scientific, std::ios::floatfield);
  ss.precision(6);
  ss << DoubleToString(this->timestamp_, 9) << ","    // timestamp
     << this->est_position_.x() << ","                // p_x
     << this->est_position_.y() << ","                // p_y
     << "0.0" << ","                                  // p_z
     << this->est_orientation_.q().x() << ","         // q_x
     << this->est_orientation_.q().y() << ","         // q_y
     << this->est_orientation_.q().z() << ","         // q_z
     << this->est_orientation_.q().w();               // q_w
  return ss.str();
}

std::string Result::OutputGTEstFormatX(void) {
  std::stringstream ss;
  ss.setf(std::ios::scientific, std::ios::floatfield);
  ss.precision(6);
  ss << DoubleToString(this->timestamp_, 9) << ","    // timestamp
     << this->gt_position_.x() << ","                 // p_x_gt
     << this->est_position_.x() << ","                // p_x_est
     << this->gt_position_.y() << ","                 // p_y_gt
     << this->est_position_.y() << ","                // p_y_est
     << this->gt_v_2d_(0) << ","                      // v_x_gt
     << this->est_v_2d_(0) << ","                     // v_x_est
     << this->gt_v_2d_(1) << ","                      // v_y_gt
     << this->est_v_2d_(1) << ","                     // v_y_est
     << this->gt_yaw_ << ","                          // yaw_gt
     << this->est_yaw_;                               // yaw_est
  return ss.str();
}

std::string Result::OutputGTEstFormatY(void) {
  std::stringstream ss;
  ss.setf(std::ios::scientific, std::ios::floatfield);
  ss.precision(6);
  ss << DoubleToString(this->timestamp_, 9) << ","    // timestamp
     << this->gt_position_.x() << ","                 // p_x_gt
     << this->est_position_.x() << ","                // p_x_est
     << this->gt_position_.y() << ","                 // p_y_gt
     << this->est_position_.y() << ","                // p_y_est
     << this->gt_v_2d_(0) << ","                      // v_x_gt
     << this->est_v_2d_(0) << ","                     // v_x_est
     << this->gt_v_2d_(1) << ","                      // v_y_gt
     << this->est_v_2d_(1) << ","                     // v_y_est
     << this->gt_displacement_2d_(0) << ","           // dp_x_gt
     << this->gt_displacement_2d_(1) << ","           // dp_y_gt
     << this->control_displacement_2d_(0) << ","      // dp_x_control
     << this->control_displacement_2d_(1) << ","      // dp_y_control
     << this->est_displacement_2d_(0) << ","          // dp_x_est
     << this->est_displacement_2d_(1) << ","          // dp_y_est
     << this->distance_variance_;                     // dist_var
  return ss.str();
}

std::string Result::OutputGTEstFormatTUM(void) {
  std::stringstream ss;
  ss.setf(std::ios::scientific, std::ios::floatfield);
  ss.precision(6);
  ss << DoubleToString(this->timestamp_, 9) << ","    // timestamp
     << this->gt_position_.x() << ","                 // gt_p_x
     << this->est_position_.x() << ","                // est_p_x
     << this->gt_position_.y() << ","                 // gt_p_y
     << this->est_position_.y() << ","                // est_p_y
     << this->gt_position_.z() << ","                 // gt_p_z
     << this->est_position_.z() << ","                // est_p_z
     << this->gt_orientation_.q().x() << ","          // gt_q_x
     << this->est_orientation_.q().x() << ","         // est_q_x
     << this->gt_orientation_.q().y() << ","          // gt_q_y
     << this->est_orientation_.q().y() << ","         // est_q_y
     << this->gt_orientation_.q().z() << ","          // gt_q_z
     << this->est_orientation_.q().z() << ","         // est_q_z
     << this->gt_orientation_.q().w() << ","          // gt_q_w
     << this->est_orientation_.q().w();               // est_q_w
  return ss.str();
}

Result::Result(void) {
  this->timestamp_ = 0.0;
  this->gt_position_.x(0.0);
  this->gt_position_.y(0.0);
  this->gt_position_.floor(0);
  this->est_position_.x(0.0);
  this->est_position_.y(0.0);
  this->est_position_.floor(0);
  this->gt_heading_v_ = 0.0;
  this->est_heading_v_ = 0.0;
  this->gt_v_2d_.setZero();
  this->est_v_2d_.setZero();
  this->gt_orientation_.q(Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
  this->est_orientation_.q(Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
  this->gt_yaw_ = 0.0;
  this->est_yaw_ = 0.0;
  this->gt_omega_z_ = 0.0;
  this->est_omega_z_ = 0.0;
  this->gt_omega_ = Eigen::AngleAxisd(0.0, Eigen::Matrix<double, 3, 1>({0.0, 0.0, 1.0}));
  this->est_omega_ = Eigen::AngleAxisd(0.0, Eigen::Matrix<double, 3, 1>({0.0, 0.0, 1.0}));
  this->bias_1d_ = 0.0;
  this->bias_3d_.setZero();
  this->gt_displacement_2d_.setZero();
  this->est_displacement_2d_.setZero();
  this->control_displacement_2d_.setZero();
  this->distance_variance_ = 0.0;
  this->gt_bluetooth_offset_ = 0.0;
  this->est_bluetooth_offset_ = 0.0;
  this->gt_wifi_offset_ = 0.0;
  this->est_wifi_offset_ = 0.0;
  this->gt_geomagnetism_bias_3d_.setZero();
  this->est_geomagnetism_bias_3d_.setZero();
  this->log_probability_est_ = 0.0;
  this->log_probability_max_ = 0.0;
  this->log_probability_min_ = 0.0;
  this->log_probability_top_20_percent_ = 0.0;
  this->log_probability_top_50_percent_ = 0.0;
  this->log_probability_top_80_percent_ = 0.0;
  this->activity_type_ = ActivityType::kActivityUnknown;
}

}  // namespace util

}  // namespace state_estimation
