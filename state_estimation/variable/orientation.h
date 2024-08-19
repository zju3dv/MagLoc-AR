/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:19:21
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-09 17:22:03
 */
#ifndef STATE_ESTIMATION_VARIABLE_ORIENTATION_H_
#define STATE_ESTIMATION_VARIABLE_ORIENTATION_H_

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <string>
#include <vector>

#include "util/misc.h"
#include "variable/base.h"

namespace state_estimation {

namespace variable {

class Orientation : public Variable {
 public:
  std::string ToKey(void);
  void FromKey(std::string orientation_key);

  void q(Eigen::Quaterniond q) {
    this->q_ = q;
  }

  Eigen::Quaterniond q(void) const {
    return this->q_;
  }

  void angle_axis(Eigen::AngleAxisd angle_axis) {
    this->q_ = Eigen::Quaterniond(angle_axis);
  }

  Eigen::AngleAxisd angle_axis(void) const {
    return Eigen::AngleAxisd(this->q_);
  }

  void rotation_matrix(Eigen::Matrix3d rotation_matrix) {
    this->q_ = Eigen::Quaterniond(rotation_matrix);
  }

  Eigen::Matrix3d rotation_matrix(void) const {
    return this->q_.toRotationMatrix();
  }

  void Normalize(void) {
    this->q_.normalize();
  }

  void Round(double orientation_resolution) {
    // TODO(xuehua): I need to define and implement a way to quantize the orientation.
  }

  static Orientation Mean(const std::vector<Orientation>& orientations, std::vector<double> weights = std::vector<double>()) {
    // Calculate the mean of orientations in the sense that minimize the summation of (sin(theta_i))^2,
    // where theta_i is the angular distance between the average quaternion and the i-th sample quaternion.
    std::vector<Eigen::Quaterniond> qs;
    for (int i = 0; i < orientations.size(); i++) {
      qs.push_back(orientations.at(i).q());
    }

    Eigen::Quaterniond q_mean = QuaternionGeometricMean(qs, weights);

    Orientation temp_orientation;
    temp_orientation.q(q_mean);
    temp_orientation.Normalize();

    return temp_orientation;
  }

  static Eigen::Quaterniond Mean(const std::vector<Eigen::Quaterniond>& qs, std::vector<double> weights = std::vector<double>()) {
    // Calculate the mean of orientations in the sense that minimize the summation of (sin(theta_i))^2,
    // where theta_i is the angular distance between the average quaternion and the i-th sample quaternion.
    return QuaternionGeometricMean(qs, weights);
  }

  static Eigen::Quaterniond Mean(const std::vector<Eigen::AngleAxisd>& angle_axises, std::vector<double> weights = std::vector<double>()) {
    // Calculate the mean of orientations in the sense that minimize the summation of (sin(theta_i))^2,
    // where theta_i is the angular distance between the average quaternion and the i-th sample quaternion.
    std::vector<Eigen::Quaterniond> qs;
    for (int i = 0; i < angle_axises.size(); i++) {
      qs.push_back(Eigen::Quaterniond(angle_axises.at(i)));
    }

    return QuaternionGeometricMean(qs, weights);
  }

  static double Variance(const std::vector<Orientation>& orientations, std::vector<double> weights = std::vector<double>()) {
    // Calculate the variance of orientations in the sense that the mean is minimizing the summation of (sin(theta_i))^2,
    // where theta_i is the angular distance between the average quaternion and the i-th sample quaternion.
    // the variance is calculated as the summation of (sin(theta_i))^2.
    std::vector<Eigen::Quaterniond> qs;
    for (int i = 0; i < orientations.size(); i++) {
      qs.push_back(orientations.at(i).q());
    }

    return QuaternionGeometricVariance(qs, weights);
  }

  static double Variance(const std::vector<Eigen::Quaterniond>& qs, std::vector<double> weights = std::vector<double>()) {
    // Calculate the variance of orientations in the sense that the mean is minimizing the summation of (sin(theta_i))^2,
    // where theta_i is the angular distance between the average quaternion and the i-th sample quaternion.
    // the variance is calculated as the summation of (sin(theta_i))^2.
    return QuaternionGeometricVariance(qs, weights);
  }

  static Eigen::Matrix<double, 3, 3> Covariance(const std::vector<Orientation>& orientations, std::vector<double> weights = std::vector<double>()) {
    // Calculate the covariance matrix in the form of angle-axis.
    std::vector<Eigen::Quaterniond> qs;
    for (int i = 0; i < orientations.size(); i++) {
      qs.push_back(orientations.at(i).q());
    }

    return QuaternionAngleAxisCovariance(qs, weights);
  }

  static Eigen::Matrix<double, 3, 3> Covariance(const std::vector<Eigen::Quaterniond>& qs, std::vector<double> weights = std::vector<double>()) {
    // Calculate the covariance matrix in the form of angle-axis.
    return QuaternionAngleAxisCovariance(qs, weights);
  }

  static Eigen::Matrix<double, 3, 3> Covariance(const std::vector<Eigen::AngleAxisd>& axs, std::vector<double> weights = std::vector<double>()) {
    // Calculate the covariance matrix in the form of angle-axis.
    std::vector<Eigen::Quaterniond> qs;
    for (int i = 0; i < axs.size(); i++) {
      qs.push_back(Eigen::Quaterniond(axs.at(i)));
    }

    return QuaternionAngleAxisCovariance(qs, weights);
  }

  Orientation(void);
  ~Orientation();

 private:
  Eigen::Quaterniond q_;
};

}  // namespace variable

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_VARIABLE_ORIENTATION_H_
