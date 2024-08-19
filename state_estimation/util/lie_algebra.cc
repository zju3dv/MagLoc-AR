/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-15 11:40:11
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-15 14:38:51
 */
#include "util/lie_algebra.h"

namespace state_estimation {

namespace util {

const double epsilon = 1e-10;

Eigen::Matrix3d Vector3dHat(const Eigen::Vector3d& a) {
  Eigen::Matrix3d a_hat = Eigen::Matrix3d::Zero();
  a_hat(1, 0) = a(2);
  a_hat(2, 0) = -a(1);
  a_hat(0, 1) = -a(2);
  a_hat(2, 1) = a(0);
  a_hat(0, 2) = a(1);
  a_hat(1, 2) = -a(0);
  return a_hat;
}

Eigen::Vector3d Matrix3dVee(const Eigen::Matrix3d& a) {
  Eigen::Vector3d a_vee = Eigen::Vector3d::Zero();
  a_vee(0) = a(2, 1);
  a_vee(1) = a(0, 2);
  a_vee(2) = a(1, 0);
  return a_vee;
}

Eigen::Vector3d LieBracketso3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  // this is the same as cross-product
  Eigen::Matrix3d a_hat = Vector3dHat(a);
  Eigen::Matrix3d b_hat = Vector3dHat(b);
  Eigen::Vector3d lie_bracket = Matrix3dVee(a_hat * b_hat - b_hat * a_hat);
  return lie_bracket;
}

Eigen::Matrix3d JacobianLeftso3(const Eigen::Vector3d& so3) {
  double theta = so3.norm();
  Eigen::Vector3d a = so3 / theta;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d aat = a * a.transpose();
  Eigen::Matrix3d a_hat = Vector3dHat(a);
  Eigen::Matrix3d jacobian;
  if (theta < epsilon) {
    // return jacobian at zero
    jacobian = I;
  } else {
    jacobian = std::sin(theta) / theta * I + (1 - std::sin(theta) / theta) * aat + ((1 - std::cos(theta)) / theta) * a_hat;
  }
  return jacobian;
}

Eigen::Matrix3d JacobianLeftInverseso3(const Eigen::Vector3d& so3) {
  double theta = so3.norm();
  Eigen::Vector3d a = so3 / theta;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d aat = a * a.transpose();
  Eigen::Matrix3d a_hat = Vector3dHat(a);
  Eigen::Matrix3d jacobian_inverse;
  double c0 = theta / 2.0;
  double c1 = c0 * std::cos(c0) / std::sin(c0);
  if (theta < epsilon) {
    // return jacobian_inverse at zero
    jacobian_inverse = I;
  } else {
    jacobian_inverse = c1 * I + (1 - c1) * aat - c0 * a_hat;
  }
  return jacobian_inverse;
}

}  // namespace util

}  // namespace state_estimation
