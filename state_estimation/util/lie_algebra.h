/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-11 21:24:39
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-15 14:15:23
 */
#ifndef STATE_ESTIMATION_UTIL_LIE_ALGEBRA_H_
#define STATE_ESTIMATION_UTIL_LIE_ALGEBRA_H_

#include <Eigen/Dense>
#include <cmath>

namespace state_estimation {

namespace util {

Eigen::Matrix3d Vector3dHat(const Eigen::Vector3d& a);

Eigen::Vector3d Matrix3dVee(const Eigen::Matrix3d& a);

Eigen::Vector3d LieBracketso3(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

Eigen::Matrix3d JacobianLeftso3(const Eigen::Vector3d& so3);

Eigen::Matrix3d JacobianLeftInverseso3(const Eigen::Vector3d& so3);

Eigen::Matrix3d JacobianRightso3(const Eigen::Vector3d& so3);

Eigen::Matrix3d JacobianRightInverseso3(const Eigen::Vector3d& so3);

}  // namespace util

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_UTIL_LIE_ALGEBRA_H_
