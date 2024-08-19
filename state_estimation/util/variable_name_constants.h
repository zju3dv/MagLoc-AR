/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-09-16 11:44:36
 * @LastEditTime: 2023-01-29 20:51:46
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#ifndef STATE_ESTIMATION_UTIL_VARIABLE_NAME_CONSTANTS_H_
#define STATE_ESTIMATION_UTIL_VARIABLE_NAME_CONSTANTS_H_

#include <string>

namespace state_estimation {

namespace util {

const char kNamePositionX[] = "position_x";
const char kNamePositionY[] = "position_y";
const char kNamePositionZ[] = "position_z";
const char kNameHeadingYaw[] = "heading_yaw";
const char kNameHeadingV[] = "heading_v";
const char kNameVx[] = "v_x";
const char kNameVy[] = "v_y";
const char kNameVLocalX[] = "v_local_x";
const char kNameVLocalY[] = "v_local_y";
const char kNameVLocalZ[] = "v_local_z";
const char kNameYaw[] = "yaw";
const char kNameOrientation1d[] = "orientation_1d";
const char kNameOrientationW[] = "orientation_w";
const char kNameOrientationX[] = "orientation_x";
const char kNameOrientationY[] = "orientation_y";
const char kNameOrientationZ[] = "orientation_z";
const char kNameOmegaAngle[] = "omega_angle";
const char kNameOmegaAxisX[] = "omega_axis_x";
const char kNameOmegaAxisY[] = "omega_axis_y";
const char kNameOmegaAxisZ[] = "omega_axis_z";
const char kNameAccX[] = "acc_x";
const char kNameAccY[] = "acc_y";
const char kNameDyaw[] = "delta_yaw";
const char kNameDOrientation1d[] = "delta_orientation_1d";
const char kNameOmegaX[] = "omega_x";
const char kNameOmegaY[] = "omega_y";
const char kNameOmegaZ[] = "omega_z";
const char kNameAlphaX[] = "alpha_x";
const char kNameAlphaY[] = "alpha_y";
const char kNameAlphaZ[] = "alpha_z";
const char kNameOmegaAngleaxisAccX[] = "omega_angleaxis_acc_x";
const char kNameOmegaAngleaxisAccY[] = "omega_angleaxis_acc_y";
const char kNameOmegaAngleaxisAccZ[] = "omega_angleaxis_acc_z";
const char kNameDpositionX[] = "delta_position_x";
const char kNameDpositionY[] = "delta_position_y";
const char kNameBluetoothDynamicOffset[] = "bluetooth_dynamic_offset";
const char kNameWifiDynamicOffset[] = "wifi_dynamic_offset";
const char kNameGeomagnetismBiasX[] = "geomagnetism_bias_x";
const char kNameGeomagnetismBiasY[] = "geomagnetism_bias_y";
const char kNameGeomagnetismBiasZ[] = "geomagnetism_bias_z";
const char kNameGeomagnetismBiasCovarianceXX[] = "geomagnetism_bias_covariance_xx";
const char kNameGeomagnetismBiasCovarianceYY[] = "geomagnetism_bias_covariance_yy";
const char kNameGeomagnetismBiasCovarianceZZ[] = "geomagnetism_bias_covariance_zz";
const char kNameGeomagnetismBiasCovarianceXY[] = "geomagnetism_bias_covariance_xy";
const char kNameGeomagnetismBiasCovarianceYX[] = "geomagnetism_bias_covariance_yx";
const char kNameGeomagnetismBiasCovarianceXZ[] = "geomagnetism_bias_covariance_xz";
const char kNameGeomagnetismBiasCovarianceZX[] = "geomagnetism_bias_covariance_zx";
const char kNameGeomagnetismBiasCovarianceYZ[] = "geomagnetism_bias_covariance_yz";
const char kNameGeomagnetismBiasCovarianceZY[] = "geomagnetism_bias_covariance_zy";
const char kNameGravityX[] = "gravity_x";
const char kNameGravityY[] = "gravity_y";
const char kNameGravityZ[] = "gravity_z";

}  // namespace util

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_UTIL_VARIABLE_NAME_CONSTANTS_H_
