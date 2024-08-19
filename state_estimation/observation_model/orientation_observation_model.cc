/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:17:12
 * @Last Modified by:   xuehua
 * @Last Modified time: 2020-12-24 16:17:12
 */
#include "observation_model/orientation_observation_model.h"

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <string>

#include "util/misc.h"

namespace state_estimation {

namespace observation_model {

static void ToEulerAngle(const Eigen::Quaterniond& q,
                         double& roll,
                         double& pitch,
                         double& yaw) {
  // roll (x-axis rotation)
  double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
  double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
  roll = atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
  if (fabs(sinp) >= 1)
    pitch = copysign(M_PI / 2, sinp);  // use 90 degrees if out of range
  else
    pitch = asin(sinp);

  // yaw (z-axis rotation)
  double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
  double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
  yaw = atan2(siny_cosp, cosy_cosp);
}

int OrientationObservation::GetObservationFromOrientationSensorLines(
    const std::vector<std::string>& orientation_lines,
    double target_timestamp) {
  if (orientation_lines.size() == 0) {
    return 0;
  }
  int timestamp_index_orientation = 0;
  int number_of_items_in_line_orientation = 6;
  int line_index_orientation;
  double time_difference_orientation = GetTimeClosestLineIndex(orientation_lines,
                                                               timestamp_index_orientation,
                                                               number_of_items_in_line_orientation,
                                                               target_timestamp,
                                                               &line_index_orientation);
  std::string orientation_line = orientation_lines.at(line_index_orientation);
  double orientation[4]{0.0, 0.0, 0.0, 0.0};  // x, y, z, w
  std::vector<std::string> orientation_line_split;
  SplitString(orientation_line, orientation_line_split, ",");
  this->timestamp_ = std::stod(orientation_line_split[0]);
  orientation[0] = std::stod(orientation_line_split[1]);
  orientation[1] = std::stod(orientation_line_split[2]);
  orientation[2] = std::stod(orientation_line_split[3]);
  orientation[3] = std::stod(orientation_line_split[4]);
  this->q_ = Eigen::Quaterniond(orientation);

  return orientation_lines.size();
}

int OrientationObservation::GetObservationFromGravityAndGeomagnetismLines(
    const std::vector<std::string>& gravity_lines,
    const std::vector<std::string>& geomagnetism_lines,
    double target_timestamp) {
  if (gravity_lines.size() == 0 || geomagnetism_lines.size() == 0) {
    return 0;
  }
  int timestamp_index_gravity = 0;
  int number_of_items_in_line_gravity = 4;
  int line_index_gravity;
  double time_difference_gravity =
      GetTimeClosestLineIndex(gravity_lines,
                              timestamp_index_gravity,
                              number_of_items_in_line_gravity,
                              target_timestamp,
                              &line_index_gravity);
  std::string gravity_line = gravity_lines[line_index_gravity];
  std::vector<std::string> gravity_line_split;
  SplitString(gravity_line, gravity_line_split, ",");
  Eigen::Vector3d gravity;
  gravity << std::stod(gravity_line_split[1]),
             std::stod(gravity_line_split[2]),
             std::stod(gravity_line_split[3]);

  int timestamp_index_geomagnetism = 0;
  int number_of_items_in_line_geomagnetism = 5;
  int line_index_geomagnetism;
  double time_difference_geomagnetism =
      GetTimeClosestLineIndex(geomagnetism_lines,
                              timestamp_index_geomagnetism,
                              number_of_items_in_line_geomagnetism,
                              target_timestamp,
                              &line_index_geomagnetism);
  std::string geomagnetism_line = geomagnetism_lines[line_index_geomagnetism];
  std::vector<std::string> geomagnetism_line_split;
  SplitString(geomagnetism_line, geomagnetism_line_split, ",");
  Eigen::Vector3d geomagnetism;
  geomagnetism << std::stod(geomagnetism_line_split[1]),
                  std::stod(geomagnetism_line_split[2]),
                  std::stod(geomagnetism_line_split[3]);

  gravity = gravity / gravity.norm();

  Eigen::Vector3d h = geomagnetism.cross(gravity);
  h = h / h.norm();

  Eigen::Vector3d m = gravity.cross(h);

  Eigen::Matrix3d R_ws;
  R_ws << h, m, gravity;
  R_ws.transposeInPlace();

  this->q_ = Eigen::Quaterniond(R_ws);

  return gravity_lines.size();
}

Eigen::Vector3d OrientationObservation::GetEulerAngles(void) {
  double roll, pitch, yaw;
  ToEulerAngle(this->q_, roll, pitch, yaw);
  Eigen::Vector3d euler_angles = {roll, pitch, yaw};
  return euler_angles;
}

}  // namespace observation_model

}  // namespace state_estimation
