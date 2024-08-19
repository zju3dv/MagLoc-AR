/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-06-08 16:50:10
 * @LastEditTime: 2021-08-02 16:11:16
 * @LastEditors: xuehua
 */
#include "observation_model/geomagnetism_observation_model.h"

#include <Eigen/Dense>

#include "util/misc.h"

namespace state_estimation {

namespace observation_model {

bool GetGeomagnetismDataFromLine(const std::string &line_data, GeomagnetismData &geomagnetism_data, double time_unit_in_second) {
  std::vector<std::string> line_split;
  SplitString(line_data, line_split, ",");
  if (line_split.size() < 4) {
    return false;
  }
  geomagnetism_data.time_unit_in_second = time_unit_in_second;
  geomagnetism_data.timestamp = std::stod(line_split.at(0));
  geomagnetism_data.x = std::stod(line_split.at(1));
  geomagnetism_data.y = std::stod(line_split.at(2));
  geomagnetism_data.z = std::stod(line_split.at(3));
  if (line_split.size() >= 5) {
    geomagnetism_data.device_accuracy = std::stoi(line_split.at(4));
  }
  return true;
}

void GeomagnetismObservation::Init(double buffer_duration,
                                   double timestamp,
                                   double time_unit_in_second,
                                   std::vector<double> R_mw_vector,
                                   GeomagnetismFeatureVectorType feature_vector_type) {
  this->buffer_duration_ = buffer_duration;
  this->timestamp(timestamp, time_unit_in_second);
  if (R_mw_vector.size() != 9) {
    std::cout << "GeomagnetismObservation::Init: the provided R_mw_vector does not have size 9." << std::endl;
  }
  assert(R_mw_vector.size() == 9);
  this->R_mw_ = VectorTo2DMatrixC(R_mw_vector, 3);
  this->feature_vector_type_ = feature_vector_type;
}

void AlignFeatureVector(std::vector<std::pair<std::string, double>>* named_feature_vector, variable::Orientation orientation) {
  Eigen::Vector3d feature_vector;
  for (int i = 0; i < named_feature_vector->size(); i++) {
    if (named_feature_vector->at(i).first == "x") {
      feature_vector(0) = named_feature_vector->at(i).second;
    }
    if (named_feature_vector->at(i).first == "y") {
      feature_vector(1) = named_feature_vector->at(i).second;
    }
    if (named_feature_vector->at(i).first == "z") {
      feature_vector(2) = named_feature_vector->at(i).second;
    }
  }
  Eigen::Quaterniond q = orientation.q();
  Eigen::Vector3d feature_vector_aligned = q.toRotationMatrix() * feature_vector;
  for (int i = 0; i < named_feature_vector->size(); i++) {
    if (named_feature_vector->at(i).first == "x") {
      named_feature_vector->at(i).second = feature_vector_aligned(0);
    }
    if (named_feature_vector->at(i).first == "y") {
      named_feature_vector->at(i).second = feature_vector_aligned(1);
    }
    if (named_feature_vector->at(i).first == "z") {
      named_feature_vector->at(i).second = feature_vector_aligned(2);
    }
  }
}

int GeomagnetismObservation::GetObservationFromLines(
    const std::vector<std::string>& geomagnetism_lines,
    double time_unit_in_second) {
  std::vector<GeomagnetismData> geomagnetism_datas;
  for (int i = 0; i < geomagnetism_lines.size(); i++) {
    GeomagnetismData data;
    if (GetGeomagnetismDataFromLine(geomagnetism_lines.at(i), data, time_unit_in_second)) {
      geomagnetism_datas.push_back(std::move(data));
    }
  }
  return this->GetObservationFromGeomagnetismDatas(geomagnetism_datas);
}

int GeomagnetismObservation::GetObservationFromGeomagnetismDatas(const std::vector<GeomagnetismData> &geomagnetism_datas) {
  // assume that the GeomagnetismDatas are sorted in time order.
  // the last is the most current.
  this->feature_values_.setZero();
  if (geomagnetism_datas.size() == 0) {
    return 0;
  }
  double current_t = this->timestamp_ * this->time_unit_in_second_;
  if (current_t < 0.0) {
    std::cout << "GeomagnetismObservation::GetObservationFromGeomagnetismDatas: the timestamp of GeomagnetismObservation is not assigned." << std::endl;
    return 0;
  }
  int valid_data_counter = 0;
  for (int i = geomagnetism_datas.size() - 1; i >= 0; i--) {
    GeomagnetismData data = geomagnetism_datas.at(i);
    double data_t = data.timestamp * data.time_unit_in_second;
    if (data_t > current_t) {
      // the data is in the future.
      continue;
    }
    if (current_t - data_t > this->buffer_duration_) {
      // the data is out-of-date.
      break;
    }
    this->feature_values_(0) = data.x;
    this->feature_values_(1) = data.y;
    this->feature_values_(2) = data.z;
    valid_data_counter++;
    break;
  }
  return valid_data_counter;
}

int GeomagnetismObservation::GetRwsFromOrientationLines(
        const std::vector<std::string>& orientation_lines,
        double target_timestamp) {
  if (orientation_lines.size() == 0) {
    return 0;
  }
  int timestamp_index_orientation = 0;
  int number_of_items_in_line_orientation = 5;
  int line_index_orientation;
  double time_difference_orientation =
      GetTimeClosestLineIndex(orientation_lines,
                              timestamp_index_orientation,
                              number_of_items_in_line_orientation,
                              target_timestamp,
                              &line_index_orientation);
  std::string orientation_line = orientation_lines[line_index_orientation];
  std::vector<std::string> orientation_line_split;
  SplitString(orientation_line, orientation_line_split, ",");
  // the quaternion in the attitude/rv.csv is recorded as x, y, z, w.
  Eigen::Quaterniond orientation(std::stod(orientation_line_split[4]),
                                 std::stod(orientation_line_split[1]),
                                 std::stod(orientation_line_split[2]),
                                 std::stod(orientation_line_split[3]));

  orientation.normalize();

  this->R_ws_ = orientation.toRotationMatrix();

  return orientation_lines.size();
}

int GeomagnetismObservation::GetRwsFromGravityAndGeomagnetismLines(
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

  this->R_ws_ = R_ws;

  return gravity_lines.size();
}

int GeomagnetismObservation::GetRwsFromGT(const Eigen::Quaterniond& gt_orientation) {
  Eigen::Quaterniond q = gt_orientation;
  q.normalize();
  this->R_ws_ = this->R_mw_.transpose() * q.toRotationMatrix();
  return 1;
}

int GeomagnetismObservation::GetRwsFromGravityAndRwsgAngle(const std::vector<std::string>& gravity_lines, double R_wsg_angle, double target_timestamp) {
  if (gravity_lines.size() == 0) {
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

  Eigen::Quaterniond q_sgs = Eigen::Quaterniond::FromTwoVectors(gravity, Eigen::Vector3d({0.0, 0.0, 1.0}));
  q_sgs.normalize();
  Eigen::AngleAxisd angleaxis_wsg(R_wsg_angle, Eigen::Vector3d({0.0, 0.0, 1.0}));

  this->R_ws_ = angleaxis_wsg.toRotationMatrix() * q_sgs.toRotationMatrix();
  return 1;
}

int GeomagnetismObservation::GetRwsFromRwsgAngle(double R_wsg_angle) {
  Eigen::AngleAxisd angleaxis_wsg(R_wsg_angle, Eigen::Vector3d({0.0, 0.0, 1.0}));
  this->R_ws_ = angleaxis_wsg.toRotationMatrix() * this->R_sgs_;
  return 1;
}

int GeomagnetismObservation::GetRsgsFromGravity(void) {
  this->R_sgs_ = Eigen::Quaterniond::FromTwoVectors(this->gravity_, Eigen::Vector3d({0.0, 0.0, 1.0})).toRotationMatrix();
  return 1;
}

int GeomagnetismObservation::GetGravityFromLines(const std::vector<std::string>& gravity_lines, double target_timestamp) {
  if (gravity_lines.size() == 0) {
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

  this->gravity_ = gravity;
  return gravity_lines.size();
}

int GeomagnetismObservation::GetGravityFromGT(const Eigen::Quaterniond& gt_orientation) {
  // the GT sensor is measuring R_ms.
  Eigen::Matrix<double, 3, 1> z_vector = {0.0, 0.0, 1.0};
  Eigen::Matrix<double, 3, 3> R_ws = this->R_mw_.transpose() * gt_orientation.toRotationMatrix();
  this->gravity_ = R_ws.transpose() * z_vector;
  return 1;
}

int GeomagnetismObservation::GetGravityFromOrientationLines(const std::vector<std::string> orientation_lines, double target_timestamp) {
  // the orientation sensor is measuring R_ws.
  if (orientation_lines.size() == 0) {
    return 0;
  }
  int timestamp_index_orientation = 0;
  int number_of_items_in_line_orientation = 6;
  int line_index_orientation;
  double time_difference_orientation =
      GetTimeClosestLineIndex(orientation_lines,
                              timestamp_index_orientation,
                              number_of_items_in_line_orientation,
                              target_timestamp,
                              &line_index_orientation);
  std::string orientation_line = orientation_lines[line_index_orientation];
  std::vector<std::string> orientation_line_split;
  SplitString(orientation_line, orientation_line_split, ",");
  // the quaternion in the attitude/rv.csv is recorded as x, y, z, w.
  Eigen::Quaterniond q_ws(std::stod(orientation_line_split[4]),
                          std::stod(orientation_line_split[1]),
                          std::stod(orientation_line_split[2]),
                          std::stod(orientation_line_split[3]));

  q_ws.normalize();

  Eigen::Matrix<double, 3, 3> R_ws = q_ws.toRotationMatrix();

  this->gravity_ = R_ws.transpose() * Eigen::Matrix<double, 3, 1>({0.0, 0.0, 1.0});
  return 1;
}

int GeomagnetismObservation::GetRwsgFromGT(const Eigen::Quaterniond& gt_orientation) {
  Eigen::Matrix<double, 3, 1> z_vector = {0.0, 0.0, 1.0};
  Eigen::Matrix<double, 3, 3> R_ws = this->R_mw_.transpose() * gt_orientation.toRotationMatrix();
  Eigen::Matrix<double, 3, 1> gravity = R_ws.transpose() * z_vector;
  Eigen::Quaterniond q_sgs = Eigen::Quaterniond::FromTwoVectors(gravity, z_vector);
  q_sgs.normalize();
  this->R_wsg_ = R_ws * q_sgs.toRotationMatrix().transpose();
  return 1;
}

int GeomagnetismObservation::GetRwsgFromRwsAndGravity(void) {
  Eigen::Quaterniond q_sgs = Eigen::Quaterniond::FromTwoVectors(this->gravity_, Eigen::Matrix<double, 3, 1>({0.0, 0.0, 1.0}));
  q_sgs.normalize();
  this->R_wsg_ = this->R_ws_ * q_sgs.toRotationMatrix().transpose();
  return 1;
}

std::vector<std::pair<std::string, double>> GeomagnetismObservation::GetFeatureVector(void) {
  // the feature vector is under the coordinate of R_ms.
#ifdef DEBUG_FOCUSING
  std::cout << "GeomagnetismObservation::GetFeatureVector" << std::endl;
#endif
  double x = this->feature_values_(0) - this->bias_(0);
  double y = this->feature_values_(1) - this->bias_(1);
  double z = this->feature_values_(2) - this->bias_(2);

  std::vector<std::pair<std::string, double>> feature_vector;
  Eigen::Vector3d v {x, y, z};
  Eigen::Vector3d rotated_v {0.0, 0.0, 0.0};
  switch (this->feature_vector_type_) {
    case GeomagnetismFeatureVectorType::kOverallMagnitude:
      double magnitude;
      magnitude = std::sqrt(x * x + y * y + z * z);
      feature_vector.push_back(std::pair<std::string, double>("magnitude", magnitude));
      break;
    case GeomagnetismFeatureVectorType::kTwoDimensionalMagnitude:
      rotated_v = this->R_mw_ * this->R_ws_ * v;
      double horizontal_magnitude;
      horizontal_magnitude = std::sqrt(rotated_v(0) * rotated_v(0) + rotated_v(1) * rotated_v(1));
      double vertical_magnitude;
      vertical_magnitude = std::abs(rotated_v(2));
      feature_vector.push_back(std::pair<std::string, double>("horizontal_magnitude", horizontal_magnitude));
      feature_vector.push_back(std::pair<std::string, double>("vertical_magnitude", vertical_magnitude));
      break;
    case GeomagnetismFeatureVectorType::kThreeDimensionalVector:
      rotated_v = this->R_mw_ * this->R_ws_ * v;
      feature_vector.push_back(std::pair<std::string, double>("x", rotated_v(0)));
      feature_vector.push_back(std::pair<std::string, double>("y", rotated_v(1)));
      feature_vector.push_back(std::pair<std::string, double>("z", rotated_v(2)));
      break;
    default:
      std::cout << "GeomagnetismObservation::GetFeatureVector: FeatureVectorType "
                << static_cast<int>(this->feature_vector_type_)
                << " is not allowed. Use the default three-axis feature vector."
                << std::endl;
      rotated_v = this->R_mw_ * this->R_ws_ * v;
      feature_vector.push_back(std::pair<std::string, double>("x", rotated_v(0)));
      feature_vector.push_back(std::pair<std::string, double>("y", rotated_v(1)));
      feature_vector.push_back(std::pair<std::string, double>("z", rotated_v(2)));
  }

  return feature_vector;
}

}  // namespace observation_model

}  // namespace state_estimation
