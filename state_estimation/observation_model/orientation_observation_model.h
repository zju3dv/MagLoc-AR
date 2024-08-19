/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:17:21
 * @Last Modified by:   xuehua
 * @Last Modified time: 2020-12-24 16:17:21
 */
#ifndef STATE_ESTIMATION_OBSERVATION_MODEL_ORIENTATION_OBSERVATION_MODEL_H_
#define STATE_ESTIMATION_OBSERVATION_MODEL_ORIENTATION_OBSERVATION_MODEL_H_

#include <Eigen/Geometry>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "observation_model/base.h"
#include "util/variable_name_constants.h"
#include "util/misc.h"

namespace state_estimation {

namespace observation_model {

struct AttitudeData {
  double timestamp = -1.0;
  double time_unit_in_second = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 1.0;
};

class OrientationObservation : public Observation {
  // the observation of OrientationObservation is under the world coordinate system.
 public:
  void Init(double buffer_duration, Eigen::Matrix3d R_mw = Eigen::Matrix3d::Identity()) {
    this->buffer_duration_ = buffer_duration;
    this->R_mw_ = R_mw;
  }

  int GetObservationFromOrientationSensorLines(const std::vector<std::string>& orientation_lines, double target_timestamp);
  int GetObservationFromGravityAndGeomagnetismLines(
      const std::vector<std::string>& gravity_lines,
      const std::vector<std::string>& geomagnetism_lines,
      double target_timestamp);

  void GetObservationFromAttitudeData(AttitudeData attitude_data) {
    this->timestamp_ = attitude_data.timestamp * attitude_data.time_unit_in_second;
    this->q_ = Eigen::Quaterniond(attitude_data.w, attitude_data.x, attitude_data.y, attitude_data.z);
  }

  void GetObservationFromOrientation(double timestamp, Eigen::Quaterniond orientation_sensor_pose_ws) {
    this->timestamp_ = timestamp;
    this->q_ = orientation_sensor_pose_ws;
  }

  void GetObservationFromGT(double timestamp, Eigen::Quaterniond gt_orientation) {
    this->timestamp_ = timestamp;
    this->q_ = Eigen::Quaterniond(this->R_mw_.transpose() * gt_orientation.normalized());
  }

  std::vector<std::pair<std::string, double>> GetFeatureVector(void) {
    std::vector<std::pair<std::string, double>> feature_vector;
    feature_vector.push_back(std::pair<std::string, double>("orientation_q_w", this->q_.w()));
    feature_vector.push_back(std::pair<std::string, double>("orientation_q_x", this->q_.x()));
    feature_vector.push_back(std::pair<std::string, double>("orientation_q_y", this->q_.y()));
    feature_vector.push_back(std::pair<std::string, double>("orientation_q_z", this->q_.z()));
    return feature_vector;
  }

  Eigen::Vector3d GetEulerAngles(void);

  double timestamp(void) const {
    return this->timestamp_;
  }

  Eigen::Quaterniond q(void) const {
    return this->q_;
  }

  Eigen::Matrix3d R_mw(void) const {
    return this->R_mw_;
  }

  double buffer_duration(void) const {
    return this->buffer_duration_;
  }

  OrientationObservation(void) {
    this->timestamp_ = -1.0;
    this->q_ = Eigen::Quaterniond::Identity();
    this->buffer_duration_ = 0.0;
    this->R_mw_ = Eigen::Matrix3d::Identity();
  }

  ~OrientationObservation() {}

 private:
  double timestamp_;
  Eigen::Quaterniond q_;
  double buffer_duration_;
  Eigen::Matrix3d R_mw_;
};

class OrientationObservationYawState : public State {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNameYaw, this->yaw_);
  }

  std::string ToKey(void) {
    return "";
  }

  void FromPredictionModelState(const prediction_model::State* prediction_model_state_ptr) {
    std::vector<std::pair<std::string, double>> named_values;
    prediction_model_state_ptr->GetAllNamedValues(&named_values);
    for (int i = 0; i < named_values.size(); i++) {
      if (named_values.at(i).first == util::kNameYaw) {
        this->yaw_ = named_values.at(i).second;
        break;
      }
    }
  }

  double yaw(void) {
    return this->yaw_;
  }

  void yaw(double yaw) {
    this->yaw_ = yaw;
  }

  OrientationObservationYawState(void) {
    this->yaw_ = 0.0;
  }

  ~OrientationObservationYawState() {}

 private:
  double yaw_;
};

class OrientationObservationYawModel : public ObservationModel {
 public:
  void Init(double min_yaw_diff, double max_yaw_diff) {
    this->min_yaw_diff_ = min_yaw_diff;
    this->max_yaw_diff_ = max_yaw_diff;
  }

  double GetProbabilityObservationConditioningState(Observation* observation, State* state) const {
    return std::exp(this->GetProbabilityObservationConditioningStateLog(observation, state));
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation, State* state) const {
    OrientationObservation* my_observation = reinterpret_cast<OrientationObservation*>(observation);
    OrientationObservationYawState* my_state = reinterpret_cast<OrientationObservationYawState*>(state);

    Eigen::Quaterniond q_ws_orientation_sensor = my_observation->q().normalized();

    Eigen::Matrix3d R_z_ws_orientation_sensor = CalculateRzFromOrientation(q_ws_orientation_sensor);

    double state_yaw_w = my_state->yaw();
    Eigen::Matrix3d R_z_ws_state = Eigen::Quaterniond(Eigen::AngleAxisd(state_yaw_w, Eigen::Vector3d({0.0, 0.0, 1.0}))).normalized().toRotationMatrix();

    double yaw_diff = GetAngleByAxisFromAngleAxis(Eigen::AngleAxisd(R_z_ws_orientation_sensor * R_z_ws_state.transpose()), Eigen::Vector3d({0.0, 0.0, 1.0}));

    if ((yaw_diff <= this->max_yaw_diff_) && (yaw_diff >= this->min_yaw_diff_)) {
      return std::log(1.0 / (this->max_yaw_diff_ - this->min_yaw_diff_));
    } else {
      return std::log(0.0);
    }
  }

  std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const {
    std::unordered_map<std::string, double> empty_map;
    return empty_map;
  }

  OrientationObservationYawModel(void) {
    this->max_yaw_diff_ = M_PI;
    this->min_yaw_diff_ = -M_PI;
  }

  ~OrientationObservationYawModel() {}

 private:
  double max_yaw_diff_;
  double min_yaw_diff_;
};

}  // namespace observation_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OBSERVATION_MODEL_ORIENTATION_OBSERVATION_MODEL_H_
