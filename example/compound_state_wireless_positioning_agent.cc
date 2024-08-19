/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-12-27 15:30:00
 * @LastEditTime: 2021-12-27 15:36:45
 * @LastEditors: xuehua
 */
#include <string>
#include <vector>

#include "prediction_model/compound_prediction_model.h"
#include "prediction_model/motion_model_2d_local_velocity_1d_rotation.h"
#include "prediction_model/motion_model_yaw_differential.h"
#include "prediction_model/parameter_model_random_walk.h"
#include "prediction_model/base.h"
#include "util/variable_name_constants.h"
#include "variable/position.h"

using state_estimation::prediction_model::State;
using state_estimation::prediction_model::CompoundPredictionModelState;
using state_estimation::prediction_model::CompoundPredictionModelControlInput;
using state_estimation::prediction_model::CompoundPredictionModel;
using state_estimation::prediction_model::MotionModel2dLocalVelocity1dRotationState;
using state_estimation::prediction_model::MotionModel2dLocalVelocity1dRotationControlInput;
using state_estimation::prediction_model::MotionModel2dLocalVelocity1dRotation;
using state_estimation::prediction_model::MotionModelYawDifferentialState;
using state_estimation::prediction_model::ParameterModelRandomWalkState;
using state_estimation::prediction_model::ParameterModelRandomWalkControlInput;
using state_estimation::variable::Position;

CompoundPredictionModelState::CompoundPredictionModelState(void) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelState::CompoundPredictionModelState" << std::endl;
#endif
  this->submodel_state_ptrs_.emplace_back(new MotionModel2dLocalVelocity1dRotationState());
  ParameterModelRandomWalkState* temp = new ParameterModelRandomWalkState();
  temp->Init(5);
  temp->parameter_names(std::vector<std::string>({util::kNameBluetoothDynamicOffset,
                                                  util::kNameWifiDynamicOffset,
                                                  util::kNameGeomagnetismBiasX,
                                                  util::kNameGeomagnetismBiasY,
                                                  util::kNameGeomagnetismBiasZ}));
  this->submodel_state_ptrs_.emplace_back(temp);
  this->numbers_of_submodel_state_variables_ = std::vector<int>({MotionModel2dLocalVelocity1dRotationState::kNumberOfStateVariables, 5});
  this->state_prediction_log_probability_ = 0.0;
  this->state_update_log_probability_ = 0.0;
}

CompoundPredictionModelState& CompoundPredictionModelState::operator=(const CompoundPredictionModelState& compound_state) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelState::operator=" << std::endl;
#endif
  // self-assignment detection
  if (&compound_state == this) {
    return *this;
  }

  // release any resource we're holding
  for (int i = 0; i < this->submodel_state_ptrs_.size(); i++) {
    delete this->submodel_state_ptrs_[i];
  }
  this->submodel_state_ptrs_.clear();

  // copy the resource
  this->submodel_state_ptrs_.emplace_back(new MotionModel2dLocalVelocity1dRotationState());
  this->submodel_state_ptrs_.emplace_back(new ParameterModelRandomWalkState());
  *reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(this->submodel_state_ptrs_.at(0)) = *reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(compound_state.at(0));
  *reinterpret_cast<ParameterModelRandomWalkState*>(this->submodel_state_ptrs_.at(1)) = *reinterpret_cast<ParameterModelRandomWalkState*>(compound_state.at(1));
  this->numbers_of_submodel_state_variables_ = compound_state.numbers_of_submodel_state_variables();
  this->state_prediction_log_probability_ = compound_state.state_prediction_log_probability();
  this->state_update_log_probability_ = compound_state.state_update_log_probability();

  return *this;
}

CompoundPredictionModelState::CompoundPredictionModelState(const CompoundPredictionModelState& compound_state) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelState::CompoundPredictionModelState(CompoundPredictionModelState&)" << std::endl;
#endif
  this->submodel_state_ptrs_.emplace_back(new MotionModel2dLocalVelocity1dRotationState());
  this->submodel_state_ptrs_.emplace_back(new ParameterModelRandomWalkState());
  *reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(this->submodel_state_ptrs_.at(0)) = *reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(compound_state.at(0));
  *reinterpret_cast<ParameterModelRandomWalkState*>(this->submodel_state_ptrs_.at(1)) = *reinterpret_cast<ParameterModelRandomWalkState*>(compound_state.at(1));
  this->numbers_of_submodel_state_variables_ = compound_state.numbers_of_submodel_state_variables();
  this->state_prediction_log_probability_ = compound_state.state_prediction_log_probability();
  this->state_update_log_probability_ = compound_state.state_update_log_probability();
}

CompoundPredictionModelState::~CompoundPredictionModelState(void) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelState::~CompoundPredictionModelState" << std::endl;
#endif
  for (int i = 0; i < this->submodel_state_ptrs_.size(); i++) {
    if (this->submodel_state_ptrs_.at(i)) {
      delete this->submodel_state_ptrs_.at(i);
    }
  }
}

Position CompoundPredictionModelState::position(void) {
  Position temp_position;
  if (this->submodel_state_ptrs_.size() < 1) {
    std::cout << "CompoundPredictionModelState::position: "
              << "The instance is not initialized."
              << std::endl;
    return temp_position;
  }
  MotionModel2dLocalVelocity1dRotationState* temp_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(this->submodel_state_ptrs_.at(0));
  temp_position = temp_ptr->position();
  return temp_position;
}

void CompoundPredictionModelState::position(Position position) {
  if (this->submodel_state_ptrs_.size() < 1) {
    std::cout << "CompoundPredictionModelState::position(position): "
              << "The instance is not initialized."
              << std::endl;
  }
  MotionModel2dLocalVelocity1dRotationState* temp_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(this->submodel_state_ptrs_.at(0));
  temp_ptr->position(position);
}

double CompoundPredictionModelState::yaw(void) {
  if (this->submodel_state_ptrs_.size() < 1) {
    std::cout << "CompoundPredictionModelState::yaw: "
              << "The instance is not initialized."
              << std::endl;
    return 0.0;
  }
  MotionModel2dLocalVelocity1dRotationState* temp_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(this->submodel_state_ptrs_.at(0));
  return temp_ptr->motion_model_1d_rotation_state_ptr()->yaw();
}

void CompoundPredictionModelState::yaw(double yaw) {
  if (this->submodel_state_ptrs_.size() < 1) {
    std::cout << "CompoundPredictionModelState::yaw(yaw): "
              << "The instance is not initialized."
              << std::endl;
  }
  MotionModel2dLocalVelocity1dRotationState* temp_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(this->submodel_state_ptrs_.at(0));
  temp_ptr->motion_model_1d_rotation_state_ptr()->yaw(yaw);
}

Eigen::Quaterniond CompoundPredictionModelState::q_ws(void) {
  if (this->submodel_state_ptrs_.size() < 1) {
    std::cout << "CompoundPredictionModelState::q_ws: "
              << "The instance is not initialized."
              << std::endl;
    return Eigen::Quaterniond::Identity();
  }
  MotionModel2dLocalVelocity1dRotationState* temp_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(this->submodel_state_ptrs_.at(0));
  return temp_ptr->motion_model_1d_rotation_state_ptr()->q_ws();
}

double CompoundPredictionModelState::bluetooth_offset(void) {
  if (this->submodel_state_ptrs_.size() < 1) {
    std::cout << "CompoundPredictionModelState::bluetooth_offset: "
              << "The instance is not initialized."
              << std::endl;
    return 0.0;
  }
  ParameterModelRandomWalkState* temp_ptr = reinterpret_cast<ParameterModelRandomWalkState*>(this->submodel_state_ptrs_.at(1));
  return temp_ptr->parameters()(0);
}

double CompoundPredictionModelState::wifi_offset(void) {
  if (this->submodel_state_ptrs_.size() < 1) {
    std::cout << "CompoundPredictionModelState::wifi_offset: "
              << "The instance is not initialized."
              << std::endl;
    return 0.0;
  }
  ParameterModelRandomWalkState* temp_ptr = reinterpret_cast<ParameterModelRandomWalkState*>(this->submodel_state_ptrs_.at(1));
  return temp_ptr->parameters()(1);
}

Eigen::Vector3d CompoundPredictionModelState::geomagnetism_bias(void) {
  if (this->submodel_state_ptrs_.size() < 1) {
    std::cout << "CompoundPredictionModelState::geomagnetism_bias: "
              << "The instance is not initialized."
              << std::endl;
    return Eigen::Vector3d::Zero();
  }
  ParameterModelRandomWalkState* temp_ptr = reinterpret_cast<ParameterModelRandomWalkState*>(this->submodel_state_ptrs_.at(1));
  Eigen::Vector3d geomagnetism_bias = Eigen::Vector3d::Zero();
  geomagnetism_bias(0) = temp_ptr->parameters()(2);
  geomagnetism_bias(1) = temp_ptr->parameters()(3);
  geomagnetism_bias(2) = temp_ptr->parameters()(4);
  return geomagnetism_bias;
}

void CompoundPredictionModelState::geomagnetism_bias(Eigen::Vector3d geomagnetism_bias) {
  ParameterModelRandomWalkState* temp_ptr = reinterpret_cast<ParameterModelRandomWalkState*>(this->submodel_state_ptrs_.at(1));
  Eigen::Matrix<double, 5, 1> parameters = temp_ptr->parameters();
  parameters(2) = geomagnetism_bias(0);
  parameters(3) = geomagnetism_bias(1);
  parameters(4) = geomagnetism_bias(2);
  temp_ptr->parameters(parameters);
}

CompoundPredictionModelControlInput::CompoundPredictionModelControlInput(void) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelControlInput::CompoundPredictionModelControlInput" << std::endl;
#endif
  this->submodel_control_input_ptrs_.emplace_back(new MotionModel2dLocalVelocity1dRotationControlInput());
  ParameterModelRandomWalkControlInput* temp = new ParameterModelRandomWalkControlInput();
  temp->Init(5);
  this->submodel_control_input_ptrs_.emplace_back(temp);
}

CompoundPredictionModelControlInput::CompoundPredictionModelControlInput(const CompoundPredictionModelControlInput& compound_control_input) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelControlInput::CompoundPredictionModelControlInput(CompoundPredictionModelControlInput&)" << std::endl;
#endif
  this->submodel_control_input_ptrs_.emplace_back(new MotionModel2dLocalVelocity1dRotationControlInput());
  this->submodel_control_input_ptrs_.emplace_back(new ParameterModelRandomWalkControlInput());
  *reinterpret_cast<MotionModel2dLocalVelocity1dRotationControlInput*>(this->submodel_control_input_ptrs_.at(0)) = *reinterpret_cast<MotionModel2dLocalVelocity1dRotationControlInput*>(compound_control_input.at(0));
  *reinterpret_cast<ParameterModelRandomWalkControlInput*>(this->submodel_control_input_ptrs_.at(1)) = *reinterpret_cast<ParameterModelRandomWalkControlInput*>(compound_control_input.at(1));
}

CompoundPredictionModelControlInput& CompoundPredictionModelControlInput::operator=(const CompoundPredictionModelControlInput& compound_control_input) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelControlInput::operator=" << std::endl;
#endif
  if (&compound_control_input == this) {
    return *this;
  }

  for (int i = 0; i < this->submodel_control_input_ptrs_.size(); i++) {
    delete this->submodel_control_input_ptrs_.at(i);
  }
  this->submodel_control_input_ptrs_.clear();

  this->submodel_control_input_ptrs_.emplace_back(new MotionModel2dLocalVelocity1dRotationControlInput());
  this->submodel_control_input_ptrs_.emplace_back(new ParameterModelRandomWalkControlInput());
  *reinterpret_cast<MotionModel2dLocalVelocity1dRotationControlInput*>(this->submodel_control_input_ptrs_.at(0)) = *reinterpret_cast<MotionModel2dLocalVelocity1dRotationControlInput*>(compound_control_input.at(0));
  *reinterpret_cast<ParameterModelRandomWalkControlInput*>(this->submodel_control_input_ptrs_.at(1)) = *reinterpret_cast<ParameterModelRandomWalkControlInput*>(compound_control_input.at(1));

  return *this;
}

CompoundPredictionModelControlInput::~CompoundPredictionModelControlInput() {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelControlInput::~CompoundPredictionModelControlInput" << std::endl;
#endif
  for (int i = 0; i < this->submodel_control_input_ptrs_.size(); i++) {
    if (this->submodel_control_input_ptrs_.at(i)) {
      delete this->submodel_control_input_ptrs_.at(i);
    }
  }
}

void CompoundPredictionModel::JitterState(State* state_t) {
  CompoundPredictionModelState* my_state_t = reinterpret_cast<CompoundPredictionModelState*>(state_t);
  MotionModel2dLocalVelocity1dRotation* motion_model_2d_local_velocity_1d_rotation_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotation*>(this->submodel_ptrs_.at(0));
  ParameterModelRandomWalk* parameter_model_random_walk_ptr = reinterpret_cast<ParameterModelRandomWalk*>(this->submodel_ptrs_.at(1));
  motion_model_2d_local_velocity_1d_rotation_ptr->JitterState(my_state_t->at(0));
  parameter_model_random_walk_ptr->JitterState(my_state_t->at(1));
  double state_prediction_log_probability = 0.0;
  for (int i = 0; i < this->submodel_ptrs_.size(); i++) {
    state_prediction_log_probability += my_state_t->at(i)->state_prediction_log_probability();
  }
  my_state_t->state_prediction_log_probability(state_prediction_log_probability);
}

void CompoundPredictionModel::JitterState(State* state_t, Eigen::Vector3d geomagnetism_s, Eigen::Vector3d gravity_s) {
  CompoundPredictionModelState* my_state_t = reinterpret_cast<CompoundPredictionModelState*>(state_t);
  MotionModel2dLocalVelocity1dRotation* motion_model_2d_local_velocity_1d_rotation_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotation*>(this->submodel_ptrs_.at(0));
  ParameterModelRandomWalk* parameter_model_random_walk_ptr = reinterpret_cast<ParameterModelRandomWalk*>(this->submodel_ptrs_.at(1));

  MotionModel2dLocalVelocity1dRotationState* my_state_motion_model_2d_local_velocity_1d_rotation_t = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(my_state_t->at(0));
  MotionModelYawDifferentialState* my_state_yaw_differential_t = my_state_motion_model_2d_local_velocity_1d_rotation_t->motion_model_1d_rotation_state_ptr();
  ParameterModelRandomWalkState* my_state_parameter_model_random_walk_t = reinterpret_cast<ParameterModelRandomWalkState*>(my_state_t->at(1));

  double yaw_w_t = my_state_yaw_differential_t->yaw();
  Eigen::Matrix<double, 5, 1> parameters_t = my_state_parameter_model_random_walk_t->parameters();
  Eigen::Vector3d geomagnetism_bias_t = parameters_t.block(2, 0, 3, 1);

  motion_model_2d_local_velocity_1d_rotation_ptr->JitterState(my_state_t->at(0));

  double yaw_w_t_jittered = my_state_yaw_differential_t->yaw();

  Eigen::Quaterniond q_sgs = Eigen::Quaterniond::FromTwoVectors(gravity_s, Eigen::Vector3d({0.0, 0.0, 1.0}));
  Eigen::Quaterniond q_wsg(Eigen::AngleAxisd(yaw_w_t, Eigen::Vector3d({0.0, 0.0, 1.0})));
  Eigen::Quaterniond q_wsg_jittered(Eigen::AngleAxisd(yaw_w_t_jittered, Eigen::Vector3d({0.0, 0.0, 1.0})));
  q_sgs.normalize();
  q_wsg.normalize();
  q_wsg_jittered.normalize();
  Eigen::Matrix3d R_ws = (q_wsg * q_sgs).normalized().toRotationMatrix();
  Eigen::Matrix3d R_ws_jittered = (q_wsg_jittered * q_sgs).normalized().toRotationMatrix();

  Eigen::Vector3d geomagnetism_bias_t_jittered = geomagnetism_s - R_ws_jittered.transpose() * R_ws * (geomagnetism_s - geomagnetism_bias_t);

  parameters_t.block(2, 0, 3, 1) = geomagnetism_bias_t_jittered;

  my_state_parameter_model_random_walk_t->parameters(parameters_t);

  double state_prediction_log_probability = 0.0;
  for (int i = 0; i < this->submodel_ptrs_.size(); i++) {
    state_prediction_log_probability += my_state_t->at(i)->state_prediction_log_probability();
  }
  my_state_t->state_prediction_log_probability(state_prediction_log_probability);
}
