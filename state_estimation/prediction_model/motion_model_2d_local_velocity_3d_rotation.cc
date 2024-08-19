/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-12-22 10:50:00
 * @LastEditTime: 2021-12-22 13:16:48
 * @LastEditors: xuehua
 */
#include "prediction_model/motion_model_2d_local_velocity_3d_rotation.h"

#include <cmath>
#include <iostream>
#include <mutex>

#include "distribution/gaussian_distribution.h"

namespace state_estimation {

namespace prediction_model {

void MotionModel2dLocalVelocity3dRotation::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
#ifdef DEBUG_FOCUSING
  std::cout << "MotionModel2dLocalVelocity3dRotation::Predict" << std::endl;
#endif
  MotionModel2dLocalVelocity3dRotationState* my_state_t = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_t);
  MotionModel2dLocalVelocity3dRotationState* my_state_tminus = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_tminus);
  MotionModel2dLocalVelocity3dRotationControlInput* my_control_input_t = reinterpret_cast<MotionModel2dLocalVelocity3dRotationControlInput*>(control_input_t);

  // predict for MotionModel1dRotationState
  MotionModel3dOrientationDifferentialControlInput rotation_control_input = my_control_input_t->motion_model_3d_rotation_control_input();
  this->motion_model_3d_rotation_.Predict(my_state_t->motion_model_3d_orientation_state_ptr(),
                                          my_state_tminus->motion_model_3d_orientation_state_ptr(),
                                          &rotation_control_input,
                                          dt);

  // sample the control_input vector
  Eigen::Vector3d v_local_mean = my_control_input_t->v_local();

  Eigen::Vector3d v_local_sampled;
  {
    std::lock_guard<std::mutex> guard(this->my_mutex_);
    this->mvg_sampler_.SetParams(v_local_mean, my_control_input_t->v_local_covariance());
    v_local_sampled = this->mvg_sampler_.Sample();
  }

  Eigen::Vector3d v_m_sampled = Eigen::Vector3d::Zero();
  if (my_control_input_t->INS_orientation_estimation()) {
    Eigen::Quaterniond q_ws = my_state_t->motion_model_3d_orientation_state().orientation();
    v_m_sampled = this->R_mw_ * q_ws * v_local_sampled;
  } else {
    v_m_sampled = this->R_mw_ * my_control_input_t->orientation_sensor_pose_ws() * v_local_sampled;
  }

  // predict positions
  variable::Position position_tminus = my_state_tminus->position();
  variable::Position position_t;
  position_t.x(position_tminus.x() + v_m_sampled(0) * dt);
  position_t.y(position_tminus.y() + v_m_sampled(1) * dt);
  position_t.floor(position_tminus.floor());
  my_state_t->position(position_t);
}

void MotionModel2dLocalVelocity3dRotation::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  MotionModel2dLocalVelocity3dRotationState* my_state_t = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_t);
  MotionModel2dLocalVelocity3dRotationState* my_state_tminus = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_tminus);
  my_state_t->position(my_state_tminus->position());
  this->motion_model_3d_rotation_.PredictWithoutControlInput(my_state_t->motion_model_3d_orientation_state_ptr(), my_state_tminus->motion_model_3d_orientation_state_ptr(), dt);
}

double MotionModel2dLocalVelocity3dRotation::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  MotionModel2dLocalVelocity3dRotationState* my_state_t = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_t);
  MotionModel2dLocalVelocity3dRotationState* my_state_tminus = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_tminus);
  MotionModel2dLocalVelocity3dRotationControlInput* my_control_input_t = reinterpret_cast<MotionModel2dLocalVelocity3dRotationControlInput*>(control_input_t);

  MotionModel3dOrientationDifferentialControlInput orientation_control_input = my_control_input_t->motion_model_3d_rotation_control_input();
  double orientation_log_prob = this->motion_model_3d_rotation_.CalculateStateTransitionProbabilityLog(my_state_t->motion_model_3d_orientation_state_ptr(),
                                                                                                       my_state_tminus->motion_model_3d_orientation_state_ptr(),
                                                                                                       &orientation_control_input,
                                                                                                       dt);

  Eigen::Quaterniond q_ws = my_state_t->motion_model_3d_orientation_state().orientation();
  Eigen::Vector3d v_local_mean = my_control_input_t->v_local();

  Eigen::Matrix3d R_ms = Eigen::Matrix3d::Identity();
  if (my_control_input_t->INS_orientation_estimation()) {
    R_ms = this->R_mw_ * q_ws;
  } else {
    R_ms = this->R_mw_ * my_control_input_t->orientation_sensor_pose_ws();
  }
  Eigen::Vector3d v_m_mean = R_ms * v_local_mean;
  Eigen::Matrix3d v_m_covariance = R_ms * my_control_input_t->v_local_covariance() * R_ms.transpose();

  std::vector<double> v_m_2d_mean_vector = {v_m_mean(0), v_m_mean(1)};
  std::vector<double> v_m_2d_covariance_vector = {v_m_covariance(0, 0),
                                                  v_m_covariance(0, 1),
                                                  v_m_covariance(1, 1)};
  distribution::MultivariateGaussian mvg(v_m_2d_mean_vector, v_m_2d_covariance_vector);

  double v_m_x = (my_state_t->position().x() - my_state_tminus->position().x()) / dt;
  double v_m_y = (my_state_t->position().y() - my_state_tminus->position().y()) / dt;
  std::vector<double> x = {v_m_x, v_m_y};

  return std::log(mvg.QuantizedProbability(x)) + orientation_log_prob;
}

}  // namespace prediction_model

}  // namespace state_estimation
