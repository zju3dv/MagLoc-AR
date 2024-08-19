/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-12-22 10:50:00
 * @LastEditTime: 2023-01-29 21:15:04
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#include "prediction_model/motion_model_2d_local_velocity_1d_rotation.h"

#include <cmath>
#include <iostream>
#include <mutex>

#include "distribution/gaussian_distribution.h"
#include "util/misc.h"

namespace state_estimation {

namespace prediction_model {

void MotionModel2dLocalVelocity1dRotation::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction) {
#ifdef DEBUG_FOCUSING
  std::cout << "MotionModel2dLocalVelocity1dRotation::Predict" << std::endl;
#endif
  MotionModel2dLocalVelocity1dRotationState* my_state_t = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_t);
  MotionModel2dLocalVelocity1dRotationState* my_state_tminus = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_tminus);
  MotionModel2dLocalVelocity1dRotationControlInput* my_control_input_t = reinterpret_cast<MotionModel2dLocalVelocity1dRotationControlInput*>(control_input_t);

  // predict for MotionModel1dRotationState
  MotionModelYawDifferentialControlInput rotation_control_input = my_control_input_t->motion_model_1d_rotation_control_input();
  this->motion_model_1d_rotation_.Predict(my_state_t->motion_model_1d_rotation_state_ptr(),
                                          my_state_tminus->motion_model_1d_rotation_state_ptr(),
                                          &rotation_control_input,
                                          dt,
                                          ideal_prediction);

  // sample the control_input vector
  Eigen::Vector3d v_local_mean = my_control_input_t->v_local();

  Eigen::Vector3d v_local_sampled;
  if (ideal_prediction) {
    v_local_sampled = v_local_mean;
  } else {
    std::lock_guard<std::mutex> guard(this->my_mutex_);
    this->mvg_sampler_.SetParams(v_local_mean, my_control_input_t->v_local_covariance());
    v_local_sampled = this->mvg_sampler_.Sample();
  }

  Eigen::Vector3d v_m_sampled = Eigen::Vector3d::Zero();
  if (my_control_input_t->use_estimated_yaw()) {
    Eigen::Quaterniond q_ws;
    if (dt > 0.0) {
      q_ws = my_state_t->motion_model_1d_rotation_state().q_ws();
    } else {
      q_ws = my_state_tminus->motion_model_1d_rotation_state().q_ws();
    }
    v_m_sampled = this->R_mw_ * q_ws * v_local_sampled;
  } else {
    v_m_sampled = this->R_mw_ * my_control_input_t->orientation_sensor_pose_ws() * v_local_sampled;
  }

  // predict positions
  variable::Position position_tminus = my_state_tminus->position();
  variable::Position position_t;
  position_t.x(position_tminus.x() + v_m_sampled(0) * dt);
  position_t.y(position_tminus.y() + v_m_sampled(1) * dt);
  position_t.z(position_tminus.z() + v_m_sampled(2) * dt);
  position_t.floor(position_tminus.floor());
  my_state_t->position(position_t);

  // calculate state_prediction_log_probability for the next state.
  std::vector<double> v_local_sampled_vector = EigenVector2Vector(v_local_sampled);
  std::vector<double> v_local_mean_vector = EigenVector2Vector(v_local_mean);
  std::vector<double> v_local_covariance_vector = CovarianceMatrixToCompactVector(my_control_input_t->v_local_covariance());
  distribution::MultivariateGaussian mvg_v_local(v_local_mean_vector, v_local_covariance_vector);
  double rotation_additional_prediction_log_probability = my_state_t->motion_model_1d_rotation_state_ptr()->state_prediction_log_probability()
                                                          - my_state_tminus->motion_model_1d_rotation_state_ptr()->state_prediction_log_probability();
  my_state_t->state_prediction_log_probability(my_state_tminus->state_prediction_log_probability() + std::log(mvg_v_local.QuantizedProbability(v_local_sampled_vector)) + rotation_additional_prediction_log_probability);
  my_state_t->state_update_log_probability(my_state_tminus->state_update_log_probability());
}

void MotionModel2dLocalVelocity1dRotation::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  MotionModel2dLocalVelocity1dRotationState* my_state_t = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_t);
  MotionModel2dLocalVelocity1dRotationState* my_state_tminus = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_tminus);
  my_state_t->position(my_state_tminus->position());
  this->motion_model_1d_rotation_.PredictWithoutControlInput(my_state_t->motion_model_1d_rotation_state_ptr(), my_state_tminus->motion_model_1d_rotation_state_ptr(), dt);
}

double MotionModel2dLocalVelocity1dRotation::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  MotionModel2dLocalVelocity1dRotationState* my_state_t = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_t);
  MotionModel2dLocalVelocity1dRotationState* my_state_tminus = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_tminus);
  MotionModel2dLocalVelocity1dRotationControlInput* my_control_input_t = reinterpret_cast<MotionModel2dLocalVelocity1dRotationControlInput*>(control_input_t);

  MotionModelYawDifferentialControlInput orientation_control_input = my_control_input_t->motion_model_1d_rotation_control_input();
  double orientation_log_prob = this->motion_model_1d_rotation_.CalculateStateTransitionProbabilityLog(my_state_t->motion_model_1d_rotation_state_ptr(),
                                                                                                       my_state_tminus->motion_model_1d_rotation_state_ptr(),
                                                                                                       &orientation_control_input,
                                                                                                       dt);

  double yaw_wsg;
  Eigen::Quaterniond q_ws;
  if (dt > 0.0) {
    q_ws = my_state_t->motion_model_1d_rotation_state().q_ws();
  } else {
    q_ws = my_state_tminus->motion_model_1d_rotation_state().q_ws();
  }

  Eigen::Vector3d v_local_mean = my_control_input_t->v_local();

  Eigen::Matrix3d R_ms = Eigen::Matrix3d::Identity();
  if (my_control_input_t->use_estimated_yaw()) {
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

void MotionModel2dLocalVelocity1dRotation::JitterState(State* state_t) {
  MotionModel2dLocalVelocity1dRotationState* my_state_t = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_t);
  if (this->position_jitter_flag_) {
    variable::Position position_t = my_state_t->position();
    Eigen::Vector2d position_jitter_delta = this->position_jitter_mvg_sampler_.Sample();

    Eigen::Vector2d position_jitter_mean = this->position_jitter_mvg_sampler_.mean();
    Eigen::Matrix2d position_jitter_covariance = this->position_jitter_mvg_sampler_.covariance();

    std::vector<double> position_jitter_mean_vector = {position_jitter_mean(0), position_jitter_mean(1)};
    std::vector<double> position_jitter_covariance_vector = {position_jitter_covariance(0, 0),
                                                             position_jitter_covariance(0, 1),
                                                             position_jitter_covariance(1, 1)};
    std::vector<double> position_jitter_delta_vector = {position_jitter_delta(0), position_jitter_delta(1)};
    distribution::MultivariateGaussian mvg_position_jitter(position_jitter_mean_vector, position_jitter_covariance_vector);

    position_t.x(position_t.x() + position_jitter_delta(0));
    position_t.y(position_t.y() + position_jitter_delta(1));
    my_state_t->position(position_t);
    my_state_t->state_prediction_log_probability(my_state_t->state_prediction_log_probability() + std::log(mvg_position_jitter.QuantizedProbability(position_jitter_delta_vector)));
  }
  if (this->yaw_jitter_flag_) {
    double original_rotation_state_prediction_log_probability = my_state_t->motion_model_1d_rotation_state_ptr()->state_prediction_log_probability();
    this->motion_model_1d_rotation_.JitterState(my_state_t->motion_model_1d_rotation_state_ptr());
    double additional_rotation_state_prediction_log_probability = my_state_t->motion_model_1d_rotation_state_ptr()->state_prediction_log_probability() - original_rotation_state_prediction_log_probability;
    my_state_t->state_prediction_log_probability(my_state_t->state_prediction_log_probability() + additional_rotation_state_prediction_log_probability);
  }
}

}  // namespace prediction_model

}  // namespace state_estimation
