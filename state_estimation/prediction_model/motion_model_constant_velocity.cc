/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:10
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-04-27 15:26:21
 */
#include "prediction_model/motion_model_constant_velocity.h"

#include <cmath>
#include <iostream>
#include <random>
#include <mutex>

#include "distribution/gaussian_distribution.h"
#include "prediction_model/base.h"
#include "variable/position.h"

namespace state_estimation {

namespace prediction_model {

MotionModelConstantVelocityState::MotionModelConstantVelocityState(void) {
  variable::Position position;
  this->position_ = position;
  this->yaw_ = 0.0;
  this->v_ = 0.0;
}

MotionModelConstantVelocityState::~MotionModelConstantVelocityState() {
#ifdef DEBUG_FOCUSING
  std::cout << "MotionModelConstantVelocityState::~MotionModelConstantVelocityState" << std::endl;
#endif
}

MotionModelConstantVelocityControlInput::
    MotionModelConstantVelocityControlInput(void) {
  this->yaw_ = 0.0;
}

MotionModelConstantVelocityControlInput::
    ~MotionModelConstantVelocityControlInput() {}

void MotionModelConstantVelocity::
    Init(double covariance[kNumberOfCovarianceItems]) {
  for (int i = 0;
       i < kNumberOfCovarianceItems; i++) {
    this->covariance_[i] = covariance[i];
  }
}

void MotionModelConstantVelocity::Predict(
    State* state_t,
    State* state_tminus,
    ControlInput* control_input_t,
    double dt) {
  MotionModelConstantVelocityState* my_state_t =
      reinterpret_cast<MotionModelConstantVelocityState*>(state_t);
  MotionModelConstantVelocityState* my_state_tminus =
      reinterpret_cast<MotionModelConstantVelocityState*>(state_tminus);
  MotionModelConstantVelocityControlInput* my_control_input_t =
      reinterpret_cast<MotionModelConstantVelocityControlInput*>(control_input_t);
  std::normal_distribution<double> acc_noise(0.0, this->covariance_[0]);
  std::normal_distribution<double> yaw_noise(0.0, this->covariance_[3]);
  double acc, yaw, line_translation = 0.0;
  {
  std::lock_guard<std::mutex> guard(this->my_mutex_);
  acc = acc_noise(this->generator_);
  yaw = my_state_tminus->yaw() + yaw_noise(this->generator_);
  }
  line_translation = my_state_tminus->v() * dt + 0.5 * acc * dt * dt;
  variable::Position position;
  position.x(
      (my_state_tminus->position()).x() + line_translation * std::cos(yaw));
  position.y(
      (my_state_tminus->position()).y() + line_translation * std::sin(yaw));
  position.floor((my_state_tminus->position()).floor());
  my_state_t->position(position);
  my_state_t->yaw(my_control_input_t->yaw());
  my_state_t->v(my_state_tminus->v() + acc * dt);
}

void MotionModelConstantVelocity::PredictWithoutControlInput(
    State* state_t,
    State* state_tminus,
    double dt) {
  MotionModelConstantVelocityState* my_state_t = reinterpret_cast<MotionModelConstantVelocityState*>(state_t);
  MotionModelConstantVelocityState* my_state_tminus = reinterpret_cast<MotionModelConstantVelocityState*>(state_tminus);
  my_state_t->v(my_state_tminus->v());
  my_state_t->yaw(my_state_tminus->yaw());
  variable::Position position;
  position.x((my_state_tminus->position()).x() + my_state_tminus->v() * dt * std::cos(my_state_tminus->yaw()));
  position.y((my_state_tminus->position()).y() + my_state_tminus->v() * dt * std::sin(my_state_tminus->yaw()));
  position.floor((my_state_tminus->position()).floor());
  my_state_t->position(position);
}

double MotionModelConstantVelocity::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  MotionModelConstantVelocityState* my_state_t = reinterpret_cast<MotionModelConstantVelocityState*>(state_t);
  MotionModelConstantVelocityState* my_state_tminus = reinterpret_cast<MotionModelConstantVelocityState*>(state_tminus);

  double dx, dy, acc, d_yaw;
  acc = (my_state_t->v() - my_state_tminus->v()) / dt;
  dx = my_state_t->position().x() - my_state_tminus->position().x();
  dy = my_state_t->position().y() - my_state_tminus->position().y();
  d_yaw = std::atan2(dy, dx) - my_state_tminus->yaw();
  std::vector<double> x = {acc, d_yaw};

  std::vector<double> means = {0.0, 0.0};
  std::vector<double> covariance = {this->covariance_[0], 0.0, this->covariance_[3]};
  distribution::MultivariateGaussian mvg(means, covariance);
  return std::log(mvg.QuantizedProbability(x));
}

}  // namespace prediction_model

}  // namespace state_estimation
