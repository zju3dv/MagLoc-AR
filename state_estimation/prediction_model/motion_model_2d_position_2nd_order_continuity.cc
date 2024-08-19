/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:10
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-02-07 21:59:56
 */
#include "prediction_model/motion_model_2d_position_2nd_order_continuity.h"

#include <cmath>
#include <iostream>
#include <random>
#include <mutex>

#include "distribution/gaussian_distribution.h"
#include "prediction_model/base.h"
#include "variable/position.h"

namespace state_estimation {

namespace prediction_model {

void MotionModel2dPosition2ndOrderContinuityState::EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
  this->ValidateStatesAndWeights(sample_state_ptrs, weights);

  variable::Position mean_position;
  Eigen::Vector2d mean_velocity;
  MotionModel2dPosition2ndOrderContinuityState* my_state_ptr;
  for (int i = 0; i < sample_state_ptrs.size(); i++) {
    my_state_ptr = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(sample_state_ptrs.at(i));
    if (i == 0) {
      mean_position = my_state_ptr->position() * weights.at(i);
      mean_velocity = my_state_ptr->velocity() * weights.at(i);
    } else {
      mean_position = mean_position + my_state_ptr->position() * weights.at(i);
      mean_velocity = mean_velocity + my_state_ptr->velocity() * weights.at(i);
    }
  }

  this->position_ = mean_position;
  this->velocity_ = mean_velocity;
}

void MotionModel2dPosition2ndOrderContinuityStateUncertainty::CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
  State::ValidateStatesAndWeights(sample_state_ptrs, weights);
  Eigen::Matrix<double, Eigen::Dynamic, 2> position_2d_vectors = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sample_state_ptrs.size(), 2);
  Eigen::Matrix<double, Eigen::Dynamic, 1> velocity_2d_vectors = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(sample_state_ptrs.size(), 2);
  MotionModel2dPosition2ndOrderContinuityState* my_state_ptr;

  for (int i = 0; i < sample_state_ptrs.size(); i++) {
    my_state_ptr = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(sample_state_ptrs.at(i));
    position_2d_vectors(i, 0) = my_state_ptr->position().x();
    position_2d_vectors(i, 1) = my_state_ptr->position().y();
    velocity_2d_vectors(i, 0) = my_state_ptr->velocity()(0);
    velocity_2d_vectors(i, 1) = my_state_ptr->velocity()(1);
  }

  Eigen::Matrix<double, Eigen::Dynamic, 1> weights_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(weights.size(), 1);
  for (int i = 0; i < weights.size(); i++) {
    weights_vector(i) = weights.at(i);
  }

  this->position_2d_covariance_ = CalculateCovariance(position_2d_vectors, weights_vector);
  this->position_2d_distance_variance_ = position_2d_vectors.rowwise().squaredNorm().sum();
  this->velocity_2d_covariance_ = CalculateCovariance(velocity_2d_vectors, weights_vector);
}

void MotionModel2dPosition2ndOrderContinuity::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  MotionModel2dPosition2ndOrderContinuityState* my_state_t = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_t);
  MotionModel2dPosition2ndOrderContinuityState* my_state_tminus = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_tminus);
  MotionModel2dPosition2ndOrderContinuityControlInput* my_control_input_t = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityControlInput*>(control_input_t);

  Eigen::Vector2d accelerations;
  {
  std::lock_guard<std::mutex> guard(this->my_mutex_);
  this->mvg_sampler_.SetParams(my_control_input_t->acceleration_means(), my_control_input_t->acceleration_covariance());
  accelerations = this->mvg_sampler_.Sample();
  }

  variable::Position position_tminus = my_state_tminus->position();
  Eigen::Vector2d v_tminus = my_state_tminus->velocity();

  my_state_t->velocity(v_tminus + accelerations * dt);
  variable::Position position_t;
  position_t.x(position_tminus.x() + v_tminus(0) * dt + 0.5 * accelerations(0) * dt * dt);
  position_t.y(position_tminus.y() + v_tminus(1) * dt + 0.5 * accelerations(1) * dt * dt);
  position_t.floor(position_tminus.floor());
  my_state_t->position(position_t);
}

void MotionModel2dPosition2ndOrderContinuity::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  MotionModel2dPosition2ndOrderContinuityState* my_state_t = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_t);
  MotionModel2dPosition2ndOrderContinuityState* my_state_tminus = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_tminus);

  variable::Position position_tminus = my_state_tminus->position();
  Eigen::Vector2d v_tminus = my_state_tminus->velocity();

  my_state_t->velocity(v_tminus);
  variable::Position position_t;
  position_t.x(position_tminus.x() + v_tminus(0) * dt);
  position_t.y(position_tminus.y() + v_tminus(1) * dt);
  position_t.floor(position_tminus.floor());
  my_state_t->position(position_t);
}

double MotionModel2dPosition2ndOrderContinuity::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  MotionModel2dPosition2ndOrderContinuityState* my_state_t = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_t);
  MotionModel2dPosition2ndOrderContinuityState* my_state_tminus = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_tminus);
  MotionModel2dPosition2ndOrderContinuityControlInput* my_control_input_t = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityControlInput*>(control_input_t);

  Eigen::Vector2d accelerations = (my_state_t->velocity() - my_state_tminus->velocity()) / dt;
  Eigen::Vector2d acceleration_means = my_control_input_t->acceleration_means();
  Eigen::Matrix2d acceleration_covariance = my_control_input_t->acceleration_covariance();

  std::vector<double> x = {accelerations(0), accelerations(1)};
  std::vector<double> means = {acceleration_means(0), acceleration_means(1)};
  std::vector<double> covariance = {acceleration_covariance(0, 0), acceleration_covariance(0, 1), acceleration_covariance(1, 1)};
  distribution::MultivariateGaussian mvg(means, covariance);
  return std::log(mvg.QuantizedProbability(x));
}

double MotionModel2dPosition2ndOrderContinuityStateSampler::CalculateStateProbabilityLog(State* state_sample) {
  MotionModel2dPosition2ndOrderContinuityState* my_state_sample =  reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_sample);
  Eigen::Vector2d velocity = my_state_sample->velocity();
  variable::Position position = my_state_sample->position();
  position.Round(this->position_griding_resolution_);
  double v_min = this->v_single_dimension_sampler_.min_value();
  double v_max = this->v_single_dimension_sampler_.max_value();
  std::set<std::string> position_key_set = this->position_key_sampler_.population();

  double log_prob = 0.0;
  if (position_key_set.find(position.ToKey()) != position_key_set.end()) {
    log_prob += std::log(1.0 / position_key_set.size());
  } else {
    log_prob += std::log(0.0);
  }
  if ((velocity(0) >= v_min) && (velocity(0) <= v_max)) {
    log_prob += std::log(1.0 / (v_max - v_min));
  } else {
    log_prob += std::log(0.0);
  }
  if ((velocity(1) >= v_min) && (velocity(1) <= v_max)) {
    log_prob += std::log(1.0 / (v_max - v_min));
  } else {
    log_prob += std::log(0.0);
  }

  return log_prob;
}

}  // namespace prediction_model

}  // namespace state_estimation
