/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:18
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-02-07 15:58:32
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_POSITION_2ND_ORDER_CONTINUITY_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_POSITION_2ND_ORDER_CONTINUITY_H_

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_range_sampler.h"
#include "sampler/uniform_set_sampler.h"
#include "variable/position.h"
#include "variable/orientation.h"
#include "util/misc.h"
#include "util/variable_name_constants.h"

namespace state_estimation {

namespace prediction_model {

class MotionModel2dPosition2ndOrderContinuityState : public State {
 public:
  static const int kNumberOfStateVariables = 2; // position and velocity

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNamePositionX, this->position_.x());
    named_values->emplace(named_values->end(), util::kNamePositionY, this->position_.y());
    named_values->emplace(named_values->end(), util::kNameVx, this->velocity_(0));
    named_values->emplace(named_values->end(), util::kNameVy, this->velocity_(1));
  }

  void EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights);

  variable::Position position(void) const {
    return this->position_;
  }

  void position(variable::Position position) {
    this->position_ = position;
  }

  Eigen::Vector2d velocity(void) const {
    return this->velocity_;
  }

  void velocity(Eigen::Vector2d velocity) {
    this->velocity_ = velocity;
  }

  std::string ToKey(std::vector<double> discretization_resolutions) {
    assert(discretization_resolutions.size() == this->kNumberOfStateVariables);
    double position_resolution = discretization_resolutions.at(0);
    double velocity_resolution = discretization_resolutions.at(1);
    variable::Position temp_position = this->position_;
    temp_position.Round(position_resolution);
    Eigen::Vector2d temp_velocity;
    temp_velocity(0) = UniqueZeroRound(this->velocity_(0), velocity_resolution);
    temp_velocity(1) = UniqueZeroRound(this->velocity_(1), velocity_resolution);
    return temp_position.ToKey() + "_" + std::to_string(temp_velocity(0)) + "_" + std::to_string(temp_velocity(1));
  }

  void Add(State* state_ptr) {
    MotionModel2dPosition2ndOrderContinuityState* my_state_ptr = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() + temp_position.x());
    position.y(this->position_.y() + temp_position.y());
    position.floor(this->position_.floor());
    this->position_ = position;
    this->velocity_(0) = this->velocity_(0) + my_state_ptr->velocity()(0);
    this->velocity_(1) = this->velocity_(1) + my_state_ptr->velocity()(1);
  }

  void Subtract(State* state_ptr) {
    MotionModel2dPosition2ndOrderContinuityState* my_state_ptr = reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() - temp_position.x());
    position.y(this->position_.y() - temp_position.y());
    position.floor(this->position_.floor());
    this->position_ = position;
    this->velocity_(0) = this->velocity_(0) - my_state_ptr->velocity()(0);
    this->velocity_(1) = this->velocity_(1) - my_state_ptr->velocity()(1);
  }

  void Multiply_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() * scalar_value);
    position.y(this->position_.y() * scalar_value);
    position.floor(this->position_.floor());
    this->position_ = position;
    this->velocity_ = this->velocity_ * scalar_value;
  }

  void Divide_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() / scalar_value);
    position.y(this->position_.y() / scalar_value);
    position.floor(this->position_.floor());
    this->position_ = position;
    this->velocity_ = this->velocity_ / scalar_value;
  }

  MotionModel2dPosition2ndOrderContinuityState(void) {
    this->position_ = variable::Position();
    this->velocity_ = Eigen::Vector2d::Zero();
  }

  ~MotionModel2dPosition2ndOrderContinuityState() {}

 private:
  variable::Position position_;
  Eigen::Vector2d velocity_;
};

class MotionModel2dPosition2ndOrderContinuityStateUncertainty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights);

  Eigen::Matrix2d position_2d_covariance(void) {
    return this->position_2d_covariance_;
  }

  double position_2d_distance_variance(void) {
    return this->position_2d_distance_variance_;
  }

  Eigen::Matrix2d velocity_2d_covariance(void) {
    return this->velocity_2d_covariance_;
  }

  MotionModel2dPosition2ndOrderContinuityStateUncertainty(void) {
    this->position_2d_covariance_ = Eigen::Matrix2d::Zero();
    this->position_2d_distance_variance_ = 0.0;
    this->velocity_2d_covariance_ = Eigen::Matrix2d::Zero();
  }

  ~MotionModel2dPosition2ndOrderContinuityStateUncertainty() {}

 private:
  Eigen::Matrix2d position_2d_covariance_;
  double position_2d_distance_variance_;
  Eigen::Matrix2d velocity_2d_covariance_;
};

class MotionModel2dPosition2ndOrderContinuityControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNameAccX, this->acceleration_means_(0));
    named_values->emplace(named_values->end(), util::kNameAccY, this->acceleration_means_(1));
    named_values->emplace(named_values->end(), "cov_00", this->acceleration_covariance_(0, 0));
    named_values->emplace(named_values->end(), "cov_01", this->acceleration_covariance_(0, 1));
    named_values->emplace(named_values->end(), "cov_10", this->acceleration_covariance_(1, 0));
    named_values->emplace(named_values->end(), "cov_11", this->acceleration_covariance_(1, 1));
  }

  Eigen::Vector2d acceleration_means(void) {
    return this->acceleration_means_;
  }

  void acceleration_means(Eigen::Vector2d acceleration_means) {
    this->acceleration_means_ = acceleration_means;
  }

  Eigen::Matrix2d acceleration_covariance(void) {
    return this->acceleration_covariance_;
  }

  void acceleration_covariance(Eigen::Matrix2d acceleration_covariance) {
    this->acceleration_covariance_ = acceleration_covariance;
  }

  MotionModel2dPosition2ndOrderContinuityControlInput(void) {
    this->acceleration_means_ = Eigen::Vector2d::Zero();
    this->acceleration_covariance_ = Eigen::Matrix2d::Zero();
  }

  ~MotionModel2dPosition2ndOrderContinuityControlInput() {}

 private:
  Eigen::Vector2d acceleration_means_;
  Eigen::Matrix2d acceleration_covariance_;
};

class MotionModel2dPosition2ndOrderContinuity : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);
  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);

  void Seed(int random_seed) {
    this->mvg_sampler_.Seed(random_seed);
  }

  MotionModel2dPosition2ndOrderContinuity(void) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
  }

  ~MotionModel2dPosition2ndOrderContinuity() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
};

class MotionModel2dPosition2ndOrderContinuityStateSampler : public StateSampler {
 public:
  void Init(std::set<std::string> sample_space_position_keys,
            double v_single_dimension_min, double v_single_dimension_max,
            double position_griding_resolution) {
    this->position_key_sampler_.Init(sample_space_position_keys);
    this->v_single_dimension_sampler_.Init(v_single_dimension_min, v_single_dimension_max);
    this->position_griding_resolution_ = position_griding_resolution;
  }

  void Sample(State* state_sample) {
    MotionModel2dPosition2ndOrderContinuityState* my_state_sample =
        reinterpret_cast<MotionModel2dPosition2ndOrderContinuityState*>(state_sample);
    variable::Position temp_position;
    temp_position.FromKey(this->position_key_sampler_.Sample());
    my_state_sample->position(temp_position);
    Eigen::Vector2d temp_velocity;
    temp_velocity(0) = this->v_single_dimension_sampler_.Sample();
    temp_velocity(1) = this->v_single_dimension_sampler_.Sample();
    my_state_sample->velocity(temp_velocity);
  }

  void Seed(int random_seed) {
    this->position_key_sampler_.Seed(random_seed);
    this->v_single_dimension_sampler_.Seed(random_seed);
  }

  double CalculateStateProbabilityLog(State* state_sample);

  double position_griding_resolution(void) {
    return this->position_griding_resolution_;
  }

  void position_griding_resolution(double position_griding_resolution) {
    this->position_griding_resolution_ = position_griding_resolution;
  }

  MotionModel2dPosition2ndOrderContinuityStateSampler(void) {
    this->position_key_sampler_ = sampler::UniformSetSampler<std::string>();
    this->v_single_dimension_sampler_ = sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double>();
    this->position_griding_resolution_ = 1.0;
  }

  ~MotionModel2dPosition2ndOrderContinuityStateSampler() {}

 private:
  sampler::UniformSetSampler<std::string> position_key_sampler_;
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double> v_single_dimension_sampler_;
  double position_griding_resolution_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif // STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_POSITION_2ND_ORDER_CONTINUITY_H_
