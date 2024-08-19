/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-04-26 17:40:38
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-16 19:24:48
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_DISPLACEMENT_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_DISPLACEMENT_H_

#include <Eigen/Dense>

#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_range_sampler.h"
#include "sampler/uniform_set_sampler.h"
#include "variable/orientation.h"
#include "variable/position.h"
#include "util/misc.h"
#include "util/variable_name_constants.h"

namespace state_estimation {

namespace prediction_model {

class MotionModel2dDisplacementState : public State {
 public:
  static const int kNumberOfStateVariables = 1;

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNamePositionX, this->position_.x());
    named_values->emplace(named_values->end(), util::kNamePositionY, this->position_.y());
  }

  void EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    assert(sample_state_ptrs.size() == weights.size());
    double weights_sum = 0.0;
    for (int i = 0; i < weights.size(); i++) {
      weights_sum += weights.at(i);
    }
    assert(weights_sum > 0.0);
    for (int i = 0; i < weights.size(); i++) {
      weights.at(i) /= weights_sum;
    }

    variable::Position mean_position;
    MotionModel2dDisplacementState* my_state_ptr;
    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModel2dDisplacementState*>(sample_state_ptrs.at(i));
      if (i == 0) {
        mean_position = my_state_ptr->position() * weights.at(i);
      } else {
        mean_position = mean_position + my_state_ptr->position() * weights.at(i);
      }
    }

    this->position_ = mean_position;
  }

  variable::Position position(void) const {
    return this->position_;
  }

  void position(variable::Position position) {
    this->position_ = position;
  }

  std::string ToKey(std::vector<double> discretization_resolution) {
    assert(discretization_resolution.size() == this->kNumberOfStateVariables);
    double position_resolution = discretization_resolution.at(0);
    variable::Position temp_position = this->position_;
    temp_position.Round(position_resolution);
    return temp_position.ToKey();
  }

  void Add(State* state_ptr) {
    MotionModel2dDisplacementState* my_state_ptr = reinterpret_cast<MotionModel2dDisplacementState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() + temp_position.x());
    position.y(this->position_.y() + temp_position.y());
    position.floor(this->position_.floor());
    this->position_ = position;
  }

  void Subtract(State* state_ptr) {
    MotionModel2dDisplacementState* my_state_ptr = reinterpret_cast<MotionModel2dDisplacementState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() - temp_position.x());
    position.y(this->position_.y() - temp_position.y());
    position.floor(this->position_.floor());
    this->position_ = position;
  }

  void Multiply_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() * scalar_value);
    position.y(this->position_.y() * scalar_value);
    position.floor(this->position_.floor());
    this->position_ = position;
  }

  void Divide_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() / scalar_value);
    position.y(this->position_.y() / scalar_value);
    position.floor(this->position_.floor());
    this->position_ = position;
  }

  MotionModel2dDisplacementState(void) {
    this->position_ = variable::Position();
  }

  ~MotionModel2dDisplacementState() {}

 private:
  variable::Position position_;
};

class MotionModel2dDisplacementStateUncertianty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    assert(sample_state_ptrs.size() == weights.size());
    Eigen::Matrix<double, Eigen::Dynamic, 2> position_2d_vectors = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sample_state_ptrs.size(), 2);
    MotionModel2dDisplacementState* my_state_ptr;

    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModel2dDisplacementState*>(sample_state_ptrs.at(i));
      position_2d_vectors(i, 0) = my_state_ptr->position().x();
      position_2d_vectors(i, 1) = my_state_ptr->position().y();
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> weights_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(weights.size(), 1);
    for (int i = 0; i < weights.size(); i++) {
      weights_vector(i) = weights.at(i);
    }
    weights_vector = weights_vector.array() / weights_vector.sum();

    position_2d_vectors.col(0) = (position_2d_vectors.col(0).array() - position_2d_vectors.col(0).mean()) * weights_vector.array().sqrt();
    position_2d_vectors.col(1) = (position_2d_vectors.col(1).array() - position_2d_vectors.col(1).mean()) * weights_vector.array().sqrt();

    this->position_2d_covariance_ = position_2d_vectors.transpose() * position_2d_vectors;
    this->position_2d_distance_variance_ = position_2d_vectors.rowwise().squaredNorm().sum();
  }

  Eigen::Matrix<double, 2, 2> position_2d_covariance(void) {
    return this->position_2d_covariance_;
  }

  double position_2d_distance_variance(void) {
    return this->position_2d_distance_variance_;
  }

  MotionModel2dDisplacementStateUncertianty(void) {
    this->position_2d_covariance_ = Eigen::Matrix<double, 2, 2>::Zero(2, 2);
    this->position_2d_distance_variance_ = 0.0;
  }

  ~MotionModel2dDisplacementStateUncertianty() {}

 private:
  Eigen::Matrix<double, 2, 2> position_2d_covariance_;
  double position_2d_distance_variance_;
};

class MotionModel2dDisplacementControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNameDpositionX, this->dp_means_(0, 0));
    named_values->emplace(named_values->end(), util::kNameDpositionY, this->dp_means_(1, 0));
  }

  double dx(void) {
    return this->dp_means_(0, 0);
  }

  void dx(double dx) {
    this->dp_means_(0, 0) = dx;
  }

  double dy(void) {
    return this->dp_means_(1, 0);
  }

  void dy(double dy) {
    this->dp_means_(1, 0) = dy;
  }

  Eigen::Matrix<double, 2, 1> dp_means(void) {
    return this->dp_means_;
  }

  void dp_means(Eigen::Matrix<double, 2, 1> dp_means) {
    this->dp_means_ = dp_means;
  }

  Eigen::Matrix<double, 2, 2> dp_covariances(void) {
    return this->dp_covariances_;
  }

  void dp_covariances(Eigen::Matrix<double, 2, 2> dp_covariances) {
    this->dp_covariances_ = dp_covariances;
  }

  MotionModel2dDisplacementControlInput(void) {
    this->dp_means_ = Eigen::Matrix<double, 2, 1>::Zero(2, 1);
    this->dp_covariances_ = Eigen::Matrix<double, 2, 2>::Zero(2, 2);
  }

  ~MotionModel2dDisplacementControlInput() {}

 private:
  Eigen::Matrix<double, 2, 1> dp_means_;
  Eigen::Matrix<double, 2, 2> dp_covariances_;
};

class MotionModel2dDisplacement : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);

  void Seed(int random_seed) {
    this->mvg_sampler_.Seed(random_seed);
  }

  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);

  MotionModel2dDisplacement(void) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
  }

  ~MotionModel2dDisplacement() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
};

class MotionModel2dDisplacementStateSampler : public StateSampler {
 public:
  void Init(std::set<std::string> sample_space_position_keys, double position_griding_resolution) {
    this->position_key_sampler_.Init(sample_space_position_keys);
    this->position_griding_resolution_ = position_griding_resolution;
  }

  void Sample(State* state_sample) {
    MotionModel2dDisplacementState* my_state_sample = reinterpret_cast<MotionModel2dDisplacementState*>(state_sample);
    variable::Position temp_position;
    temp_position.FromKey(this->position_key_sampler_.Sample());
    my_state_sample->position(temp_position);
  }

  void Seed(int random_seed) {
    this->position_key_sampler_.Seed(random_seed);
  }

  double CalculateStateProbabilityLog(State* state_sample) {
    MotionModel2dDisplacementState* my_state_sample = reinterpret_cast<MotionModel2dDisplacementState*>(state_sample);
    variable::Position position = my_state_sample->position();
    position.Round(this->position_griding_resolution_);
    std::set<std::string> position_key_set = this->position_key_sampler_.population();

    double log_prob = 0.0;
    if (position_key_set.find(position.ToKey()) != position_key_set.end()) {
      log_prob += std::log(1.0 / position_key_set.size());
    } else {
      log_prob += std::log(0.0);
    }

    return log_prob;
  }

  double position_griding_resolution(void) {
    return this->position_griding_resolution_;
  }

  void position_griding_resolution(double position_griding_resolution) {
    this->position_griding_resolution_ = position_griding_resolution;
  }

  MotionModel2dDisplacementStateSampler(void) {}
  ~MotionModel2dDisplacementStateSampler() {}

 private:
  sampler::UniformSetSampler<std::string> position_key_sampler_;
  double position_griding_resolution_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_DISPLACEMENT_H_
