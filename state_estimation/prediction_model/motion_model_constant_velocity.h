/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:18
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-08 19:44:17
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_CONSTANT_VELOCITY_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_CONSTANT_VELOCITY_H_

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "sampler/uniform_range_sampler.h"
#include "sampler/uniform_set_sampler.h"
#include "variable/position.h"
#include "variable/orientation.h"
#include "util/misc.h"
#include "util/variable_name_constants.h"

namespace state_estimation {

namespace prediction_model {

static const int kNumberOfCovarianceItems = 4;

class MotionModelConstantVelocityState : public State {
 public:
  static const int kNumberOfStateVariables = 3;

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNamePositionX, this->position_.x());
    named_values->emplace(named_values->end(), util::kNamePositionY, this->position_.y());
    named_values->emplace(named_values->end(), util::kNameHeadingYaw, this->yaw_);
    named_values->emplace(named_values->end(), util::kNameHeadingV, this->v_);
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

    std::vector<Eigen::Quaterniond> qs;
    variable::Position mean_position;
    double mean_v = 0.0;
    MotionModelConstantVelocityState* my_state_ptr;
    Eigen::Matrix<double, 3, 1> z_vector = {0.0, 0.0, 1.0};
    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModelConstantVelocityState*>(sample_state_ptrs.at(i));
      if (i == 0) {
        mean_position = my_state_ptr->position() * weights.at(i);
      } else {
        mean_position = mean_position + my_state_ptr->position() * weights.at(i);
      }
      mean_v = mean_v + my_state_ptr->v() * weights.at(i);
      qs.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(my_state_ptr->yaw(), z_vector)));
    }

    Eigen::AngleAxisd mean_yaw_angle_axis(variable::Orientation::Mean(qs, weights));
    assert(mean_yaw_angle_axis.axis()(0) < 1e-5);
    assert(mean_yaw_angle_axis.axis()(1) < 1e-5);
    double mean_yaw;
    if (mean_yaw_angle_axis.axis()(2) >= 0.0) {
      mean_yaw = mean_yaw_angle_axis.angle();
    } else {
      mean_yaw = -mean_yaw_angle_axis.angle();
    }

    this->position_ = mean_position;
    this->yaw_ = mean_yaw;
    this->v_ = mean_v;
  }

  variable::Position position(void) const {
    return this->position_;
  }

  void position(variable::Position position) {
    this->position_ = position;
  }

  double yaw(void) const {
    return this->yaw_;
  }

  void yaw(double yaw) {
    this->yaw_ = yaw;
  }

  double v(void) const {
    return this->v_;
  }

  void v(double v) {
    this->v_ = v;
  }

  std::string ToKey(std::vector<double> discretization_resolutions) {
    assert(discretization_resolutions.size() == this->kNumberOfStateVariables);
    double position_resolution = discretization_resolutions.at(0);
    double yaw_resolution = discretization_resolutions.at(1);
    double v_resolution = discretization_resolutions.at(2);
    variable::Position temp_position = this->position_;
    temp_position.Round(position_resolution);
    double temp_yaw = UniqueZeroRound(this->yaw_, yaw_resolution);
    double temp_v = UniqueZeroRound(this->v_, v_resolution);
    return temp_position.ToKey() + "_" + std::to_string(temp_yaw) + "_" + std::to_string(temp_v);
  }

  void Add(State* state_ptr) {
    MotionModelConstantVelocityState* my_state_ptr = reinterpret_cast<MotionModelConstantVelocityState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() + temp_position.x());
    position.y(this->position_.y() + temp_position.y());
    position.floor(this->position_.floor());
    double v = this->v_ + my_state_ptr->v();
    double yaw = this->yaw_ + my_state_ptr->yaw();
    this->position_ = position;
    this->v_ = v;
    this->yaw_ = yaw;
  }

  void Subtract(State* state_ptr) {
    MotionModelConstantVelocityState* my_state_ptr = reinterpret_cast<MotionModelConstantVelocityState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() - temp_position.x());
    position.y(this->position_.y() - temp_position.y());
    position.floor(this->position_.floor());
    double v = this->v_ - my_state_ptr->v();
    double yaw = this->yaw_ - my_state_ptr->yaw();
    this->position_ = position;
    this->v_ = v;
    this->yaw_ = yaw;
  }

  void Multiply_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() * scalar_value);
    position.y(this->position_.y() * scalar_value);
    position.floor(this->position_.floor());
    double v = this->v_ * scalar_value;
    double yaw = this->yaw_ * scalar_value;
    this->position_ = position;
    this->v_ = v;
    this->yaw_ = yaw;
  }

  void Divide_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() / scalar_value);
    position.y(this->position_.y() / scalar_value);
    position.floor(this->position_.floor());
    double v = this->v_ / scalar_value;
    double yaw = this->yaw_ / scalar_value;
    this->position_ = position;
    this->v_ = v;
    this->yaw_ = yaw;
  }

  MotionModelConstantVelocityState(void);
  ~MotionModelConstantVelocityState();

 private:
  variable::Position position_;
  double yaw_;
  double v_;
};

class MotionModelConstantVelocityStateUncertainty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    assert(weights.size() == sample_state_ptrs.size());
    Eigen::Matrix<double, Eigen::Dynamic, 2> position_2d_vectors = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sample_state_ptrs.size(), 2);
    MotionModelConstantVelocityState* my_state_ptr;
    std::vector<Eigen::Quaterniond> q_yaws;
    Eigen::Matrix<double, Eigen::Dynamic, 1> vs = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(sample_state_ptrs.size(), 1);
    Eigen::Matrix<double, 3, 1> z_vector = {0.0, 0.0, 1.0};

    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModelConstantVelocityState*>(sample_state_ptrs.at(i));
      position_2d_vectors(i, 0) = my_state_ptr->position().x();
      position_2d_vectors(i, 1) = my_state_ptr->position().y();
      q_yaws.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(my_state_ptr->yaw(), z_vector)));
      vs(i) = my_state_ptr->v();
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
    this->yaw_variance_ = variable::Orientation::Variance(q_yaws, weights);
    this->v_variance_ = (vs.array() - vs.mean()).pow(2.0).matrix().transpose() * weights_vector;
  }

  Eigen::Matrix<double, 2, 2> position_2d_covariance(void) {
    return this->position_2d_covariance_;
  }

  double position_2d_distance_variance(void) {
    return this->position_2d_distance_variance_;
  }

  double yaw_variance(void) {
    return this->yaw_variance_;
  }

  double v_variance(void) {
    return this->v_variance_;
  }

  MotionModelConstantVelocityStateUncertainty(void) {
    this->position_2d_covariance_ = Eigen::Matrix<double, 2, 2>::Zero(2, 2);
    this->position_2d_distance_variance_ = 0.0;
    this->yaw_variance_ = 0.0;
    this->v_variance_ = 0.0;
  }

  ~MotionModelConstantVelocityStateUncertainty() {}

 private:
  Eigen::Matrix<double, 2, 2> position_2d_covariance_;
  double position_2d_distance_variance_;
  double yaw_variance_;
  double v_variance_;
};

class MotionModelConstantVelocityControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNameHeadingYaw, this->yaw_);
  }

  double yaw(void) {
    return this->yaw_;
  }

  void yaw(double yaw) {
    this->yaw_ = yaw;
  }

  MotionModelConstantVelocityControlInput(void);
  ~MotionModelConstantVelocityControlInput();

 private:
  double yaw_;
};

class MotionModelConstantVelocity : public PredictionModel {
 public:
  void Init(double covariance[kNumberOfCovarianceItems]);
  void Predict(State* state_t,
               State* state_tminus,
               ControlInput* control_input_t,
               double dt);
  void PredictWithoutControlInput(State* state_t,
                                  State* state_tminus,
                                  double dt);

  void Seed(int random_seed) {
    this->generator_.seed(static_cast<uint64_t>(random_seed));
  }

  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus,
                                                ControlInput* control_input_t,
                                                double dt);

  MotionModelConstantVelocity(void) {}

  ~MotionModelConstantVelocity() {}

 private:
  double covariance_[kNumberOfCovarianceItems];
  std::default_random_engine generator_;
};

class MotionModelConstantVelocityStateSampler : public StateSampler {
 public:
  void Init(std::set<std::string> sample_space_position_keys,
            double v_min, double v_max,
            double yaw_min, double yaw_max,
            double position_griding_resolution) {
    this->position_key_sampler_.Init(sample_space_position_keys);
    this->v_sampler_.Init(v_min, v_max);
    this->yaw_sampler_.Init(yaw_min, yaw_max);
    this->position_griding_resolution_ = position_griding_resolution;
  }

  void Sample(State* state_sample) {
#ifdef DEBUG_FOCUSING
    std::cout << "MotionModelConstantVelocityStateSampler::Sample" << std::endl;
#endif
    MotionModelConstantVelocityState* my_state_sample =
        reinterpret_cast<MotionModelConstantVelocityState*>(state_sample);
    variable::Position temp_position;
    temp_position.FromKey(this->position_key_sampler_.Sample());
    my_state_sample->position(temp_position);
    my_state_sample->v(this->v_sampler_.Sample());
    my_state_sample->yaw(this->yaw_sampler_.Sample());
  }

  void Seed(int random_seed) {
    this->position_key_sampler_.Seed(random_seed);
    this->v_sampler_.Seed(random_seed);
    this->yaw_sampler_.Seed(random_seed);
  }

  double CalculateStateProbabilityLog(State* state_sample) {
    MotionModelConstantVelocityState* my_state_sample =  reinterpret_cast<MotionModelConstantVelocityState*>(state_sample);
    double v = my_state_sample->v();
    double yaw = my_state_sample->yaw();
    variable::Position position = my_state_sample->position();
    position.Round(this->position_griding_resolution_);
    double v_min = this->v_sampler_.min_value();
    double v_max = this->v_sampler_.max_value();
    double yaw_min = this->yaw_sampler_.min_value();
    double yaw_max = this->yaw_sampler_.max_value();
    std::set<std::string> position_key_set = this->position_key_sampler_.population();

    double log_prob = 0.0;
    if (position_key_set.find(position.ToKey()) != position_key_set.end()) {
      log_prob += std::log(1.0 / position_key_set.size());
    } else {
      log_prob += std::log(0.0);
    }
    if ((v >= v_min) && (v <= v_max)) {
      log_prob += std::log(1.0 / (v_max - v_min));
    } else {
      log_prob += std::log(0.0);
    }
    if ((yaw >= yaw_min) && (yaw <= yaw_max)) {
      log_prob += std::log(1.0 / (yaw_max - yaw_min));
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

  MotionModelConstantVelocityStateSampler(void) {}
  ~MotionModelConstantVelocityStateSampler() {}

 private:
  sampler::UniformSetSampler<std::string> position_key_sampler_;
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double>
      v_sampler_;
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double>
      yaw_sampler_;
  double position_griding_resolution_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_CONSTANT_VELOCITY_H_
