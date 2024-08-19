/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-08 19:18:28
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-18 16:42:38
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_CVCR_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_CVCR_H_

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_3d_orientation_sampler.h"
#include "sampler/uniform_range_sampler.h"
#include "sampler/uniform_set_sampler.h"
#include "util/misc.h"
#include "util/variable_name_constants.h"
#include "variable/orientation.h"
#include "variable/position.h"

namespace state_estimation {

namespace prediction_model {

// CVCR is short for Constant Velocity and Constant Rotation
// reference: "MonoSLAM: Real-Time Single Camera SLAM"
class MotionModelCVCRState : public State {
 public:
  // static const int kNumberOfStateVariables = 4;
  static const int kNumberOfStateVariables = 1;

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNamePositionX, this->position_.x());
    named_values->emplace(named_values->end(), util::kNamePositionY, this->position_.y());
    named_values->emplace(named_values->end(), util::kNameVx, this->v_(0));
    named_values->emplace(named_values->end(), util::kNameVy, this->v_(1));
    named_values->emplace(named_values->end(), util::kNameOrientationW, this->orientation_.q().w());
    named_values->emplace(named_values->end(), util::kNameOrientationX, this->orientation_.q().x());
    named_values->emplace(named_values->end(), util::kNameOrientationY, this->orientation_.q().y());
    named_values->emplace(named_values->end(), util::kNameOrientationZ, this->orientation_.q().z());
    named_values->emplace(named_values->end(), util::kNameOmegaAngle, this->omega_.angle());
    named_values->emplace(named_values->end(), util::kNameOmegaAxisX, this->omega_.axis()(0));
    named_values->emplace(named_values->end(), util::kNameOmegaAxisY, this->omega_.axis()(1));
    named_values->emplace(named_values->end(), util::kNameOmegaAxisZ, this->omega_.axis()(2));
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
    Eigen::Matrix<double, 2, 1> mean_v = {0.0, 0.0};
    std::vector<variable::Orientation> orientations;
    Eigen::Matrix<double, 3, 1> mean_omega_vector = {0.0, 0.0, 0.0};
    MotionModelCVCRState* my_state_ptr;
    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModelCVCRState*>(sample_state_ptrs.at(i));
      if (i == 0) {
        mean_position = my_state_ptr->position() * weights.at(i);
      } else {
        mean_position = mean_position + my_state_ptr->position() * weights.at(i);
      }
      mean_v = mean_v + my_state_ptr->v() * weights.at(i);
      mean_omega_vector = mean_omega_vector + my_state_ptr->omega().axis() * my_state_ptr->omega().angle() * weights.at(i);
      orientations.push_back(my_state_ptr->orientation());
    }

    this->position_ = mean_position;
    this->v_ = mean_v;
    this->orientation_ = variable::Orientation::Mean(orientations, weights);
    this->omega_ = Eigen::AngleAxisd(mean_omega_vector.norm(), mean_omega_vector / mean_omega_vector.norm());
  }

  variable::Position position(void) const {
    return this->position_;
  }

  void position(variable::Position position) {
    this->position_ = position;
  }

  Eigen::Matrix<double, 2, 1> v(void) {
    return this->v_;
  }

  void v(Eigen::Matrix<double, 2, 1> v) {
    this->v_ = v;
  }

  variable::Orientation orientation(void) {
    return this->orientation_;
  }

  void orientation(variable::Orientation orientation) {
    this->orientation_ = orientation;
  }

  Eigen::AngleAxisd omega(void) {
    return this->omega_;
  }

  void omega(Eigen::AngleAxisd omega) {
    this->omega_ = omega;
  }

  // std::string ToKey(std::vector<double> discretization_resolutions) {
  //   assert(discretization_resolutions.size() == this->kNumberOfStateVariables);
  //   double position_resolution = discretization_resolutions.at(0);
  //   double v_resolution = discretization_resolutions.at(1);
  //   double orientation_resolution = discretization_resolutions.at(2);
  //   double omega_resolution = discretization_resolutions.at(3);
  //   variable::Position temp_position = this->position_;
  //   temp_position.Round(position_resolution);
  //   double temp_v_x = UniqueZeroRound(this->v_(0), v_resolution);
  //   double temp_v_y = UniqueZeroRound(this->v_(1), v_resolution);
  //   variable::Orientation temp_orientation = this->orientation_;
  //   temp_orientation.Round(orientation_resolution);

  //   // TODO(xuehua): I need to correct the way to quantize omega.
  //   double temp_omega_x = UniqueZeroRound((this->omega_.angle() * this->omega_.axis())(0), omega_resolution);
  //   double temp_omega_y = UniqueZeroRound((this->omega_.angle() * this->omega_.axis())(1), omega_resolution);
  //   double temp_omega_z = UniqueZeroRound((this->omega_.angle() * this->omega_.axis())(2), omega_resolution);

  //   return temp_position.ToKey() +
  //          "_" + std::to_string(temp_v_x) +
  //          "_" + std::to_string(temp_v_y) +
  //          "_" + temp_orientation.ToKey() +
  //          "_" + std::to_string(temp_omega_x) +
  //          "_" + std::to_string(temp_omega_y) +
  //          "_" + std::to_string(temp_omega_z);
  // }

  std::string ToKey(std::vector<double> discretization_resolutions) {
    assert(discretization_resolutions.size() == this->kNumberOfStateVariables);
    double position_resolution = discretization_resolutions.at(0);
    variable::Position temp_position = this->position_;
    temp_position.Round(position_resolution);

    return temp_position.ToKey();
  }

  void Add(State* state_ptr) {
    MotionModelCVCRState* my_state_ptr = reinterpret_cast<MotionModelCVCRState*>(state_ptr);
    variable::Position position;
    position.x(this->position_.x() + my_state_ptr->position().x());
    position.y(this->position_.y() + my_state_ptr->position().y());
    position.floor(this->position_.floor());
    this->position_ = position;
  }

  void Subtract(State* state_ptr) {
    MotionModelCVCRState* my_state_ptr = reinterpret_cast<MotionModelCVCRState*>(state_ptr);
    variable::Position position;
    position.x(this->position_.x() - my_state_ptr->position().x());
    position.y(this->position_.y() - my_state_ptr->position().y());
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

  MotionModelCVCRState(void) {
    this->position_ = variable::Position();
    this->v_ = Eigen::Matrix<double, 2, 1>::Zero(2, 1);
    this->orientation_ = variable::Orientation();
    this->omega_ = Eigen::AngleAxisd(Eigen::Matrix3d::Identity());
  }

  ~MotionModelCVCRState() {}

 private:
  variable::Position position_;
  Eigen::Matrix<double, 2, 1> v_;
  variable::Orientation orientation_;
  Eigen::AngleAxisd omega_;
};

class MotionModelCVCRStateUncertainty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    assert(sample_state_ptrs.size() == weights.size());
    MotionModelCVCRState* my_state_ptr;
    Eigen::Matrix<double, Eigen::Dynamic, 2> position_2d_vectors = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sample_state_ptrs.size(), 2);
    Eigen::Matrix<double, Eigen::Dynamic, 2> vs = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sample_state_ptrs.size(), 2);
    std::vector<variable::Orientation> orientations;
    std::vector<Eigen::AngleAxisd> angle_axis_omegas;

    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModelCVCRState*>(sample_state_ptrs.at(i));
      position_2d_vectors(i, 0) = my_state_ptr->position().x();
      position_2d_vectors(i, 1) = my_state_ptr->position().y();
      vs(i, 0) = my_state_ptr->v()(0);
      vs(i, 1) = my_state_ptr->v()(1);
      orientations.push_back(my_state_ptr->orientation());
      angle_axis_omegas.push_back(my_state_ptr->omega());
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> weights_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(weights.size(), 1);
    for (int i = 0; i < weights.size(); i++) {
      weights_vector(i) = weights.at(i);
    }
    weights_vector = weights_vector.array() / weights_vector.sum();

    position_2d_vectors.col(0) = (position_2d_vectors.col(0).array() - position_2d_vectors.col(0).mean()) * weights_vector.array().sqrt();
    position_2d_vectors.col(1) = (position_2d_vectors.col(1).array() - position_2d_vectors.col(1).mean()) * weights_vector.array().sqrt();
    vs.col(0) = (vs.col(0).array() - vs.col(0).mean()) * weights_vector.array().sqrt();
    vs.col(1) = (vs.col(1).array() - vs.col(1).mean()) * weights_vector.array().sqrt();

    this->position_2d_covariance_ = position_2d_vectors.transpose() * position_2d_vectors;
    this->position_2d_distance_variance_ = position_2d_vectors.rowwise().squaredNorm().sum();
    this->v_covariance_ = vs.transpose() * vs;
    this->orientation_variance_ = variable::Orientation::Variance(orientations, weights);
    this->omega_covariance_ = variable::Orientation::Covariance(angle_axis_omegas, weights);
  }

  Eigen::Matrix<double, 2, 2> position_2d_covariance(void) {
    return this->position_2d_covariance_;
  }

  double position_2d_distance_variance(void) {
    return this->position_2d_distance_variance_;
  }

  Eigen::Matrix<double, 2, 2> v_covariance(void) {
    return this->v_covariance_;
  }

  double orientation_variance(void) {
    return this->orientation_variance_;
  }

  Eigen::Matrix<double, 3, 3> omega_covariance(void) {
    return this->omega_covariance_;
  }

  MotionModelCVCRStateUncertainty(void) {
    this->position_2d_covariance_ = Eigen::Matrix<double, 2, 2>::Zero(2, 2);
    this->position_2d_distance_variance_ = 0.0;
    this->v_covariance_ = Eigen::Matrix<double, 2, 2>::Zero(2, 2);
    this->orientation_variance_ = 0.0;
    this->omega_covariance_ = Eigen::Matrix<double, 3, 3>::Zero(3, 3);
  }

  ~MotionModelCVCRStateUncertainty() {}

 private:
  Eigen::Matrix<double, 2, 2> position_2d_covariance_;
  double position_2d_distance_variance_;
  Eigen::Matrix<double, 2, 2> v_covariance_;
  double orientation_variance_;
  Eigen::Matrix<double, 3, 3> omega_covariance_;
};

class MotionModelCVCRControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNameAccX, this->means_(0));
    named_values->emplace(named_values->end(), util::kNameAccY, this->means_(1));
    named_values->emplace(named_values->end(), util::kNameOmegaAngleaxisAccX, this->means_(2));
    named_values->emplace(named_values->end(), util::kNameOmegaAngleaxisAccY, this->means_(3));
    named_values->emplace(named_values->end(), util::kNameOmegaAngleaxisAccZ, this->means_(4));
  }

  Eigen::Matrix<double, 5, 1> means(void) const {
    return this->means_;
  }

  void means(Eigen::Matrix<double, 5, 1> means) {
    this->means_ = means;
  }

  Eigen::Matrix<double, 5, 5> covariances(void) const {
    return this->covariances_;
  }

  void covariances(Eigen::Matrix<double, 5, 5> covariances) {
    this->covariances_ = covariances;
  }

  MotionModelCVCRControlInput(void) {
    this->means_ = Eigen::Matrix<double, 5, 1>::Zero(5, 1);
    this->covariances_ = Eigen::Matrix<double, 5, 5>::Identity(5, 5);
  }

  ~MotionModelCVCRControlInput() {}

 private:
  Eigen::Matrix<double, 5, 1> means_;
  Eigen::Matrix<double, 5, 5> covariances_;
};

class MotionModelCVCR : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);

  void Seed(int random_seed) {
    this->mvg_sampler_.Seed(random_seed);
  }

  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus,
                                                ControlInput* control_input_t,
                                                double dt);

  MotionModelCVCR(void) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
  }

  ~MotionModelCVCR() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
};

class MotionModelCVCRStateSampler : public StateSampler {
 public:
  void Init(std::set<std::string> sample_space_position_keys,
            double v_single_dimension_min, double v_single_dimension_max,
            double omega_angle_min, double omega_angle_max,
            double position_griding_resolution) {
    this->position_key_sampler_.Init(sample_space_position_keys);
    this->v_single_dimension_sampler_.Init(v_single_dimension_min, v_single_dimension_max);
    this->omega_angle_sampler_.Init(omega_angle_min, omega_angle_max);
    this->position_griding_resolution_ = position_griding_resolution;
  }

  void Sample(State* state_sample) {
    MotionModelCVCRState* my_state_sample =
        reinterpret_cast<MotionModelCVCRState*>(state_sample);

    variable::Position temp_position;
    temp_position.FromKey(this->position_key_sampler_.Sample());
    my_state_sample->position(temp_position);

    Eigen::Matrix<double, 2, 1> temp_v;
    temp_v(0) = this->v_single_dimension_sampler_.Sample();
    temp_v(1) = this->v_single_dimension_sampler_.Sample();
    my_state_sample->v(temp_v);

    variable::Orientation temp_orientation;
    temp_orientation.q(this->orientation_sampler_.Sample());
    my_state_sample->orientation(temp_orientation);

    Eigen::AngleAxisd temp_omega(this->omega_angle_sampler_.Sample(), this->omega_axis_sampler_.Sample());
    my_state_sample->omega(temp_omega);
  }

  void Seed(int random_seed) {
    this->position_key_sampler_.Seed(random_seed);
    this->v_single_dimension_sampler_.Seed(random_seed);
    this->orientation_sampler_.Seed(random_seed);
    this->omega_angle_sampler_.Seed(random_seed);
    this->omega_axis_sampler_.Seed(random_seed);
  }

  double CalculateStateProbabilityLog(State* state_sample) {
#ifdef DEBUG_FOCUSING
    std::cout << "MotionModelCVCRStateSampler::CalculateStateProbabilityLog" << std::endl;
#endif
    MotionModelCVCRState* my_state_sample = reinterpret_cast<MotionModelCVCRState*>(state_sample);
    variable::Position position = my_state_sample->position();
    position.Round(this->position_griding_resolution_);
    double v_x = my_state_sample->v()(0);
    double v_y = my_state_sample->v()(1);
    double omega_angle = my_state_sample->omega().angle();
    std::set<std::string> position_key_set = this->position_key_sampler_.population();
    double v_min = this->v_single_dimension_sampler_.min_value();
    double v_max = this->v_single_dimension_sampler_.max_value();
    double omega_angle_min = this->omega_angle_sampler_.min_value();
    double omega_angle_max = this->omega_angle_sampler_.max_value();

    double log_prob = 0.0;
    if (position_key_set.find(position.ToKey()) != position_key_set.end()) {
      log_prob += std::log(1.0 / position_key_set.size());
    } else {
      log_prob += std::log(0.0);
    }
    if ((v_x >= v_min) && (v_x <= v_max)) {
      log_prob += std::log(1.0 / (v_max - v_min));
    } else {
      log_prob += std::log(0.0);
    }
    if ((v_y >= v_min) && (v_y <= v_max)) {
      log_prob += std::log(1.0 / (v_max - v_min));
    } else {
      log_prob += std::log(0.0);
    }
    if ((omega_angle >= omega_angle_min) && (omega_angle <= omega_angle_max)) {
      log_prob += std::log(1.0 / (omega_angle_max - omega_angle_min));
    } else {
      log_prob += std::log(0.0);
    }

    return log_prob + 3.0 * std::log(1.0 / (2.0 * M_PI)) + 3.0 * std::log(1.0);
  }

  double position_griding_resolution(void) {
    return this->position_griding_resolution_;
  }

  void position_griding_resolution(double position_griding_resolution) {
    this->position_griding_resolution_ = position_griding_resolution;
  }

  MotionModelCVCRStateSampler(void) {}

  ~MotionModelCVCRStateSampler() {}

 private:
  sampler::UniformSetSampler<std::string> position_key_sampler_;
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double> v_single_dimension_sampler_;
  sampler::Uniform3DOrientationSampler orientation_sampler_;
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double> omega_angle_sampler_;
  sampler::Uniform3DRotationAxisSampler omega_axis_sampler_;
  double position_griding_resolution_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_CVCR_H_
