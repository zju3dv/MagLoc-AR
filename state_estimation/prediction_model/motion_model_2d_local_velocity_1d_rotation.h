/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-12-22 10:49:49
 * @LastEditTime: 2023-03-27 16:35:01
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_LOCAL_VELOCITY_1D_ROTATION_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_LOCAL_VELOCITY_1D_ROTATION_H_

#include <Eigen/Dense>

#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "distribution/probability_mapper_2d.h"
#include "distribution/gaussian_distribution.h"
#include "prediction_model/base.h"
#include "prediction_model/motion_model_yaw_differential.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_range_sampler.h"
#include "sampler/uniform_set_sampler.h"
#include "variable/orientation.h"
#include "variable/position.h"
#include "util/misc.h"
#include "util/variable_name_constants.h"

namespace state_estimation {

namespace prediction_model {

class MotionModel2dLocalVelocity1dRotationState : public State {
 public:
  static const int kNumberOfStateVariables = 1 + MotionModelYawDifferentialState::kNumberOfStateVariables;

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNamePositionX, this->position_.x());
    named_values->emplace(named_values->end(), util::kNamePositionY, this->position_.y());

    std::vector<std::pair<std::string, double>> rotation_state_named_values;
    this->motion_model_1d_rotation_state_.GetAllNamedValues(&rotation_state_named_values);
    for (int i = 0; i < rotation_state_named_values.size(); i++) {
      if (rotation_state_named_values.at(i).first == util::kNamePositionX) {
        continue;
      }
      if (rotation_state_named_values.at(i).first == util::kNamePositionY) {
        continue;
      }
      named_values->emplace_back(rotation_state_named_values.at(i));
    }
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
    double mean_state_prediction_log_probability;
    double mean_state_update_log_probability;
    MotionModel2dLocalVelocity1dRotationState* my_state_ptr;
    std::vector<State*> rotation_sample_state_ptrs;
    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(sample_state_ptrs.at(i));
      double weighted_state_prediction_log_probability = 0.0;
      double weighted_state_update_log_probability = 0.0;
      if (weights.at(i) < 1e-20) {
        weighted_state_prediction_log_probability = 0.0;
        weighted_state_update_log_probability = 0.0;
      } else {
        weighted_state_prediction_log_probability = my_state_ptr->state_prediction_log_probability() * weights.at(i);
        weighted_state_update_log_probability = my_state_ptr->state_update_log_probability() * weights.at(i);
      }
      if (i == 0) {
        mean_position = my_state_ptr->position() * weights.at(i);
        mean_state_prediction_log_probability = weighted_state_prediction_log_probability;
        mean_state_update_log_probability = weighted_state_update_log_probability;
      } else {
        mean_position = mean_position + my_state_ptr->position() * weights.at(i);
        mean_state_prediction_log_probability += weighted_state_prediction_log_probability;
        mean_state_update_log_probability += weighted_state_update_log_probability;
      }
      rotation_sample_state_ptrs.push_back(my_state_ptr->motion_model_1d_rotation_state_ptr());
    }

    MotionModelYawDifferentialState rotation_estimate_state;
    rotation_estimate_state.EstimateFromSamples(rotation_sample_state_ptrs, weights);

    this->position_ = mean_position;
    this->motion_model_1d_rotation_state_ = rotation_estimate_state;
    this->state_prediction_log_probability_ = mean_state_prediction_log_probability;
    this->state_update_log_probability_ = mean_state_update_log_probability;
  }

  variable::Position position(void) const {
    return this->position_;
  }

  void position(variable::Position position) {
    this->position_ = position;
  }

  MotionModelYawDifferentialState motion_model_1d_rotation_state(void) {
    return this->motion_model_1d_rotation_state_;
  }

  void motion_model_1d_rotation_state(MotionModelYawDifferentialState motion_model_1d_rotation_state) {
    this->motion_model_1d_rotation_state_ = motion_model_1d_rotation_state;
  }

  MotionModelYawDifferentialState* motion_model_1d_rotation_state_ptr(void) {
    return &(this->motion_model_1d_rotation_state_);
  }

  std::string ToKey(std::vector<double> discretization_resolution) {
    assert(discretization_resolution.size() == this->kNumberOfStateVariables);
    double position_resolution = discretization_resolution.at(0);
    variable::Position temp_position = this->position_;
    temp_position.Round(position_resolution);
    std::vector<double> orientation_discretization_resolutions = {discretization_resolution.at(1)};
    return temp_position.ToKey() + "_" + this->motion_model_1d_rotation_state_.ToKey(orientation_discretization_resolutions);
  }

  void Add(State* state_ptr) {
    MotionModel2dLocalVelocity1dRotationState* my_state_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() + temp_position.x());
    position.y(this->position_.y() + temp_position.y());
    position.floor(this->position_.floor());
    this->position_ = position;
    MotionModelYawDifferentialState* rotation_state_ptr = my_state_ptr->motion_model_1d_rotation_state_ptr();
    this->motion_model_1d_rotation_state_.Add(rotation_state_ptr);
  }

  void Subtract(State* state_ptr) {
    MotionModel2dLocalVelocity1dRotationState* my_state_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() - temp_position.x());
    position.y(this->position_.y() - temp_position.y());
    position.floor(this->position_.floor());
    this->position_ = position;
    MotionModelYawDifferentialState* rotation_state_ptr = my_state_ptr->motion_model_1d_rotation_state_ptr();
    this->motion_model_1d_rotation_state_.Subtract(rotation_state_ptr);
  }

  void Multiply_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() * scalar_value);
    position.y(this->position_.y() * scalar_value);
    position.floor(this->position_.floor());
    this->position_ = position;
    this->motion_model_1d_rotation_state_.Multiply_scalar(scalar_value);
  }

  void Divide_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() / scalar_value);
    position.y(this->position_.y() / scalar_value);
    position.floor(this->position_.floor());
    this->position_ = position;
    this->motion_model_1d_rotation_state_.Divide_scalar(scalar_value);
  }

  MotionModel2dLocalVelocity1dRotationState(void) {
    this->position_ = variable::Position();
    this->motion_model_1d_rotation_state_ = MotionModelYawDifferentialState();
  }

  ~MotionModel2dLocalVelocity1dRotationState() {}

 private:
  variable::Position position_;  // the 2d position under the map coordinate-system
  MotionModelYawDifferentialState motion_model_1d_rotation_state_;  // yaw under the world coordinate-system
};

class MotionModel2dLocalVelocity1dRotationStateUncertainty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    assert(sample_state_ptrs.size() == weights.size());
    Eigen::Matrix<double, Eigen::Dynamic, 2> position_2d_vectors = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sample_state_ptrs.size(), 2);
    MotionModel2dLocalVelocity1dRotationState* my_state_ptr;
    std::vector<State*> rotation_state_sample_ptrs;

    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(sample_state_ptrs.at(i));
      position_2d_vectors(i, 0) = my_state_ptr->position().x();
      position_2d_vectors(i, 1) = my_state_ptr->position().y();
      rotation_state_sample_ptrs.push_back(my_state_ptr->motion_model_1d_rotation_state_ptr());
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
    this->motion_model_1d_rotation_state_uncertainty_.CalculateFromSamples(rotation_state_sample_ptrs, weights);
  }

  Eigen::Matrix<double, 2, 2> position_2d_covariance(void) {
    return this->position_2d_covariance_;
  }

  double position_2d_distance_variance(void) {
    return this->position_2d_distance_variance_;
  }

  MotionModelYawDifferentialStateUncertainty motion_model_1d_rotation_state_uncertainty(void) {
    return this->motion_model_1d_rotation_state_uncertainty_;
  }

  void motion_model_1d_rotation_state_uncertainty(MotionModelYawDifferentialStateUncertainty motion_model_1d_rotation_state_uncertainty) {
    this->motion_model_1d_rotation_state_uncertainty_ = motion_model_1d_rotation_state_uncertainty;
  }

  MotionModel2dLocalVelocity1dRotationStateUncertainty(void) {
    this->position_2d_covariance_ = Eigen::Matrix<double, 2, 2>::Zero(2, 2);
    this->position_2d_distance_variance_ = 0.0;
    this->motion_model_1d_rotation_state_uncertainty_ = MotionModelYawDifferentialStateUncertainty();
  }

  ~MotionModel2dLocalVelocity1dRotationStateUncertainty() {}

 private:
  Eigen::Matrix<double, 2, 2> position_2d_covariance_;
  double position_2d_distance_variance_;
  MotionModelYawDifferentialStateUncertainty motion_model_1d_rotation_state_uncertainty_;
};

class MotionModel2dLocalVelocity1dRotationControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNameGravityX, this->gravity_(0));
    named_values->emplace(named_values->end(), util::kNameGravityY, this->gravity_(1));
    named_values->emplace(named_values->end(), util::kNameGravityZ, this->gravity_(2));
    named_values->emplace(named_values->end(), util::kNameVLocalX, this->v_local_(0));
    named_values->emplace(named_values->end(), util::kNameVLocalY, this->v_local_(1));
    named_values->emplace(named_values->end(), util::kNameVLocalZ, this->v_local_(2));
    named_values->emplace(named_values->end(), "cov_0", this->v_local_covariance_(0, 0));
    named_values->emplace(named_values->end(), "cov_1", this->v_local_covariance_(0, 1));
    named_values->emplace(named_values->end(), "cov_2", this->v_local_covariance_(0, 2));
    named_values->emplace(named_values->end(), "cov_3", this->v_local_covariance_(1, 0));
    named_values->emplace(named_values->end(), "cov_4", this->v_local_covariance_(1, 1));
    named_values->emplace(named_values->end(), "cov_5", this->v_local_covariance_(1, 2));
    named_values->emplace(named_values->end(), "cov_6", this->v_local_covariance_(2, 0));
    named_values->emplace(named_values->end(), "cov_7", this->v_local_covariance_(2, 1));
    named_values->emplace(named_values->end(), "cov_8", this->v_local_covariance_(2, 2));

    std::vector<std::pair<std::string, double>> rotation_control_input_named_values;
    this->motion_model_1d_rotation_control_input_.GetAllNamedValues(&rotation_control_input_named_values);
    for (int i = 0; i < rotation_control_input_named_values.size(); i++) {
      named_values->emplace_back(rotation_control_input_named_values.at(i));
    }
  }

  Eigen::Matrix<double, 3, 1> gravity(void) {
    return this->gravity_;
  }

  void gravity(Eigen::Matrix<double, 3, 1> gravity) {
    this->gravity_ = gravity;
  }

  Eigen::Matrix<double, 3, 1> v_local(void) {
    return this->v_local_;
  }

  void v_local(Eigen::Matrix<double, 3, 1> v_local) {
    this->v_local_ = v_local;
  }

  Eigen::Matrix<double, 3, 3> v_local_covariance(void) {
    return this->v_local_covariance_;
  }

  void v_local_covariance(Eigen::Matrix<double, 3, 3> v_local_covariance) {
    this->v_local_covariance_ = v_local_covariance;
  }

  MotionModelYawDifferentialControlInput motion_model_1d_rotation_control_input(void) {
    return this->motion_model_1d_rotation_control_input_;
  }

  void motion_model_1d_rotation_control_input(MotionModelYawDifferentialControlInput motion_model_1d_rotation_control_input) {
    this->motion_model_1d_rotation_control_input_ = motion_model_1d_rotation_control_input;
  }

  Eigen::Quaterniond orientation_sensor_pose_ws(void) {
    return this->orientation_sensor_pose_ws_;
  }

  void orientation_sensor_pose_ws(Eigen::Quaterniond orientation_sensor_pose_ws) {
    this->orientation_sensor_pose_ws_ = orientation_sensor_pose_ws;
  }

  bool use_estimated_yaw(void) {
    return this->use_estimated_yaw_;
  }

  void use_estimated_yaw(bool use_estimated_yaw) {
    this->use_estimated_yaw_ = use_estimated_yaw;
  }

  MotionModel2dLocalVelocity1dRotationControlInput(void) {
    this->gravity_ = Eigen::Matrix<double, 3, 1>{0.0, 0.0, 1.0};
    this->v_local_ = Eigen::Matrix<double, 3, 1>::Zero();
    this->v_local_covariance_ = Eigen::Matrix<double, 3, 3>::Zero();
    this->motion_model_1d_rotation_control_input_ = MotionModelYawDifferentialControlInput();
    this->orientation_sensor_pose_ws_ = Eigen::Quaterniond::Identity();
    this->use_estimated_yaw_ = true;
  }

  ~MotionModel2dLocalVelocity1dRotationControlInput() {}

 private:
  Eigen::Matrix<double, 3, 1> gravity_;
  Eigen::Matrix<double, 3, 1> v_local_;
  Eigen::Matrix<double, 3, 3> v_local_covariance_;
  MotionModelYawDifferentialControlInput motion_model_1d_rotation_control_input_;
  Eigen::Quaterniond orientation_sensor_pose_ws_;
  bool use_estimated_yaw_;
};

class MotionModel2dLocalVelocity1dRotation : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction = false);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);

  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);

  void JitterState(State* state_t);

  void Seed(int random_seed) {
    this->mvg_sampler_.Seed(random_seed);
    this->motion_model_1d_rotation_.Seed(random_seed);
    this->position_jitter_mvg_sampler_.Seed(random_seed);
  }

  void SetPositionJitteringDistributionCovariance(Eigen::Matrix2d position_jitter_covariance) {
    this->position_jitter_mvg_sampler_.SetParams(Eigen::Vector2d::Zero(), position_jitter_covariance);
  }

  void SetYawJitteringDistributionVariance(double yaw_jitter_variance) {
    this->motion_model_1d_rotation_.SetYawJitteringDistributionVariance(yaw_jitter_variance);
  }

  Eigen::Matrix3d R_mw(void) {
    return this->R_mw_;
  }

  void R_mw(Eigen::Matrix3d R_mw) {
    this->R_mw_ = R_mw;
  }

  bool position_jitter_flag(void) {
    return this->position_jitter_flag_;
  }

  void position_jitter_flag(bool position_jitter_flag) {
    this->position_jitter_flag_ = position_jitter_flag;
  }

  bool yaw_jitter_flag(void) {
    return this->yaw_jitter_flag_;
  }

  void yaw_jitter_flag(bool yaw_jitter_flag) {
    this->yaw_jitter_flag_ = yaw_jitter_flag;
  }

  MotionModel2dLocalVelocity1dRotation(Eigen::Matrix<double, 3, 3> R_mw = Eigen::Matrix<double, 3, 3>::Identity()) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
    this->position_jitter_mvg_sampler_ = sampler::MultivariateGaussianSampler();
    this->position_jitter_mvg_sampler_.Init(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
    this->motion_model_1d_rotation_ = MotionModelYawDifferential();
    this->R_mw_ = R_mw;
    this->position_jitter_flag_ = false;
    this->yaw_jitter_flag_ = false;
  }

  ~MotionModel2dLocalVelocity1dRotation() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
  sampler::MultivariateGaussianSampler position_jitter_mvg_sampler_;
  MotionModelYawDifferential motion_model_1d_rotation_;
  Eigen::Matrix<double, 3, 3> R_mw_;
  bool position_jitter_flag_;
  bool yaw_jitter_flag_;
};

template <typename MotionModelYawDifferentialStateSampler>
class MotionModel2dLocalVelocity1dRotationStateSampler : public StateSampler {
 public:
  void Init(std::set<std::string> sample_space_position_keys,
            double position_griding_resolution,
            MotionModelYawDifferentialStateSampler motion_model_1d_rotation_state_sampler) {
    this->position_key_sampler_.Init(sample_space_position_keys);
    this->position_griding_resolution_ = position_griding_resolution;
    this->motion_model_1d_rotation_state_sampler_ = motion_model_1d_rotation_state_sampler;
  }

  void Sample(State* state_sample) {
#ifdef DEBUG_FOCUSING
    std::cout << "MotionModel2dLocalVelocity1dRotationStateSampler::Sample" << std::endl;
#endif
    MotionModel2dLocalVelocity1dRotationState* my_state_sample = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_sample);
    variable::Position temp_position;
    temp_position.FromKey(this->position_key_sampler_.Sample());
    my_state_sample->position(temp_position);
    this->motion_model_1d_rotation_state_sampler_.Sample(my_state_sample->motion_model_1d_rotation_state_ptr());
    my_state_sample->state_prediction_log_probability(this->CalculateStateProbabilityLog(state_sample));
    my_state_sample->state_update_log_probability(0.0);
  }

  void Seed(int random_seed) {
    this->position_key_sampler_.Seed(random_seed);
    this->motion_model_1d_rotation_state_sampler_.Seed(random_seed);
  }

  double CalculateStateProbabilityLog(State* state_sample) {
    MotionModel2dLocalVelocity1dRotationState* my_state_sample = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_sample);
    variable::Position position = my_state_sample->position();
    position.Round(this->position_griding_resolution_);
    std::set<std::string> position_key_set = this->position_key_sampler_.population();

    double log_prob = 0.0;
    if (position_key_set.find(position.ToKey()) != position_key_set.end()) {
      log_prob += std::log(1.0 / position_key_set.size());
    } else {
      log_prob += std::log(0.0);
    }

    log_prob += this->motion_model_1d_rotation_state_sampler_.CalculateStateProbabilityLog(my_state_sample->motion_model_1d_rotation_state_ptr());

    return log_prob;
  }

  double position_griding_resolution(void) {
    return this->position_griding_resolution_;
  }

  void position_griding_resolution(double position_griding_resolution) {
    this->position_griding_resolution_ = position_griding_resolution;
  }

  MotionModelYawDifferentialStateSampler motion_model_1d_rotation_state_sampler(void) {
    return this->motion_model_1d_rotation_state_sampler_;
  }

  void motion_model_1d_rotation_state_sampler(MotionModelYawDifferentialStateSampler motion_model_1d_rotation_state_sampler) {
    this->motion_model_1d_rotation_state_sampler_ = motion_model_1d_rotation_state_sampler;
  }

  MotionModel2dLocalVelocity1dRotationStateSampler(void) {
    this->position_key_sampler_ = sampler::UniformSetSampler<std::string>();
    this->position_griding_resolution_ = 1.0;
    this->motion_model_1d_rotation_state_sampler_ = MotionModelYawDifferentialStateSampler();
  }

  ~MotionModel2dLocalVelocity1dRotationStateSampler() {}

 private:
  sampler::UniformSetSampler<std::string> position_key_sampler_;
  double position_griding_resolution_;
  MotionModelYawDifferentialStateSampler motion_model_1d_rotation_state_sampler_;
};

class MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler : public StateSampler {
 public:
  void Init(distribution::ProbabilityMapper2D geomagnetism_probability_mapper, double position_griding_resolution, Eigen::Matrix3d R_mw, Eigen::Vector3d geomagnetism_s, Eigen::Matrix3d geomagnetism_s_cov, Eigen::Vector3d gravity_s, Eigen::Matrix3d gravity_s_cov, Eigen::Quaterniond q_ms_gt = Eigen::Quaterniond::Identity(), bool use_gt = false) {
    this->geomagnetism_probability_mapper_ = geomagnetism_probability_mapper;
    this->R_mw_ = R_mw;
    this->position_key_sampler_.Init(geomagnetism_probability_mapper.GetStateKeys());
    this->geomagnetism_s_sampler_.SetParams(geomagnetism_s, geomagnetism_s_cov);
    this->gravity_s_sampler_.SetParams(gravity_s, gravity_s_cov);
    this->q_ms_gt_ = q_ms_gt;
    this->use_gt_ = use_gt;
    this->position_griding_resolution_ = position_griding_resolution;
    this->position_keys_ = geomagnetism_probability_mapper.GetStateKeys();
    std::set<double> yaw_shift_degree_set = {-60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    this->yaw_shift_sampler_.Init(yaw_shift_degree_set);
  }

  void Sample(State* state_sample) {
#ifdef DEBUG_FOCUSING
    std::cout << "MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler::Sample" << std::endl;
#endif
    MotionModel2dLocalVelocity1dRotationState* my_state_sample = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_sample);
    variable::Position temp_position;
    std::string sampled_position_key = this->position_key_sampler_.Sample();
    temp_position.FromKey(sampled_position_key);
    my_state_sample->position(temp_position);
    if (this->use_gt_) {
      Eigen::Quaterniond q_ws_gt(this->R_mw_.transpose() * this->q_ms_gt_.normalized());
      my_state_sample->motion_model_1d_rotation_state_ptr()->q_ws(q_ws_gt);
      return;
    }
    const std::unordered_map<std::string, std::vector<double>>* geomagnetism_distribution = this->geomagnetism_probability_mapper_.GetDistributionParams(sampled_position_key);
    assert(geomagnetism_distribution->find("x") != geomagnetism_distribution->end());
    assert(geomagnetism_distribution->find("y") != geomagnetism_distribution->end());
    assert(geomagnetism_distribution->find("z") != geomagnetism_distribution->end());
    Eigen::Vector3d m_map = Eigen::Vector3d::Zero();
    m_map(0) = geomagnetism_distribution->at("x").at(0);
    m_map(1) = geomagnetism_distribution->at("y").at(0);
    m_map(2) = geomagnetism_distribution->at("z").at(0);

    // through all the calculations below, we assume that the gravity percepted by the gravity sensor is exact.
    Eigen::Vector3d g_world = Eigen::Vector3d({0.0, 0.0, 1.0});
    Eigen::Matrix3d R_mwlocal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g_map = this->R_mw_ * g_world;
    R_mwlocal.col(0) = (m_map.cross(g_map)).normalized();
    R_mwlocal.col(1) = (g_map.cross(R_mwlocal.col(0))).normalized();
    R_mwlocal.col(2) = g_map.normalized();

    Eigen::Vector3d gravity_s_sample = this->gravity_s_sampler_.Sample();
    Eigen::Vector3d geomagnetism_s_sample = this->geomagnetism_s_sampler_.Sample();
    Eigen::Matrix3d R_wlocals = Eigen::Matrix3d::Zero();
    R_wlocals.row(0) = (geomagnetism_s_sample.cross(gravity_s_sample)).normalized();
    R_wlocals.row(1) = (gravity_s_sample.cross(R_wlocals.row(0))).normalized();
    R_wlocals.row(2) = gravity_s_sample.normalized();

    Eigen::Quaterniond q_ws_sample(this->R_mw_.transpose() * R_mwlocal * R_wlocals);
    // uniformly rotated the yaw
    double yaw_shift_degree = this->yaw_shift_sampler_.Sample();
    Eigen::Quaterniond q_yaw_shift(Eigen::AngleAxisd(yaw_shift_degree / 180.0 * M_PI, g_world));
    my_state_sample->motion_model_1d_rotation_state_ptr()->q_ws(q_yaw_shift * q_ws_sample);

    // calculate the sampled state log probability.
    std::vector<double> gravity_sample_vector = EigenVector2Vector(gravity_s_sample);
    std::vector<double> geomagnetism_sample_vector = EigenVector2Vector(geomagnetism_s_sample);
    std::vector<double> gravity_mean_vector = EigenVector2Vector(this->gravity_s_sampler_.mean());
    std::vector<double> gravity_covariance_vector = CovarianceMatrixToCompactVector(this->gravity_s_sampler_.covariance());
    std::vector<double> geomagnetism_mean_vector = EigenVector2Vector(this->geomagnetism_s_sampler_.mean());
    std::vector<double> geomagnetism_covariance_vector = CovarianceMatrixToCompactVector(this->geomagnetism_s_sampler_.covariance());
    distribution::MultivariateGaussian mvg_gravity(gravity_mean_vector, gravity_covariance_vector);
    distribution::MultivariateGaussian mvg_geomagnetism(geomagnetism_mean_vector, geomagnetism_covariance_vector);
    my_state_sample->state_prediction_log_probability(std::log(1.0 / this->position_key_sampler_.population().size()) +
                                                      std::log(mvg_gravity.QuantizedProbability(gravity_sample_vector)) +
                                                      std::log(mvg_geomagnetism.QuantizedProbability(geomagnetism_sample_vector)));
    my_state_sample->state_update_log_probability(0.0);
  }

  void Seed(int random_seed) {
    this->position_key_sampler_.Seed(random_seed);
    this->univariate_gaussian_sampler_.Seed(random_seed);
    this->gravity_s_sampler_.Seed(random_seed);
    this->geomagnetism_s_sampler_.Seed(random_seed);
  }

  double position_griding_resolution(void) {
    return this->position_griding_resolution_;
  }

  void position_griding_resolution(double position_griding_resolution) {
    this->position_griding_resolution_ = position_griding_resolution;
  }

  int geomagnetism_probability_mapper(distribution::ProbabilityMapper2D *res) {
    *res = this->geomagnetism_probability_mapper_;
    return 1;
  }

  int R_mw(Eigen::Matrix3d *res) {
    *res = this->R_mw_;
    return 1;
  }

  int gravity_s(Eigen::Vector3d *res) {
    *res = this->gravity_s_sampler_.mean();
    return 1;
  }

  int geomagnetism_s(Eigen::Vector3d *res) {
    *res = this->geomagnetism_s_sampler_.mean();
    return 1;
  }

  double CalculateStateProbabilityLog(State* state_sample) {
    MotionModel2dLocalVelocity1dRotationState* my_state_sample = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_sample);
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

  int SetPositionPopulation(std::set<std::string> position_population) {
    this->position_key_sampler_.Init(position_population);
    return this->position_key_sampler_.population().size();
  }

  void Traverse(State* state_sample) {
#ifdef DEBUG_FOCUSING
    std::cout << "MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler::Traverse" << std::endl;
#endif
    if (this->IsTraverseFinished()) {
      std::cout << "MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler::Traverse: traversing finished." << std::endl;
      return;
    }
    if (this->traversed_position_sample_count_ >= this->samples_per_position_for_traversing_) {
      std::cout << "MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler::Traverse: traversed_position_count is equal to or larger than samples_per_position_for_traversing." << std::endl;
      return;
    }
    MotionModel2dLocalVelocity1dRotationState* my_state_sample = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(state_sample);
    variable::Position temp_position;
    std::string sampled_position_key = *(this->position_key_traverse_iterator_);
    this->traversed_position_sample_count_++;
    if (this->traversed_position_sample_count_ >= this->samples_per_position_for_traversing_) {
      this->position_key_traverse_iterator_++;
      this->traversed_position_sample_count_ = 0;
    }
    temp_position.FromKey(sampled_position_key);
    my_state_sample->position(temp_position);
    const std::unordered_map<std::string, std::vector<double>>* geomagnetism_distribution = this->geomagnetism_probability_mapper_.GetDistributionParams(sampled_position_key);
    assert(geomagnetism_distribution->find("x") != geomagnetism_distribution->end());
    assert(geomagnetism_distribution->find("y") != geomagnetism_distribution->end());
    assert(geomagnetism_distribution->find("z") != geomagnetism_distribution->end());
    Eigen::Vector3d m_map = Eigen::Vector3d::Zero();
    m_map(0) = geomagnetism_distribution->at("x").at(0);
    m_map(1) = geomagnetism_distribution->at("y").at(0);
    m_map(2) = geomagnetism_distribution->at("z").at(0);

    Eigen::Vector3d g_world = Eigen::Vector3d({0.0, 0.0, 1.0});
    Eigen::Matrix3d R_mwlocal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g_map = this->R_mw_ * g_world;
    R_mwlocal.col(0) = (m_map.cross(g_map)).normalized();
    R_mwlocal.col(1) = (g_map.cross(R_mwlocal.col(0))).normalized();
    R_mwlocal.col(2) = g_map.normalized();

    Eigen::Vector3d gravity_s_sample = this->gravity_s_sampler_.Sample();
    Eigen::Vector3d geomagnetism_s_sample = this->geomagnetism_s_sampler_.Sample();
    Eigen::Matrix3d R_wlocals = Eigen::Matrix3d::Zero();
    R_wlocals.row(0) = (geomagnetism_s_sample.cross(gravity_s_sample)).normalized();
    R_wlocals.row(1) = (gravity_s_sample.cross(R_wlocals.row(0))).normalized();
    R_wlocals.row(2) = gravity_s_sample.normalized();

    Eigen::Quaterniond q_ws_sample(this->R_mw_.transpose() * R_mwlocal * R_wlocals);
    my_state_sample->motion_model_1d_rotation_state_ptr()->q_ws(q_ws_sample);

    // calculate the sampled state log probability.
    std::vector<double> gravity_sample_vector = EigenVector2Vector(gravity_s_sample);
    std::vector<double> geomagnetism_sample_vector = EigenVector2Vector(geomagnetism_s_sample);
    std::vector<double> gravity_mean_vector = EigenVector2Vector(this->gravity_s_sampler_.mean());
    std::vector<double> gravity_covariance_vector = CovarianceMatrixToCompactVector(this->gravity_s_sampler_.covariance());
    std::vector<double> geomagnetism_mean_vector = EigenVector2Vector(this->geomagnetism_s_sampler_.mean());
    std::vector<double> geomagnetism_covariance_vector = CovarianceMatrixToCompactVector(this->geomagnetism_s_sampler_.covariance());
    distribution::MultivariateGaussian mvg_gravity(gravity_mean_vector, gravity_covariance_vector);
    distribution::MultivariateGaussian mvg_geomagnetism(geomagnetism_mean_vector, geomagnetism_covariance_vector);
    my_state_sample->state_prediction_log_probability(std::log(1.0 / this->position_key_sampler_.population().size()) +
                                                      std::log(mvg_gravity.QuantizedProbability(gravity_sample_vector)) +
                                                      std::log(mvg_geomagnetism.QuantizedProbability(geomagnetism_sample_vector)));
    my_state_sample->state_update_log_probability(0.0);
  }

  void ResetTraverseState(void) {
    this->position_key_traverse_iterator_ = this->position_keys_.begin();
    this->traversed_position_sample_count_ = 0;
  }

  bool IsTraverseFinished(void) {
#ifdef DEBUG_FOCUSING
    std::cout << "MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler::IsTraverseFinished" << std::endl;
#endif
    return (this->position_key_traverse_iterator_ == this->position_keys_.end());
  }

  void samples_per_position_for_traversing(int samples_per_position_for_traversing) {
    this->samples_per_position_for_traversing_ = samples_per_position_for_traversing;
  }

  int samples_per_position_for_traversing(void) {
    return this->samples_per_position_for_traversing_;
  }

  MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler(void) {
    this->geomagnetism_probability_mapper_ = distribution::ProbabilityMapper2D();
    this->position_key_sampler_ = sampler::UniformSetSampler<std::string>();
    this->position_griding_resolution_ = 0.5;
    this->univariate_gaussian_sampler_ = sampler::UnivariateGaussianSamplerStd();
    this->gravity_s_sampler_ = sampler::MultivariateGaussianSampler();
    this->geomagnetism_s_sampler_ = sampler::MultivariateGaussianSampler();
    this->R_mw_ = Eigen::Matrix3d::Identity();
    this->q_ms_gt_ = Eigen::Quaterniond::Identity();
    this->use_gt_ = false;
    this->position_keys_ = std::set<std::string>();
    this->position_key_traverse_iterator_ = std::set<std::string>::iterator();
    this->samples_per_position_for_traversing_ = 0;
    this->traversed_position_sample_count_ = 0;
  }

  ~MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler() {}

 private:
  distribution::ProbabilityMapper2D geomagnetism_probability_mapper_;
  sampler::UniformSetSampler<std::string> position_key_sampler_;
  sampler::UniformSetSampler<double> yaw_shift_sampler_;
  double position_griding_resolution_;
  sampler::UnivariateGaussianSamplerStd univariate_gaussian_sampler_;
  sampler::MultivariateGaussianSampler gravity_s_sampler_;
  sampler::MultivariateGaussianSampler geomagnetism_s_sampler_;
  Eigen::Matrix3d R_mw_;
  Eigen::Quaterniond q_ms_gt_;
  bool use_gt_;
  std::set<std::string> position_keys_;
  std::set<std::string>::iterator position_key_traverse_iterator_;
  int samples_per_position_for_traversing_;
  int traversed_position_sample_count_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_LOCAL_VELOCITY_1D_ROTATION_H_
