/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2022-02-07 19:23:45
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-03-09 21:07:34
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_3D_ORIENTATION_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_3D_ORIENTATION_H_

#include <cassert>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_3d_orientation_sampler.h"
#include "sampler/gaussian_3d_orientation_sampler.h"
#include "util/misc.h"
#include "util/variable_name_constants.h"
#include "variable/orientation.h"

namespace state_estimation {

namespace prediction_model {

static const double kEpsilon3dOrientation = 1e-10;

class MotionModel3dOrientationState : public State {
 public:
  static const int kNumberOfStateVariables = 1;

  void MakeUnique(void) {
    this->orientation_.normalize();
    if (this->orientation_.w() < kEpsilon3dOrientation) {
      this->orientation_ = Eigen::Quaterniond(-this->orientation_.w(), -this->orientation_.x(), -this->orientation_.y(), -this->orientation_.z());
    }
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear(); this->MakeUnique(); named_values->emplace_back(util::kNameOrientationW, this->orientation_.w());
    named_values->emplace_back(util::kNameOrientationX, this->orientation_.x());
    named_values->emplace_back(util::kNameOrientationY, this->orientation_.y());
    named_values->emplace_back(util::kNameOrientationZ, this->orientation_.z());
  }

  std::string ToKey(std::vector<double> discretization_resolution) {
    assert(discretization_resolution.size() == this->kNumberOfStateVariables);
    double quaternion_resolution = discretization_resolution.at(0);
    this->MakeUnique();
    std::string key_string;
    std::vector<std::string> string_vector;
    string_vector.push_back(std::to_string(UniqueZeroRound(this->orientation_.w(), quaternion_resolution)));
    string_vector.push_back(std::to_string(UniqueZeroRound(this->orientation_.x(), quaternion_resolution)));
    string_vector.push_back(std::to_string(UniqueZeroRound(this->orientation_.y(), quaternion_resolution)));
    string_vector.push_back(std::to_string(UniqueZeroRound(this->orientation_.z(), quaternion_resolution)));
    JoinString(string_vector, "_", &key_string);
    return key_string;
  }

  void EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights);

  void Add(State* state_ptr) {
    MotionModel3dOrientationState* my_state_ptr = reinterpret_cast<MotionModel3dOrientationState*>(state_ptr);
    Eigen::AngleAxisd angle_axis_this(this->orientation_);
    Eigen::AngleAxisd angle_axis_that(my_state_ptr->orientation());
    Eigen::Vector3d log_this = angle_axis_this.angle() * angle_axis_this.axis() + angle_axis_that.angle() * angle_axis_that.axis();
    this->orientation_ = LogVector2Quaternion(log_this);
  }

  void Subtract(State* state_ptr) {
    MotionModel3dOrientationState* my_state_ptr = reinterpret_cast<MotionModel3dOrientationState*>(state_ptr);
    Eigen::AngleAxisd angle_axis_this(this->orientation_);
    Eigen::AngleAxisd angle_axis_that(my_state_ptr->orientation());
    Eigen::Vector3d log_this = angle_axis_this.angle() * angle_axis_this.axis() - angle_axis_that.angle() * angle_axis_that.axis();
    this->orientation_ = LogVector2Quaternion(log_this);
  }

  void Multiply_scalar(double scalar_value) {
    Eigen::AngleAxisd angle_axis_this(this->orientation_);
    Eigen::Vector3d log_this = angle_axis_this.angle() * angle_axis_this.axis() * scalar_value;
    this->orientation_ = LogVector2Quaternion(log_this);
  }

  void Divide_scalar(double scalar_value) {
    Eigen::AngleAxisd angle_axis_this(this->orientation_);
    Eigen::Vector3d log_this = angle_axis_this.angle() * angle_axis_this.axis() / scalar_value;
    this->orientation_ = LogVector2Quaternion(log_this);
  }

  Eigen::Quaterniond orientation(void) {
    this->MakeUnique();
    return this->orientation_;
  }

  void orientation(Eigen::Quaterniond orientation) {
    this->orientation_ = orientation;
    this->MakeUnique();
  }

  MotionModel3dOrientationState(void) {
    this->orientation_ = Eigen::Quaterniond::Identity();
  }

  ~MotionModel3dOrientationState() {}

 private:
  Eigen::Quaterniond orientation_;
};

class MotionModel3dOrientationStateUncertainty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights);

  double geometric_variance(void) {
    return this->geometric_variance_;
  }

  MotionModel3dOrientationStateUncertainty(void) {
    this->geometric_variance_ = 0.0;
  }

  ~MotionModel3dOrientationStateUncertainty() {}

 private:
  double geometric_variance_;
};

class MotionModel3dOrientationDifferentialControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const;

  Eigen::Vector3d dR_log(void) {
    return this->dR_log_;
  }

  void dR_log(Eigen::Vector3d dR_log) {
    this->dR_log_ = dR_log;
  }

  Eigen::Matrix3d dR_log_error_cov(void) {
    return this->dR_log_error_cov_;
  }

  void dR_log_error_cov(Eigen::Matrix3d dR_log_error_cov) {
    this->dR_log_error_cov_ = dR_log_error_cov;
  }

  MotionModel3dOrientationDifferentialControlInput(void) {
    this->dR_log_ = Eigen::Vector3d::Zero();
    this->dR_log_error_cov_ = Eigen::Matrix3d::Zero();
  }

  ~MotionModel3dOrientationDifferentialControlInput() {}

 private:
  Eigen::Vector3d dR_log_;
  Eigen::Matrix3d dR_log_error_cov_;
};

class MotionModel3dOrientationDifferential : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);
  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);

  void Seed(int random_seed) {
    this->mvg_sampler_.Seed(random_seed);
  }

  MotionModel3dOrientationDifferential(void) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
  }

  ~MotionModel3dOrientationDifferential() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
};

class MotionModel3dOrientationUniformStateSampler : public StateSampler {
 public:
  void Init(Eigen::Quaterniond reference_orientation, double max_angular_distance) {
    this->reference_orientation_ = reference_orientation;
    this->max_angular_distance_ = max_angular_distance;
  }

  void Sample(State* state_sample);
  double CalculateStateProbabilityLog(State* state_sample);

  void Seed(int random_seed) {
    this->orientation_sampler_.Seed(random_seed);
  }

  Eigen::Quaterniond reference_orientation(void) {
    return this->reference_orientation_;
  }

  double max_angular_distance(void) {
    return this->max_angular_distance_;
  }

  MotionModel3dOrientationUniformStateSampler(void) {
    this->orientation_sampler_ = sampler::Uniform3DOrientationSampler();
    this->reference_orientation_ = Eigen::Quaterniond::Identity();
    this->max_angular_distance_ = M_PI;
  }

  ~MotionModel3dOrientationUniformStateSampler() {}

 private:
  sampler::Uniform3DOrientationSampler orientation_sampler_;
  Eigen::Quaterniond reference_orientation_;
  double max_angular_distance_;
};

class MotionModel3dOrientationGaussianStateSampler : public StateSampler {
 public:
  void Init(Eigen::Quaterniond q_mean, Eigen::Matrix3d covariance) {
    this->q_mean_ = q_mean;
    this->covariance_ = covariance;
  }

  void Sample(State* state_sample);
  double CalculateStateProbabilityLog(State* state_sample);

  void Seed(int random_seed) {
    this->gaussian_3d_orientation_sampler_.Seed(random_seed);
  }

  Eigen::Quaterniond q_mean(void) {
    return this->q_mean_;
  }

  Eigen::Matrix3d covariance(void) {
    return this->covariance_;
  }

  MotionModel3dOrientationGaussianStateSampler(void) {
    this->gaussian_3d_orientation_sampler_ = sampler::Gaussian3DOrientationSampler();
    this->q_mean_ = Eigen::Quaterniond::Identity();
    this->covariance_ = Eigen::Matrix3d::Zero();
  }

  ~MotionModel3dOrientationGaussianStateSampler() {}

 private:
  sampler::Gaussian3DOrientationSampler gaussian_3d_orientation_sampler_;
  Eigen::Quaterniond q_mean_;
  Eigen::Matrix3d covariance_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_3D_ORIENTATION_H_
