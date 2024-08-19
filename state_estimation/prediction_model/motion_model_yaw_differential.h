/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2022-02-07 19:23:45
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-02-07 19:45:35
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_YAW_DIFFERENTIAL_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_YAW_DIFFERENTIAL_H_

#include <Eigen/Eigen>

#include <cassert>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_range_sampler.h"
#include "util/misc.h"
#include "util/variable_name_constants.h"
#include "variable/orientation.h"

namespace state_estimation {

namespace prediction_model {

class MotionModelYawDifferentialState : public State {
 public:
  static const int kNumberOfStateVariables = 1;

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNameYaw, this->yaw_);
    named_values->emplace(named_values->end(), util::kNameOrientationW, this->q_ws_.w());
    named_values->emplace(named_values->end(), util::kNameOrientationX, this->q_ws_.x());
    named_values->emplace(named_values->end(), util::kNameOrientationY, this->q_ws_.y());
    named_values->emplace(named_values->end(), util::kNameOrientationZ, this->q_ws_.z());
  }

  std::string ToKey(std::vector<double> discretization_resolution) {
    assert(discretization_resolution.size() == this->kNumberOfStateVariables);
    double yaw_resolution_in_rad = discretization_resolution.at(0);
    return std::to_string(UniqueZeroRound(this->yaw_, yaw_resolution_in_rad));
  }

  void EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights);

  void Add(State* state_ptr) {
    std::cout << "MotionModelYawDifferrentialState::Add: the implementation is left empty since the orientation cannot theoretically be operated linearly." << std::endl;
    assert(false);
  }

  void Subtract(State* state_ptr) {
    std::cout << "MotionModelYawDifferrentialState::Subtract: the implementation is left empty since the orientation cannot theoretically be operated linearly." << std::endl;
    assert(false);
  }

  void Multiply_scalar(double scalar_value) {
    std::cout << "MotionModelYawDifferrentialState::Multiply_scalar: the implementation is left empty since the orientation cannot theoretically be operated linearly." << std::endl;
    assert(false);
  }

  void Divide_scalar(double scalar_value) {
    std::cout << "MotionModelYawDifferrentialState::Divide_scalar: the implementation is left empty since the orientation cannot theoretically be operated linearly." << std::endl;
    assert(false);
  }

  double yaw(void) const {
    return this->yaw_;
  }

  void yaw(double yaw) {
    this->yaw_ = yaw;
    this->q_ws_ = Eigen::Quaterniond(CalculateRotationMatrixAlongAxisZ(yaw)) * this->q_sgs_.normalized();
  }

  Eigen::Quaterniond q_sgs(void) {
    return this->q_sgs_.normalized();
  }

  void q_sgs(const Eigen::Quaterniond& q_sgs) {
    this->q_sgs_ = q_sgs.normalized();
    this->yaw(this->yaw_);
  }

  Eigen::Quaterniond q_ws(void) {
    return this->q_ws_.normalized();
  }

  void q_ws(Eigen::Quaterniond q_ws) {
    this->q_ws_ = q_ws.normalized();
    Eigen::Quaterniond q_wsg;
    Eigen::Quaterniond q_sgs;
    OrientationDecompositionRzRxy(this->q_ws_, &q_wsg, &q_sgs);
    this->q_sgs_ = q_sgs.normalized();
    Eigen::AngleAxisd angleaxis_wsg(q_wsg.normalized());
    this->yaw_ = GetAngleByAxisFromAngleAxis(angleaxis_wsg, Eigen::Vector3d({0.0, 0.0, 1.0}));
  }

  MotionModelYawDifferentialState(void) {
    this->yaw_ = 0.0;
    this->q_sgs_ = Eigen::Quaterniond::Identity();
    this->q_ws_ = Eigen::Quaterniond::Identity();
  }

  ~MotionModelYawDifferentialState() {}

 private:
  double yaw_;
  Eigen::Quaterniond q_sgs_;
  Eigen::Quaterniond q_ws_;
};

class MotionModelYawDifferentialStateUncertainty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights);

  double yaw_variance(void) {
    return this->yaw_variance_;
  }

  MotionModelYawDifferentialStateUncertainty(void) {
    this->yaw_variance_ = 0.0;
  }

  ~MotionModelYawDifferentialStateUncertainty() {}

 private:
  double yaw_variance_;
};

class MotionModelYawDifferentialControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const;

  Eigen::Matrix3d dR(void) {
    return this->dR_;
  }

  void dR(Eigen::Matrix3d dR) {
    this->dR_ = dR;
  }

  Eigen::Matrix3d dR_error_log_cov(void) {
    return this->dR_error_log_cov_;
  }

  void dR_error_log_cov(Eigen::Matrix3d dR_error_log_cov) {
    this->dR_error_log_cov_ = dR_error_log_cov;
  }

  Eigen::Quaterniond q_sgs(void) {
    return this->q_sgs_.normalized();
  }

  void q_sgs(const Eigen::Quaterniond& q_sgs) {
    this->q_sgs_ = q_sgs.normalized();
  }

  MotionModelYawDifferentialControlInput(void) {
    this->dR_ = Eigen::Matrix3d::Identity();
    this->dR_error_log_cov_ = Eigen::Matrix3d::Zero();
    this->q_sgs_ = Eigen::Quaterniond::Identity();
  }

  ~MotionModelYawDifferentialControlInput() {}

 private:
  Eigen::Matrix3d dR_;
  Eigen::Matrix3d dR_error_log_cov_;
  Eigen::Quaterniond q_sgs_;
};

class MotionModelYawDifferential : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction = false);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);
  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);

  void JitterState(State* state_t);

  void Seed(int random_seed) {
    this->mvg_sampler_.Seed(random_seed);
    this->yaw_jitter_uvg_sampler_.Seed(random_seed);
  }

  void SetYawJitteringDistributionVariance(double yaw_jitter_variance) {
    this->yaw_jitter_uvg_sampler_.SetParams(0.0, yaw_jitter_variance);
  }

  MotionModelYawDifferential(void) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
    this->yaw_jitter_uvg_sampler_ = sampler::UnivariateGaussianSamplerStd(0.0, 1.0);
  }

  ~MotionModelYawDifferential() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
  sampler::UnivariateGaussianSamplerStd yaw_jitter_uvg_sampler_;
};

class MotionModelYawDifferentialUniformStateSampler : public StateSampler {
 public:
  void Init(double yaw_min, double yaw_max, Eigen::Quaterniond q_sgs) {
    this->yaw_sampler_.Init(yaw_min, yaw_max);
    this->q_sgs_ = q_sgs.normalized();
  }

  void Sample(State* state_sample);
  double CalculateStateProbabilityLog(State* state_sample);

  void Seed(int random_seed) {
    this->yaw_sampler_.Seed(random_seed);
  }

  double yaw_min(void) {
    return this->yaw_sampler_.min_value();
  }

  double yaw_max(void) {
    return this->yaw_sampler_.max_value();
  }

  Eigen::Quaterniond q_sgs(void) {
    return this->q_sgs_;
  }

  void q_sgs(const Eigen::Quaterniond& q_sgs) {
    this->q_sgs_ = q_sgs.normalized();
  }

  MotionModelYawDifferentialUniformStateSampler(void) {
    this->yaw_sampler_ = sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double>();
    this->q_sgs_ = Eigen::Quaterniond::Identity();
  }

  ~MotionModelYawDifferentialUniformStateSampler() {}

 private:
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double> yaw_sampler_;
  Eigen::Quaterniond q_sgs_;
};

class MotionModelYawDifferentialGaussianStateSampler : public StateSampler {
 public:
  void Init(double yaw_mean, double yaw_variance, Eigen::Quaterniond q_sgs) {
    this->yaw_mean_ = yaw_mean;
    this->yaw_variance_ = yaw_variance;
    this->univariate_gaussian_sampler_.Init(yaw_mean, yaw_variance);
    this->q_sgs_ = q_sgs.normalized();
  }

  void Sample(State* state_sample);
  double CalculateStateProbabilityLog(State* state_sample);

  void Seed(int random_seed) {
    this->univariate_gaussian_sampler_.Seed(random_seed);
  }

  double yaw_mean(void) {
    return this->yaw_mean_;
  }

  double yaw_variance(void) {
    return this->yaw_variance_;
  }

  Eigen::Quaterniond q_sgs(void) {
    return this->q_sgs_;
  }

  void q_sgs(const Eigen::Quaterniond& q_sgs) {
    this->q_sgs_ = q_sgs.normalized();
  }

  MotionModelYawDifferentialGaussianStateSampler(void) {
    this->univariate_gaussian_sampler_ = sampler::UnivariateGaussianSamplerStd();
    this->yaw_mean_ = 0.0;
    this->yaw_variance_ = 0.0;
    this->q_sgs_ = Eigen::Quaterniond::Identity();
  }

  ~MotionModelYawDifferentialGaussianStateSampler() {}

 private:
  sampler::UnivariateGaussianSamplerStd univariate_gaussian_sampler_;
  double yaw_mean_;
  double yaw_variance_;
  Eigen::Quaterniond q_sgs_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_YAW_DIFFERENTIAL_H_
