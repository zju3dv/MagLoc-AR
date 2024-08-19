/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-08-11 14:12:42
 * @LastEditTime: 2021-08-11 14:14:23
 * @LastEditors: xuehua
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_PARAMETER_MODEL_RANDOM_WALK_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_PARAMETER_MODEL_RANDOM_WALK_H_

#include <Eigen/Dense>

#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "distribution/gaussian_distribution.h"
#include "prediction_model/base.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_range_sampler.h"
#include "util/misc.h"

namespace state_estimation {

namespace prediction_model {

static const double kEpsilon = 1e-9;

class ParameterModelRandomWalkState : public State {
 public:
  void Init(Eigen::Matrix<double, Eigen::Dynamic, 1> parameters, std::vector<std::string> parameter_names = std::vector<std::string>()) {
    this->parameters_ = parameters;
    this->number_of_parameters_ = parameters.size();
    if (parameter_names.size() != parameters.size()) {
      this->has_names_ = false;
    } else {
      this->has_names_ = true;
      this->parameter_names_ = parameter_names;
    }
  }

  void Init(int number_of_parameters) {
    this->number_of_parameters_ = number_of_parameters;
    this->parameters_.setZero(number_of_parameters, 1);
    this->parameter_names_.clear();
    this->has_names_ = false;
  }

  void parameter_names(std::vector<std::string> parameter_names) {
    if ((parameter_names.size() != 0) && (parameter_names.size() != this->number_of_parameters_)) {
      std::cout << "ParameterModelRandomWalkState::parameter_names: the parameter_names size "
                << parameter_names.size()
                << " does not match the number of parameters "
                << this->number_of_parameters_
                << std::endl;
      return;
    }
    this->parameter_names_ = parameter_names;
    if (parameter_names.size() != 0) {
      this->has_names_ = true;
    } else {
      this->has_names_ = false;
    }
  }

  std::vector<std::string> parameter_names(void) const {
    return this->parameter_names_;
  }

  void parameters(Eigen::Matrix<double, Eigen::Dynamic, 1> parameters) {
    if (parameters.size() != this->number_of_parameters_) {
      std::cout << "ParameterModelRandomWalkState::parameters(parameters): Provided parameter size "
                << parameters.size()
                << " does not match the number of parameters "
                << this->parameter_names_.size() << ". "
                << "Parameters stay unchanged."
                << std::endl;
      return;
    }
    this->parameters_ = parameters;
  }

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters(void) const {
    return this->parameters_;
  }

  int number_of_parameters(void) const {
    assert(this->number_of_parameters_ == this->parameters_.size());
    return this->parameters_.size();
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    std::string parameter_name;
    for (int i = 0; i < this->parameters_.size(); i++) {
      if (this->has_names_) {
        parameter_name = this->parameter_names_.at(i);
      } else {
        parameter_name = "parameter_" + std::to_string(i);
      }
      named_values->emplace(named_values->end(), parameter_name, this->parameters_(i));
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

    Eigen::Matrix<double, Eigen::Dynamic, 1> mean_parameters;
    ParameterModelRandomWalkState* my_state_ptr;
    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<ParameterModelRandomWalkState*>(sample_state_ptrs.at(i));
      if (i == 0) {
        mean_parameters = my_state_ptr->parameters() * weights.at(i);
      } else {
        mean_parameters = mean_parameters + my_state_ptr->parameters() * weights.at(i);
      }
    }

    assert(this->number_of_parameters_ == mean_parameters.size());
    this->parameters_ = mean_parameters;
  }

  std::string ToKey(std::vector<double> discretization_resolution) {
    assert(discretization_resolution.size() == this->number_of_parameters_);
    std::vector<std::string> parameter_string_vector;
    double temp_parameter;
    for (int i = 0; i < this->parameters_.size(); i++) {
      temp_parameter = UniqueZeroRound(this->parameters_(i), discretization_resolution.at(i));
      parameter_string_vector.push_back(std::to_string(temp_parameter));
    }
    std::string key_string;
    JoinString(parameter_string_vector, "_", &key_string);
    return key_string;
  }

  void Add(State* state_ptr) {
    ParameterModelRandomWalkState* my_state_ptr = reinterpret_cast<ParameterModelRandomWalkState*>(state_ptr);
    Eigen::Matrix<double, Eigen::Dynamic, 1> parameters = my_state_ptr->parameters();
    if (this->parameters_.size() != parameters.size()) {
      std::cout << "ParameterModelRandomWalkState::Add: "
                << "Parameter numbers of the two operands does not match, which are "
                << this->parameters_.size() << " and "
                << parameters.size() << ". "
                << "The parameters stay unchanged."
                << std::endl;
      return;
    }
    this->parameters_ = this->parameters_ + parameters;
  }

  void Subtract(State* state_ptr) {
    ParameterModelRandomWalkState* my_state_ptr = reinterpret_cast<ParameterModelRandomWalkState*>(state_ptr);
    Eigen::Matrix<double, Eigen::Dynamic, 1> parameters = my_state_ptr->parameters();
    if (this->parameters_.size() != parameters.size()) {
      std::cout << "ParameterModelRandomWalkState::Subtract: "
                << "Parameter numbers of the two operands does not match, which are "
                << this->parameters_.size() << " and "
                << parameters.size() << ". "
                << "The parameters stay unchanged."
                << std::endl;
      return;
    }
    this->parameters_ = this->parameters_ - parameters;
  }

  void Multiply_scalar(double scalar_value) {
    this->parameters_ = this->parameters_ * scalar_value;
  }

  void Divide_scalar(double scalar_value) {
    this->parameters_ = this->parameters_ / scalar_value;
  }

  ParameterModelRandomWalkState(void) {
    std::vector<std::string> parameter_names;
    this->parameter_names_ = parameter_names;
    Eigen::Matrix<double, Eigen::Dynamic, 1> parameters;
    this->parameters_ = parameters;
    this->number_of_parameters_ = 0;
    this->has_names_ = false;
  }

  ~ParameterModelRandomWalkState() {
#ifdef DEBUG_FOCUSING
    std::cout << "ParameterModelRandomWalkState::~ParameterModelRandomWalkState" << std::endl;
#endif
  }

 private:
  std::vector<std::string> parameter_names_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_;
  int number_of_parameters_;
  bool has_names_;
};

class ParameterModelRandomWalkStateUncertainty : public StateUncertainty {
 public:
  void Init(int number_of_parameters) {
    this->number_of_parameters_ = number_of_parameters;
  }

  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    assert(sample_state_ptrs.size() == weights.size());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(sample_state_ptrs.size(), this->number_of_parameters_);
    ParameterModelRandomWalkState* my_state_ptr;
    Eigen::Matrix<double, Eigen::Dynamic, 1> parameters;

    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<ParameterModelRandomWalkState*>(sample_state_ptrs.at(i));
      parameters = my_state_ptr->parameters();
      assert(parameters.size() == this->number_of_parameters_);
      parameter_matrix.row(i) = parameters;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> weights_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(weights.size(), 1);
    for (int i = 0; i < weights.size(); i++) {
      weights_vector(i) = weights.at(i);
    }
    weights_vector = weights_vector.array() / weights_vector.sum();

    parameter_matrix = (parameter_matrix.array().rowwise() - parameter_matrix.colwise().mean().array()).colwise() * weights_vector.array().sqrt();

    this->parameter_covariance_ = parameter_matrix.transpose() * parameter_matrix;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance(void) {
    return this->parameter_covariance_;
  }

  int number_of_parameters(void) {
    return this->number_of_parameters_;
  }

  ParameterModelRandomWalkStateUncertainty(void) {
    this->parameter_covariance_ = Eigen::Matrix<double, 1, 1>::Zero(1, 1);
    this->number_of_parameters_ = 1;
  }

  ~ParameterModelRandomWalkStateUncertainty() {}

 private:
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance_;
  int number_of_parameters_;
};

class ParameterModelRandomWalkControlInput : public ControlInput {
 public:
  void Init(int number_of_parameters) {
    this->number_of_parameters_ = number_of_parameters;
    this->parameter_covariance_.setZero(number_of_parameters, number_of_parameters);
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    std::string value_name;
    for (int i = 0; i < this->parameter_covariance_.rows(); i++) {
      for (int j = 0; j < this->parameter_covariance_.cols(); j++) {
        value_name = "cov_" + std::to_string(i) + std::to_string(j);
        named_values->emplace(named_values->end(), value_name, this->parameter_covariance_(i, j));
      }
    }
  }

  void parameter_covariance(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance) {
    if (parameter_covariance.cols() != parameter_covariance.rows()) {
      std::cout << "ParameterModelRandomWalkControlInput::parameter_covariance: "
                << "The provided matrix is not square."
                << "The parameter_covariance stays unchanged." << std::endl;
      return;
    }
    if (parameter_covariance.cols() != this->number_of_parameters_) {
      std::cout << "ParameterModelRandomWalkControlInput::parameter_covariance: "
                << "The provided covariance matrix with size "
                << parameter_covariance.cols()
                << " doest not match the number_of_parameters "
                << this->number_of_parameters_ << ". "
                << "The parameter_covariance stays unchanged." << std::endl;
      return;
    }
    this->parameter_covariance_ = parameter_covariance;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance(void) {
    return this->parameter_covariance_;
  }

  int number_of_parameters(void) {
    return this->number_of_parameters_;
  }

  ParameterModelRandomWalkControlInput(void) {
    this->number_of_parameters_ = 0;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance;
    this->parameter_covariance_ = parameter_covariance;
  }

  ~ParameterModelRandomWalkControlInput() {}

 private:
  int number_of_parameters_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance_;
};

class ParameterModelRandomWalk : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction = false);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);
  void Seed(int random_seed);
  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);

  void JitterState(State* state_t);

  void SetParameterJitteringDistributionCovariance(Eigen::MatrixXd parameter_jitter_covariance) {
    int n_parameters = parameter_jitter_covariance.cols();
    this->parameter_jitter_mvg_sampler_.SetParams(Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n_parameters), parameter_jitter_covariance);
  }

  ParameterModelRandomWalk(void) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
    this->parameter_jitter_mvg_sampler_ = sampler::MultivariateGaussianSampler();
  }

  ~ParameterModelRandomWalk() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
  sampler::MultivariateGaussianSampler parameter_jitter_mvg_sampler_;
};

class ParameterModelRandomWalkGaussianStateSampler : public StateSampler {
 public:
  void Init(int number_of_parameters) {
    this->number_of_parameters_ = number_of_parameters;
    this->parameter_means_.setZero(number_of_parameters, 1);
    this->parameter_covariance_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(number_of_parameters, number_of_parameters);
  }

  void parameter_means(Eigen::Matrix<double, Eigen::Dynamic, 1> parameter_means) {
    if (parameter_means.size() != this->number_of_parameters_) {
      std::cout << "ParameterModelRandomWalkGaussianStateSampler::parameter_means: "
                << "The provided parameter_means size "
                << parameter_means.size()
                << " does not match the number_of_parameters "
                << this->number_of_parameters_ << ". "
                << " The parameters stays unchanged."
                << std::endl;
      return;
    }
    this->parameter_means_ = parameter_means;
  }

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameter_means(void) {
    return this->parameter_means_;
  }

  void parameter_covariance(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance) {
    if (parameter_covariance.cols() != parameter_covariance.rows()) {
      std::cout << "ParameterModelRandomWalkGaussianStateSampler::parameter_covariance: "
                << "The provided parameter_covariance is not square. "
                << "The parameters stays unchanged."
                << std::endl;
      return;
    }
    if (parameter_covariance.cols() != this->number_of_parameters_) {
      std::cout << "ParameterModelRandomWalkGaussianStateSampler::parameter_covariance: "
                << "The provided parameter_covariance size " << parameter_covariance.cols()
                << " does not match the number_of_parameters " << this->number_of_parameters_ << ". "
                << "The parameter_covariance stays unchanged." << std::endl;
      return;
    }
    this->parameter_covariance_ = parameter_covariance;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance(void) {
    return this->parameter_covariance_;
  }

  void Sample(State* state_sample);
  void Seed(int random_seed);

  double CalculateStateProbabilityLog(State* state_sample) {
    ParameterModelRandomWalkState* my_state_sample = reinterpret_cast<ParameterModelRandomWalkState*>(state_sample);
    Eigen::Matrix<double, Eigen::Dynamic, 1> parameters = my_state_sample->parameters();
    std::vector<double> parameter_vector;
    for (int i = 0; i < parameters.size(); i++) {
      parameter_vector.push_back(parameters(i));
    }
    std::vector<double> mean_vector;
    for (int i = 0; i < this->number_of_parameters_; i++) {
      mean_vector.push_back(this->parameter_means_(i));
    }
    std::vector<double> covariance_vector;
    for (int i = 0; i < this->number_of_parameters_; i++) {
      for (int j = i; j < this->number_of_parameters_; j++) {
        covariance_vector.push_back(this->parameter_covariance_(i, j));
      }
    }
    distribution::MultivariateGaussian mvg_distribution(mean_vector, covariance_vector);

    return std::log(mvg_distribution.QuantizedProbability(parameter_vector));
  }

  ParameterModelRandomWalkGaussianStateSampler(void) {
    this->number_of_parameters_ = 0;
    this->parameter_means_ = Eigen::Matrix<double, Eigen::Dynamic, 1>();
    this->parameter_covariance_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>();
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
  }

  ~ParameterModelRandomWalkGaussianStateSampler() {}

 private:
  int number_of_parameters_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> parameter_means_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance_;
  sampler::MultivariateGaussianSampler mvg_sampler_;
};

class ParameterModelRandomWalkUniformStateSampler : public StateSampler {
 public:
  void Init(std::vector<std::pair<double, double>> parameter_ranges) {
    this->parameter_ranges_ = parameter_ranges;
    this->number_of_parameters_ = parameter_ranges.size();
  }

  int number_of_parameters(void) {
    return this->number_of_parameters_;
  }

  void parameter_ranges(std::vector<std::pair<double, double>> parameter_ranges) {
    if (parameter_ranges.size() != this->number_of_parameters_) {
      std::cout << "ParameterModelRandomWalkUniformStateSampler::parameter_ranges: "
                << "The provided parameter_ranges size "
                << parameter_ranges.size()
                << " does not match the number_of_parameters "
                << this->number_of_parameters_ << ". "
                << "The parameter_ranges stays unchanged."
                << std::endl;
      return;
    }
    this->parameter_ranges_ = parameter_ranges;
  }

  std::vector<std::pair<double, double>> parameter_ranges(void) {
    return this->parameter_ranges_;
  }

  void Sample(State* state_sample);
  void Seed(int random_seed);

  double CalculateStateProbabilityLog(State* state_sample) {
    ParameterModelRandomWalkState* my_state_sample = reinterpret_cast<ParameterModelRandomWalkState*>(state_sample);
    Eigen::Matrix<double, Eigen::Dynamic, 1> parameters = my_state_sample->parameters();

    double log_prob = 0.0;
    for (int i = 0; i < this->number_of_parameters_; i++) {
      if ((parameters(i) >= this->parameter_ranges_.at(i).first - kEpsilon) && (parameters(i) <= this->parameter_ranges_.at(i).second + kEpsilon)) {
        if (this->parameter_ranges_.at(i).second - this->parameter_ranges_.at(i).first >= kEpsilon) {
          log_prob += std::log(1.0 / (this->parameter_ranges_.at(i).second - this->parameter_ranges_.at(i).first));
        } else {
          log_prob += std::log(1.0);
        }
      } else {
        log_prob += std::log(0.0);
      }
    }

    return log_prob;
  }

  ParameterModelRandomWalkUniformStateSampler(void) {
    this->number_of_parameters_ = 0;
    std::vector<std::pair<double, double>> parameter_ranges;
    this->parameter_ranges_ = parameter_ranges;
    this->parameter_range_sampler_ = sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double>();
  }

  ~ParameterModelRandomWalkUniformStateSampler() {}

 private:
  int number_of_parameters_;
  std::vector<std::pair<double, double>> parameter_ranges_;
  sampler::UniformRangeSampler<std::uniform_real_distribution<double>, double> parameter_range_sampler_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_PARAMETER_MODEL_RANDOM_WALK_H_
