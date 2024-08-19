/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-08-11 14:12:55
 * @LastEditTime: 2021-08-11 14:12:55
 * @LastEditors: xuehua
 */
#include "prediction_model/parameter_model_random_walk.h"

#include <Eigen/Dense>

#include <cassert>
#include <mutex>

#include "distribution/gaussian_distribution.h"
#include "prediction_model/base.h"
#include "util/misc.h"

namespace state_estimation {

namespace prediction_model {

void ParameterModelRandomWalk::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction) {
  ParameterModelRandomWalkState* my_state_t = reinterpret_cast<ParameterModelRandomWalkState*>(state_t);
  ParameterModelRandomWalkState* my_state_tminus = reinterpret_cast<ParameterModelRandomWalkState*>(state_tminus);
  ParameterModelRandomWalkControlInput* my_control_input_t = reinterpret_cast<ParameterModelRandomWalkControlInput*>(control_input_t);

  assert(my_state_t->number_of_parameters() == my_state_tminus->number_of_parameters());
  assert(my_control_input_t->number_of_parameters() == my_state_tminus->number_of_parameters());

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_tminus = my_state_tminus->parameters();
  std::vector<std::string> parameter_names_tminus = my_state_tminus->parameter_names();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance_t = my_control_input_t->parameter_covariance();

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_t;
  if (ideal_prediction) {
    parameters_t = parameters_tminus;
  } else {
    std::lock_guard<std::mutex> guard(this->my_mutex_);
    this->mvg_sampler_.SetParams(parameters_tminus, parameter_covariance_t);
    parameters_t = this->mvg_sampler_.Sample();
  }

  my_state_t->parameters(parameters_t);
  my_state_t->parameter_names(parameter_names_tminus);
}

void ParameterModelRandomWalk::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  ParameterModelRandomWalkState* my_state_t = reinterpret_cast<ParameterModelRandomWalkState*>(state_t);
  ParameterModelRandomWalkState* my_state_tminus = reinterpret_cast<ParameterModelRandomWalkState*>(state_tminus);

  assert(my_state_t->number_of_parameters() == my_state_tminus->number_of_parameters());

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_tminus = my_state_tminus->parameters();
  std::vector<std::string> parameter_names_tminus = my_state_tminus->parameter_names();

  my_state_t->parameters(parameters_tminus);
  my_state_t->parameter_names(parameter_names_tminus);
}

void ParameterModelRandomWalk::Seed(int random_seed) {
  this->mvg_sampler_.Seed(random_seed);
  this->parameter_jitter_mvg_sampler_.Seed(random_seed);
}

double ParameterModelRandomWalk::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  ParameterModelRandomWalkState* my_state_t = reinterpret_cast<ParameterModelRandomWalkState*>(state_t);
  ParameterModelRandomWalkState* my_state_tminus = reinterpret_cast<ParameterModelRandomWalkState*>(state_tminus);
  ParameterModelRandomWalkControlInput* my_control_input_t = reinterpret_cast<ParameterModelRandomWalkControlInput*>(control_input_t);

  int number_of_parameters = my_state_t->number_of_parameters();

  assert(my_state_tminus->number_of_parameters() == number_of_parameters);
  assert(my_control_input_t->number_of_parameters() == number_of_parameters);

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_tminus = my_state_tminus->parameters();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameter_covariance_t = my_control_input_t->parameter_covariance();

  std::vector<double> parameter_means;
  for (int i = 0; i < number_of_parameters; i++) {
    parameter_means.push_back(parameters_tminus(i));
  }
  std::vector<double> parameter_covariance;
  for (int i = 0; i < number_of_parameters; i++) {
    for (int j = i; j < number_of_parameters; j++) {
      parameter_covariance.push_back(parameter_covariance_t(i, j));
    }
  }
  distribution::MultivariateGaussian mvg(parameter_means, parameter_covariance);

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_t = my_state_t->parameters();
  std::vector<double> x;
  for (int i = 0; i < number_of_parameters; i++) {
    x.push_back(parameters_t(i));
  }

  return std::log(mvg.QuantizedProbability(x));
}

void ParameterModelRandomWalk::JitterState(State* state_t) {
  ParameterModelRandomWalkState* my_state_t = reinterpret_cast<ParameterModelRandomWalkState*>(state_t);

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_t = my_state_t->parameters();
  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_jitter_delta = this->parameter_jitter_mvg_sampler_.Sample();
  assert(parameters_t.size() == parameters_jitter_delta.size());

  Eigen::Matrix<double, Eigen::Dynamic, 1> parameters_jitter_mean = this->parameter_jitter_mvg_sampler_.mean();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> parameters_jitter_covariance = this->parameter_jitter_mvg_sampler_.covariance();

  std::vector<double> parameters_jitter_mean_vector = EigenVector2Vector(parameters_jitter_mean);
  std::vector<double>  parameters_jitter_covariance_compact_vector = CovarianceMatrixToCompactVector(parameters_jitter_covariance);
  std::vector<double> parameters_jitter_delta_vector = EigenVector2Vector(parameters_jitter_delta);

  distribution::MultivariateGaussian mvg_parameters_jitter(parameters_jitter_mean_vector, parameters_jitter_covariance_compact_vector);

  my_state_t->parameters(parameters_t + parameters_jitter_delta);
  my_state_t->state_prediction_log_probability(my_state_t->state_prediction_log_probability() + std::log(mvg_parameters_jitter.QuantizedProbability(parameters_jitter_delta_vector)));
}

void ParameterModelRandomWalkGaussianStateSampler::Sample(State* state_sample) {
#ifdef DEBUG_FOCUSING
  std::cout << "ParameterModelRandomWalkGaussianStateSampler::Sample" << std::endl;
#endif
  ParameterModelRandomWalkState* my_state_sample = reinterpret_cast<ParameterModelRandomWalkState*>(state_sample);
  assert(this->number_of_parameters_ == my_state_sample->number_of_parameters());

  this->mvg_sampler_.SetParams(this->parameter_means_, this->parameter_covariance_);
  my_state_sample->parameters(this->mvg_sampler_.Sample());
}

void ParameterModelRandomWalkGaussianStateSampler::Seed(int random_seed) {
  this->mvg_sampler_.Seed(random_seed);
}

void ParameterModelRandomWalkUniformStateSampler::Sample(State* state_sample) {
#ifdef DEBUG_FOCUSING
  std::cout << "ParameterModelRandomWalkUniformStateSampler::Sample" << std::endl;
#endif
  ParameterModelRandomWalkState* my_state_sample = reinterpret_cast<ParameterModelRandomWalkState*>(state_sample);
  assert(this->number_of_parameters_ == my_state_sample->number_of_parameters());

  std::pair<double, double> parameter_range;
  Eigen::Matrix<double, Eigen::Dynamic, 1> sample_parameters;
  sample_parameters.setZero(this->number_of_parameters_, 1);
  for (int i = 0; i < this->number_of_parameters_; i++) {
    parameter_range = this->parameter_ranges_.at(i);
    this->parameter_range_sampler_.SetRange(parameter_range.first, parameter_range.second);
    sample_parameters(i) = this->parameter_range_sampler_.Sample();
  }
  my_state_sample->parameters(sample_parameters);
}

void ParameterModelRandomWalkUniformStateSampler::Seed(int random_seed) {
  this->parameter_range_sampler_.Seed(random_seed);
}

}  // namespace prediction_model

}  // namespace state_estimation
