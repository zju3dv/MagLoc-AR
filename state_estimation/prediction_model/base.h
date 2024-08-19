/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:02
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-02-07 15:19:49
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_BASE_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_BASE_H_

#include <cassert>
#include <string>
#include <utility>
#include <vector>
#include <mutex>

#include "util/misc.h"

namespace state_estimation {

namespace prediction_model {

class State {
 public:
  virtual void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const = 0;
  virtual std::string ToKey(std::vector<double> discretization_resolutions) = 0;
  virtual void EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) = 0;

  virtual void Add(State* state_ptr) = 0;
  virtual void Subtract(State* state_ptr) = 0;
  virtual void Multiply_scalar(double scalar_value) = 0;
  virtual void Divide_scalar(double scalar_value) = 0;

  virtual ~State() {}

  static void ValidateStatesAndWeights(std::vector<State*> &state_ptrs, std::vector<double> &weights) {
    assert(state_ptrs.size() == weights.size());
    NormalizeWeights(weights);
  }

  State(void) {
    this->state_prediction_log_probability_ = 0.0;
    this->state_update_log_probability_ = 0.0;
  }

  double state_prediction_log_probability(void) const {
    return this->state_prediction_log_probability_;
  }

  void state_prediction_log_probability(double state_prediction_log_probability) {
    this->state_prediction_log_probability_ = state_prediction_log_probability;
  }

  double state_update_log_probability(void) const {
    return this->state_update_log_probability_;
  }

  void state_update_log_probability(double state_update_log_probability) {
    this->state_update_log_probability_ = state_update_log_probability;
  }

  double state_log_probability(void) const {
    return this->state_prediction_log_probability_ + this->state_update_log_probability_;
  }

 protected:
  double state_prediction_log_probability_;
  double state_update_log_probability_;
};

class StateUncertainty {
 public:
  virtual void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) = 0;

  virtual ~StateUncertainty() {}
};

class ControlInput {
 public:
  virtual void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const = 0;

  virtual ~ControlInput() {}
};

class PredictionModel {
 public:
  // Predict and PredictWithoutControlInput should take the sign of dt into consideration.
  // The provided control_input assumes that dt is positive.
  virtual void Predict(State* state_t, State* state_tminus,
                       ControlInput* control_input_t,
                       double dt, bool ideal_prediction) = 0;
  virtual void PredictWithoutControlInput(State* state_t,
                                          State* state_tminus, double dt) = 0;
  virtual void Seed(int random_seed) = 0;

  virtual double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus,
                                                        ControlInput* control_input_t,
                                                        double dt) = 0;

  void JitterState(State* state_t) {
    return;
  }

  PredictionModel(void) {}

  PredictionModel(const PredictionModel &prediction_model) {}

  PredictionModel& operator=(const PredictionModel &prediction_model) {
    return *this;
  }

  virtual ~PredictionModel() {}

 protected:
  std::mutex my_mutex_;
};

class StateSampler {
 public:
  virtual void Sample(State* state_sample) = 0;
  virtual void Seed(int random_seed) = 0;
  virtual double CalculateStateProbabilityLog(State* state_sample) = 0;

  virtual void Traverse(State* state_sample) {}
  virtual void ResetTraverseState(void) {}
  virtual bool IsTraverseFinished(void) {
    return true;
  }

  virtual ~StateSampler() {}
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_BASE_H_
