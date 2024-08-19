/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:16:20
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-21 14:14:09
 */
#ifndef STATE_ESTIMATION_OBSERVATION_MODEL_BASE_H_
#define STATE_ESTIMATION_OBSERVATION_MODEL_BASE_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distribution/probability_mapper_base.h"
#include "prediction_model/base.h"

namespace state_estimation {

namespace observation_model {

class Observation {
 public:
  virtual std::vector<std::pair<std::string, double>> GetFeatureVector(void) = 0;
};

class State {
 public:
  virtual void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) = 0;
  virtual std::string ToKey(void) = 0;
  virtual void FromPredictionModelState(const prediction_model::State* prediction_model_state_ptr) = 0;
};

class ObservationModel {
 public:
  // return p(observation|state)
  virtual double GetProbabilityObservationConditioningState(
      Observation* observation, State* state) const = 0;

  virtual double GetProbabilityObservationConditioningStateLog(
      Observation* observation, State* state) const = 0;

  // return p(state_i|observation) for all possible i.
  virtual std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const = 0;

  void JitterState(State* state) {
    return;
  }
};

}  // namespace observation_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OBSERVATION_MODEL_BASE_H_
