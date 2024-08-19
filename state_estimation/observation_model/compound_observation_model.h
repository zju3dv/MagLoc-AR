/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-09-15 13:56:01
 * @LastEditTime: 2021-09-15 13:56:01
 * @LastEditors: xuehua
 */
#ifndef STATE_ESTIMATION_OBSERVATION_MODEL_COMPOUND_OBSERVATION_MODEL_H_
#define STATE_ESTIMATION_OBSERVATION_MODEL_COMPOUND_OBSERVATION_MODEL_H_

#include <assert.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "observation_model/base.h"
#include "prediction_model/base.h"
#include "util/misc.h"

namespace state_estimation {

namespace observation_model {

class CompoundObservation : public Observation {
 public:
  void Init(std::vector<Observation*> submodel_observation_ptrs) {
    this->submodel_observation_ptrs_ = submodel_observation_ptrs;
  }

  std::vector<std::pair<std::string, double>> GetFeatureVector(void) {
    std::vector<std::pair<std::string, double>> feature_vector;
    std::vector<std::pair<std::string, double>> submodel_feature_vector;
    for (int i = 0; i < this->submodel_observation_ptrs_.size(); i++) {
      submodel_feature_vector = this->submodel_observation_ptrs_.at(i)->GetFeatureVector();
      for (int j = 0; j < submodel_feature_vector.size(); j++) {
        feature_vector.push_back(submodel_feature_vector.at(j));
      }
    }
    return feature_vector;
  }

  int GetNumberOfSubModelObservations(void) {
    return this->submodel_observation_ptrs_.size();
  }

  Observation* at(int index) const {
    if (index >= this->submodel_observation_ptrs_.size()) {
      std::cout << "CompoundObservation::at: index "
                << index
                << " is out-of-range "
                << this->submodel_observation_ptrs_.size()
                << " returning a nullptr"
                << std::endl;
      return nullptr;
    }
    return this->submodel_observation_ptrs_.at(index);
  }

  CompoundObservation(void) {
    this->submodel_observation_ptrs_ = std::vector<Observation*>();
  }

  ~CompoundObservation() {}

 private:
  std::vector<Observation*> submodel_observation_ptrs_;
};

class CompoundObservationState : public State {
 public:
  void Init(std::vector<State*> submodel_observation_state_ptrs) {
    this->submodel_observation_state_ptrs_ = submodel_observation_state_ptrs;
  }

  void FromPredictionModelState(const prediction_model::State* prediction_model_state_ptr) {
    for (int i = 0; i < this->submodel_observation_state_ptrs_.size(); i++) {
      this->submodel_observation_state_ptrs_.at(i)->FromPredictionModelState(prediction_model_state_ptr);
    }
  }

  std::string ToKey(void) {
    std::vector<std::string> submodel_keys;
    for (int i = 0; i < this->submodel_observation_state_ptrs_.size(); i++) {
      submodel_keys.push_back(this->submodel_observation_state_ptrs_.at(i)->ToKey());
    }
    std::string compound_key;
    JoinString(submodel_keys, "_", &compound_key);
    return compound_key;
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) {
    std::vector<std::pair<std::string, double>> submodel_named_values;
    named_values->clear();
    for (int i = 0; i < this->submodel_observation_state_ptrs_.size(); i++) {
      this->submodel_observation_state_ptrs_.at(i)->GetAllNamedValues(&submodel_named_values);
      for (int j = 0; j < submodel_named_values.size(); j++) {
        named_values->push_back(submodel_named_values.at(j));
      }
    }
  }

  int GetNumberOfSubModelObservationStates(void) {
    return this->submodel_observation_state_ptrs_.size();
  }

  State* at(int index) const {
    if (index >= this->submodel_observation_state_ptrs_.size()) {
      std::cout << "CompoundObservationState::at: index "
                << index
                << " is out-of-range "
                << this->submodel_observation_state_ptrs_.size()
                << " returning a nullptr"
                << std::endl;
      return nullptr;
    }
    return this->submodel_observation_state_ptrs_.at(index);
  }

  CompoundObservationState(void) {
    this->submodel_observation_state_ptrs_ = std::vector<State*>();
  }

  ~CompoundObservationState() {}

 private:
  std::vector<State*> submodel_observation_state_ptrs_;
};

class CompoundObservationModel : public ObservationModel {
 public:
  void Init(std::vector<ObservationModel*> submodel_ptrs) {
    this->submodel_ptrs_ = submodel_ptrs;
  }

  double GetProbabilityObservationConditioningState(Observation* observation, State* state) const {
    CompoundObservation* my_observation = reinterpret_cast<CompoundObservation*>(observation);
    CompoundObservationState* my_state = reinterpret_cast<CompoundObservationState*>(state);

    assert(this->submodel_ptrs_.size() == my_observation->GetNumberOfSubModelObservations());
    assert(this->submodel_ptrs_.size() == my_state->GetNumberOfSubModelObservationStates());

    double p = 1.0;
    for (int i = 0; i < this->submodel_ptrs_.size(); i++) {
      p *= this->submodel_ptrs_.at(i)->GetProbabilityObservationConditioningState(my_observation->at(i), my_state->at(i));
    }

    return p;
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation, State* state) const {
    CompoundObservation* my_observation = reinterpret_cast<CompoundObservation*>(observation);
    CompoundObservationState* my_state = reinterpret_cast<CompoundObservationState*>(state);

    assert(this->submodel_ptrs_.size() == my_observation->GetNumberOfSubModelObservations());
    assert(this->submodel_ptrs_.size() == my_state->GetNumberOfSubModelObservationStates());

    double log_p = 0.0;
    for (int i = 0; i < this->submodel_ptrs_.size(); i++) {
      log_p += this->submodel_ptrs_.at(i)->GetProbabilityObservationConditioningStateLog(my_observation->at(i), my_state->at(i));
    }

    return log_p;
  }

  std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const {
    CompoundObservation* my_observation = reinterpret_cast<CompoundObservation*>(observation);

    assert(this->submodel_ptrs_.size() == my_observation->GetNumberOfSubModelObservations());

    std::vector<std::unordered_map<std::string, double>> submodel_state_maps;
    std::vector<std::vector<std::string>> submodel_state_keys;
    std::vector<std::string> state_keys;
    for (int i = 0; i < this->submodel_ptrs_.size(); i++) {
      submodel_state_maps.push_back(this->submodel_ptrs_.at(i)->GetProbabilityStatesConditioningObservation(my_observation->at(i)));
      state_keys.clear();
      for (auto it = submodel_state_maps.at(i).begin(); it != submodel_state_maps.at(i).end(); it++) {
        state_keys.push_back(it->first);
      }
      submodel_state_keys.push_back(state_keys);
    }

    std::vector<std::vector<std::string>> submodel_state_key_permutations;
    Permutate(&submodel_state_key_permutations, submodel_state_keys, 0);

    std::unordered_map<std::string, double> compound_state_map;
    for (int i = 0; i < submodel_state_key_permutations.size(); i++) {
      double p = 1.0;
      for (int j = 0; j < submodel_state_key_permutations.at(i).size(); j++) {
        p *= submodel_state_maps.at(j).at(submodel_state_key_permutations.at(i).at(j));
      }
      std::string compound_state_key;
      JoinString(submodel_state_key_permutations.at(i), "_", &compound_state_key);
      compound_state_map.insert(std::pair<std::string, double>(compound_state_key, p));
    }

    return compound_state_map;
  }

  CompoundObservationModel(void) {
    this->submodel_ptrs_ = std::vector<ObservationModel*>();
  }

  ~CompoundObservationModel() {}

 private:
  std::vector<ObservationModel*> submodel_ptrs_;
};

}  // namespace observation_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OBSERVATION_MODEL_COMPOUND_OBSERVATION_MODEL_H_
