/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-08-09 15:08:32
 * @LastEditTime: 2021-08-09 15:09:44
 * @LastEditors: xuehua
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_COMPOUND_PREDICTION_MODEL_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_COMPOUND_PREDICTION_MODEL_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace prediction_model {

class CompoundPredictionModelState : public State {
 public:
  void AddSubModelState(State* submodel_state_ptr) {
    // The submodel_state_ptr should point to memory space that will not be automatically recollected by the program,
    // if the deconstructor of this class will delete all the element in this->submodel_state_ptrs_.
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelState::AddSubModelState" << std::endl;
#endif
    // when adding submodel_state_ptr, the object should owned only by this CompoundPredictionModelState,
    // so the caller should release the object when using AddSubModelState.
    this->submodel_state_ptrs_.push_back(submodel_state_ptr);
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    std::vector<std::pair<std::string, double>> sub_named_values;
    for (int i = 0; i < this->submodel_state_ptrs_.size(); i++) {
      submodel_state_ptrs_.at(i)->GetAllNamedValues(&sub_named_values);
      for (int j = 0; j < sub_named_values.size(); j++) {
        named_values->push_back(sub_named_values.at(j));
      }
    }
  }

  void EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    std::vector<State*> submodel_sample_state_ptrs;
    CompoundPredictionModelState* my_state_ptr;
    for (int i = 0; i < this->submodel_state_ptrs_.size(); i++) {
      submodel_sample_state_ptrs.clear();
      for (int j = 0; j < sample_state_ptrs.size(); j++) {
        my_state_ptr = reinterpret_cast<CompoundPredictionModelState*>(sample_state_ptrs.at(j));
        submodel_sample_state_ptrs.push_back(my_state_ptr->at(i));
      }
      this->submodel_state_ptrs_.at(i)->EstimateFromSamples(submodel_sample_state_ptrs, weights);
    }
    NormalizeWeights(weights);
    double mean_state_prediction_log_probability;
    double mean_state_update_log_probability;
    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<CompoundPredictionModelState*>(sample_state_ptrs.at(i));
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
        mean_state_prediction_log_probability = weighted_state_prediction_log_probability;
        mean_state_update_log_probability = weighted_state_update_log_probability;
      } else {
        mean_state_prediction_log_probability += weighted_state_prediction_log_probability;
        mean_state_update_log_probability += weighted_state_update_log_probability;
      }
    }
    this->state_prediction_log_probability_ = mean_state_prediction_log_probability;
    this->state_update_log_probability_ = mean_state_update_log_probability;
  }

  State* at(int index) const {
    if (index >= this->submodel_state_ptrs_.size()) {
      std::cout << "CompoundPredictionModelState::at: index "
                << index
                << " is out-of-range "
                << this->submodel_state_ptrs_.size()
                << " returning a nullptr"
                << std::endl;
      return nullptr;
    }
    return this->submodel_state_ptrs_.at(index);
  }

  std::string ToKey(std::vector<double> discretization_resolutions) {
    assert(this->numbers_of_submodel_state_variables_.size() == this->submodel_state_ptrs_.size());
    int total_number_of_variables = 0;
    for (int i = 0; i < this->numbers_of_submodel_state_variables_.size(); i++) {
      total_number_of_variables += this->numbers_of_submodel_state_variables_.at(i);
    }
    assert(total_number_of_variables == discretization_resolutions.size());
    std::vector<std::string> submodel_keys;
    int variable_start_index = 0;
    std::vector<double>::const_iterator first, last;
    for (int i = 0; i < this->numbers_of_submodel_state_variables_.size(); i++) {
      first = discretization_resolutions.begin() + variable_start_index;
      last = discretization_resolutions.begin() + variable_start_index + this->numbers_of_submodel_state_variables_.at(i);
      submodel_keys.push_back(this->submodel_state_ptrs_.at(i)->ToKey(std::vector<double>(first, last)));
      variable_start_index += this->numbers_of_submodel_state_variables_.at(i);
    }
    std::string compound_key;
    JoinString(submodel_keys, "_", &compound_key);
    return compound_key;
  }

  int GetNumberOfSubModelStates(void) const {
    return this->submodel_state_ptrs_.size();
  }

  void Add(State* state_ptr) {
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelState::Add" << std::endl;
#endif
    CompoundPredictionModelState* my_state_ptr = reinterpret_cast<CompoundPredictionModelState*>(state_ptr);
    if (this->submodel_state_ptrs_.size() != my_state_ptr->GetNumberOfSubModelStates()) {
      std::cout << "CompoundPredictionModelState::Add: "
                << "Operation is invalid. "
                << "The operands have different numbers of submodel_states which are "
                << this->submodel_state_ptrs_.size() << " and "
                << my_state_ptr->GetNumberOfSubModelStates() << ". "
                << "The first operand stay unchanged."
                << std::endl;
      return;
    }
    for (int i = 0; i < this->submodel_state_ptrs_.size(); i++) {
      try {
        (this->at(i))->Add(my_state_ptr->at(i));
      } catch (std::bad_exception) {
        std::cout << "CompoundPredictionModelState::Add: "
                  << "Exception raised for the " << i << "/" << this->submodel_state_ptrs_.size() << "th "
                  << "sub-model-state Add()."
                  << std::endl;
      }
    }
  }

  void Subtract(State* state_ptr) {
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelState::Subtract" << std::endl;
#endif
    CompoundPredictionModelState* my_state_ptr = reinterpret_cast<CompoundPredictionModelState*>(state_ptr);
    if (this->submodel_state_ptrs_.size() != my_state_ptr->GetNumberOfSubModelStates()) {
      std::cout << "CompoundPredictionModelState::Subtract: "
                << "Operation is invalid. "
                << "The operands have different numbers of submodel_states which are "
                << this->submodel_state_ptrs_.size() << " and "
                << my_state_ptr->GetNumberOfSubModelStates() << ". "
                << "The first operand stay unchanged."
                << std::endl;
      return;
    }
    for (int i = 0; i < this->submodel_state_ptrs_.size(); i++) {
      try {
        (this->at(i))->Subtract(my_state_ptr->at(i));
      } catch (std::bad_exception) {
        std::cout << "CompoundPredictionModelState::Subtract: "
                  << "Exception raised for the " << i << "/" << this->submodel_state_ptrs_.size() << "th "
                  << "sub-model-state Subtract()."
                  << std::endl;
      }
    }
  }

  void Multiply_scalar(double scalar_value) {
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelState::Multiply_scalar" << std::endl;
#endif
    for (int i = 0; i < this->submodel_state_ptrs_.size(); i++) {
      try {
        (this->at(i))->Multiply_scalar(scalar_value);
      } catch (std::bad_exception) {
        std::cout << "CompoundPredictionModelState::Multiply_scalar: "
                  << "Exception raised for the " << i << "/" << this->submodel_state_ptrs_.size() << "th "
                  << "sub-model-state Multiply_scalar()."
                  << std::endl;
      }
    }
  }

  void Divide_scalar(double scalar_value) {
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelState::Divide_scalar" << std::endl;
#endif
    for (int i = 0; i < this->submodel_state_ptrs_.size(); i++) {
      try {
        (this->at(i))->Divide_scalar(scalar_value);
      } catch (std::bad_exception) {
        std::cout << "CompoundPredictionModelState::Divide_scalar: "
                  << "Exception raised for the " << i << "/" << this->submodel_state_ptrs_.size() << "th "
                  << "sub-model-state Divide_scalar()."
                  << std::endl;
      }
    }
  }

  std::vector<int> numbers_of_submodel_state_variables(void) const {
    return this->numbers_of_submodel_state_variables_;
  }

  void numbers_of_submodel_state_variables(std::vector<int> numbers_of_submodel_state_variables) {
    this->numbers_of_submodel_state_variables_ = numbers_of_submodel_state_variables;
  }

  // for anyone who want to use CompoundPredictionModelState, he/she should explicitly implement member functions below:
  variable::Position position(void);
  void position(variable::Position position);
  double yaw(void);
  void yaw(double yaw);
  Eigen::Quaterniond q_ws(void);
  void q_ws(Eigen::Quaterniond q_ws);
  double bluetooth_offset(void);
  void bluetooth_offset(double bluetooth_offset);
  double wifi_offset(void);
  void wifi_offset(double wifi_offset);
  Eigen::Vector3d geomagnetism_bias(void);
  void geomagnetism_bias(Eigen::Vector3d geomagnetism_bias);
  CompoundPredictionModelState& operator=(const CompoundPredictionModelState& compound_state);
  CompoundPredictionModelState(void);
  CompoundPredictionModelState(const CompoundPredictionModelState& compound_state);
  ~CompoundPredictionModelState();

 private:
  std::vector<State*> submodel_state_ptrs_;
  std::vector<int> numbers_of_submodel_state_variables_;
};

class CompoundPredictionModelStateUncertainty : public StateUncertainty {
 public:
  void Init(std::vector<StateUncertainty*> submodel_state_uncertainty_ptrs) {
    // Elements of the submodel_state_uncertainty_ptrs should point to memory space that will not be automatically recollected by the program,
    // if the deconstructor of this class will delete all the element in this->submodel_state_uncertainty_ptrs_.
    this->submodel_state_uncertainty_ptrs_ = submodel_state_uncertainty_ptrs;
  }

  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelStateUncertainty::CalculateFromSamples" << std::endl;
#endif
    std::vector<State*> submodel_sample_state_ptrs;
    CompoundPredictionModelState* my_state_ptr;
    for (int i = 0; i < this->submodel_state_uncertainty_ptrs_.size(); i++) {
      submodel_sample_state_ptrs.clear();
      for (int j = 0; j < sample_state_ptrs.size(); j++) {
        my_state_ptr = reinterpret_cast<CompoundPredictionModelState*>(sample_state_ptrs.at(j));
        submodel_sample_state_ptrs.push_back(my_state_ptr->at(i));
      }
      this->submodel_state_uncertainty_ptrs_.at(i)->CalculateFromSamples(submodel_sample_state_ptrs, weights);
    }
  }

  int NumberOfSubmodelStateUncertainties(void) {
    return this->submodel_state_uncertainty_ptrs_.size();
  }

  StateUncertainty* at(int index) const {
    if (index >= this->submodel_state_uncertainty_ptrs_.size()) {
      std::cout << "CompoundPredictionModelStateUncertainty::at: index "
                << index
                << " is out-of-range "
                << this->submodel_state_uncertainty_ptrs_.size()
                << " returning a nullptr"
                << std::endl;
      return nullptr;
    }
    return this->submodel_state_uncertainty_ptrs_.at(index);
  }

  // for anyone who want to use CompoundPredictionModelStateUncertainty, he/she should explicitly implement member functions below:
  CompoundPredictionModelStateUncertainty(void);
  CompoundPredictionModelStateUncertainty& operator=(const CompoundPredictionModelStateUncertainty& compound_state_uncertainty);
  CompoundPredictionModelStateUncertainty(const CompoundPredictionModelStateUncertainty& compound_state_uncertainty);
  ~CompoundPredictionModelStateUncertainty();

 private:
  std::vector<StateUncertainty*> submodel_state_uncertainty_ptrs_;
};

class CompoundPredictionModelControlInput : public ControlInput {
 public:
  void Init(std::vector<ControlInput*> submodel_control_input_ptrs) {
    // Elements of the submodel_control_input_ptrs should point to memory space that will not be automatically recollected by the program,
    // if the deconstructor of this class will delete all the element in this->submodel_control_input_ptrs_.
    // delete whatever has already be allocated.
    for (int i = 0; i < this->submodel_control_input_ptrs_.size(); i++) {
      delete this->submodel_control_input_ptrs_.at(i);
    }
    this->submodel_control_input_ptrs_ = submodel_control_input_ptrs;
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    std::vector<std::pair<std::string, double>> sub_named_values;
    for (int i = 0; i < this->submodel_control_input_ptrs_.size(); i++) {
      submodel_control_input_ptrs_.at(i)->GetAllNamedValues(&sub_named_values);
      for (int j = 0; j < sub_named_values.size(); j++) {
        named_values->push_back(sub_named_values.at(j));
      }
    }
  }

  void AddSubModelControlInput(ControlInput* submodel_control_input_ptr) {
    // The submodel_control_input_ptr should point to memory space that will not be automatically recollected by the program,
    // if the deconstructor of this class will delete all the element in this->submodel_control_input_ptrs_.
    this->submodel_control_input_ptrs_.push_back(submodel_control_input_ptr);
  }

  int GetNumberOfSubModelControlInputs(void) {
    return this->submodel_control_input_ptrs_.size();
  }

  ControlInput* at(int index) const {
    if (index >= this->submodel_control_input_ptrs_.size()) {
      std::cout << "CompoundPredictionModelControlInput::at: index "
                << index
                << " is out-of-range "
                << this->submodel_control_input_ptrs_.size()
                << " returning a nullptr"
                << std::endl;
      return nullptr;
    }
    return this->submodel_control_input_ptrs_.at(index);
  }

  // for anyone who want to use CompoundPredictionModelControlInput, he/she should explicitly implement member functions below:
  CompoundPredictionModelControlInput(void);
  CompoundPredictionModelControlInput& operator=(const CompoundPredictionModelControlInput& compound_control_input);
  CompoundPredictionModelControlInput(const CompoundPredictionModelControlInput& compound_control_input);
  ~CompoundPredictionModelControlInput();

 private:
  std::vector<ControlInput*> submodel_control_input_ptrs_;
};

class CompoundPredictionModel : public PredictionModel {
 public:
  void Init(std::vector<PredictionModel*> submodel_ptrs) {
    // Elements of the submodel_ptrs should point to memory space that will not be automatically recollected by the program,
    // if the deconstructor of this class will delete all the element in this->submodel_ptrs_.
    this->submodel_ptrs_ = submodel_ptrs;
  }

  void AddSubModel(PredictionModel* submodel_ptr) {
    // The submodel_ptr should point to memory space that will not be automatically recollected by the program,
    // if the deconstructor of this class will delete all the element in this->submodel_ptrs_.
    this->submodel_ptrs_.push_back(submodel_ptr);
  }

  int GetNumberOfSubModels(void) {
    return this->submodel_ptrs_.size();
  }

  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction = false);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);

  void Seed(int random_seed) {
    for (int i = 0; i < this->submodel_ptrs_.size(); i++) {
      this->submodel_ptrs_.at(i)->Seed(random_seed);
    }
  }

  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tmins,
                                                ControlInput* control_input_t,
                                                double dt);

  void JitterState(State* state_t);
  void JitterState(State* state_t, Eigen::Vector3d geomagnetism_s, Eigen::Vector3d gravity_s);

  CompoundPredictionModel(void) {
    std::vector<PredictionModel*> submodel_ptrs;
    this->submodel_ptrs_ = submodel_ptrs;
  }

  ~CompoundPredictionModel() {}

 private:
  std::vector<PredictionModel*> submodel_ptrs_;
};

class CompoundPredictionModelStateSampler : public StateSampler {
 public:
  void Init(std::vector<StateSampler*> submodel_state_sampler_ptrs) {
    // Elements of the submodel_state_sampler_ptrs should point to memory space that will not be automatically recollected by the program,
    // if the deconstructor of this class will delete all the element in this->submodel_state_sampler_ptrs_.
    this->submodel_state_sampler_ptrs_ = submodel_state_sampler_ptrs;
  }

  void AddSubModelStateSampler(StateSampler* submodel_state_sampler_ptr) {
    // The submodel_state_sampler_ptr should point to memory space that will not be automatically recollected by the program,
    // if the deconstructor of this class will delete all the element in this->submodel_state_sampler_ptrs_.
    this->submodel_state_sampler_ptrs_.push_back(submodel_state_sampler_ptr);
  }

  int GetNumberOfSubModelStateSamplers(void) {
    return this->submodel_state_sampler_ptrs_.size();
  }

  int submodel_state_sampler_ptrs(std::vector<StateSampler*> *submodel_state_sampler_ptrs) {
    submodel_state_sampler_ptrs->clear();
    for (int i = 0; i < this->submodel_state_sampler_ptrs_.size(); i++) {
      submodel_state_sampler_ptrs->push_back(this->submodel_state_sampler_ptrs_.at(i));
    }
    return this->submodel_state_sampler_ptrs_.size();
  }

  void Sample(State* state_sample);
  void Seed(int random_seed);

  double CalculateStateProbabilityLog(State* state_sample) {
    CompoundPredictionModelState* my_state_sample = reinterpret_cast<CompoundPredictionModelState*>(state_sample);
    assert(my_state_sample->GetNumberOfSubModelStates() == this->submodel_state_sampler_ptrs_.size());

    double log_prob = 0.0;
    for (int i = 0; i < this->submodel_state_sampler_ptrs_.size(); i++) {
      log_prob += this->submodel_state_sampler_ptrs_.at(i)->CalculateStateProbabilityLog(my_state_sample->at(i));
    }

    return log_prob;
  }

  CompoundPredictionModelStateSampler(void) {
    std::vector<StateSampler*> submodel_state_sampler_ptrs;
    this->submodel_state_sampler_ptrs_ = submodel_state_sampler_ptrs;
  }

  ~CompoundPredictionModelStateSampler() {}

 private:
  std::vector<StateSampler*> submodel_state_sampler_ptrs_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_COMPOUND_PREDICTION_MODEL_H_
