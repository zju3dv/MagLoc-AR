/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-08-11 10:36:30
 * @LastEditTime: 2021-08-11 10:36:30
 * @LastEditors: xuehua
 */
#include "prediction_model/compound_prediction_model.h"

#include <cassert>

#include "prediction_model/base.h"

namespace state_estimation {

namespace prediction_model {

void CompoundPredictionModel::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModel::Predict" << std::endl;
#endif
  CompoundPredictionModelState* my_state_t = reinterpret_cast<CompoundPredictionModelState*>(state_t);
  CompoundPredictionModelState* my_state_tminus = reinterpret_cast<CompoundPredictionModelState*>(state_tminus);
  CompoundPredictionModelControlInput* my_control_input_t = reinterpret_cast<CompoundPredictionModelControlInput*>(control_input_t);

  int number_of_submodels = this->submodel_ptrs_.size();

  assert(number_of_submodels == my_state_t->GetNumberOfSubModelStates());
  assert(number_of_submodels == my_state_tminus->GetNumberOfSubModelStates());
  assert(number_of_submodels == my_control_input_t->GetNumberOfSubModelControlInputs());

  for (int i = 0; i < number_of_submodels; i++) {
    this->submodel_ptrs_.at(i)->Predict(my_state_t->at(i), my_state_tminus->at(i), my_control_input_t->at(i), dt, ideal_prediction);
  }

  double state_prediction_log_probability = 0.0;
  for (int i = 0; i < number_of_submodels; i++) {
    state_prediction_log_probability += my_state_t->at(i)->state_prediction_log_probability();
  }
  my_state_t->state_prediction_log_probability(state_prediction_log_probability);
  my_state_t->state_update_log_probability(my_state_tminus->state_update_log_probability());
}

void CompoundPredictionModel::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModel::PredictWithoutControlInput" << std::endl;
#endif
  CompoundPredictionModelState* my_state_t = reinterpret_cast<CompoundPredictionModelState*>(state_t);
  CompoundPredictionModelState* my_state_tminus = reinterpret_cast<CompoundPredictionModelState*>(state_tminus);

  int number_of_submodels = this->submodel_ptrs_.size();

  assert(number_of_submodels == my_state_t->GetNumberOfSubModelStates());
  assert(number_of_submodels == my_state_tminus->GetNumberOfSubModelStates());

  for (int i = 0; i < number_of_submodels; i++) {
    this->submodel_ptrs_.at(i)->PredictWithoutControlInput(my_state_t->at(i), my_state_tminus->at(i), dt);
  }
}

double CompoundPredictionModel::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  CompoundPredictionModelState* my_state_t = reinterpret_cast<CompoundPredictionModelState*>(state_t);
  CompoundPredictionModelState* my_state_tminus = reinterpret_cast<CompoundPredictionModelState*>(state_tminus);
  CompoundPredictionModelControlInput* my_control_input_t = reinterpret_cast<CompoundPredictionModelControlInput*>(control_input_t);

  int number_of_submodels = this->submodel_ptrs_.size();
  assert(number_of_submodels == my_state_t->GetNumberOfSubModelStates());
  assert(number_of_submodels == my_state_tminus->GetNumberOfSubModelStates());
  assert(number_of_submodels == my_control_input_t->GetNumberOfSubModelControlInputs());

  double transition_log_prob = 0.0;
  for (int i = 0; i < number_of_submodels; i++) {
    transition_log_prob += this->submodel_ptrs_.at(i)->CalculateStateTransitionProbabilityLog(my_state_t->at(i), my_state_tminus->at(i), my_control_input_t->at(i), dt);
  }
  return transition_log_prob;
}

void CompoundPredictionModelStateSampler::Sample(State* state_sample) {
#ifdef DEBUG_FOCUSING
  std::cout << "CompoundPredictionModelStateSampler::Sample" << std::endl;
#endif
  CompoundPredictionModelState* my_state = reinterpret_cast<CompoundPredictionModelState*>(state_sample);

  int number_of_submodel_state_samplers = this->submodel_state_sampler_ptrs_.size();

  assert(my_state->GetNumberOfSubModelStates() == number_of_submodel_state_samplers);

  double state_prediction_log_probability = 0.0;
  for (int i = 0; i < number_of_submodel_state_samplers; i++) {
    this->submodel_state_sampler_ptrs_.at(i)->Sample(my_state->at(i));
    state_prediction_log_probability += my_state->at(i)->state_prediction_log_probability();
  }

  my_state->state_prediction_log_probability(state_prediction_log_probability);
  my_state->state_update_log_probability(0.0);
}

void CompoundPredictionModelStateSampler::Seed(int random_seed) {
  for (int i = 0; i < this->submodel_state_sampler_ptrs_.size(); i++) {
    this->submodel_state_sampler_ptrs_.at(i)->Seed(random_seed);
  }
}

}  // namespace prediction_model

}  // namespace state_estimation
