/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-09-03 11:28:26
 * @LastEditTime: 2021-09-03 11:28:36
 * @LastEditors: xuehua
 */
#ifndef STATE_ESTIMATION_EVALUATION_FILTER_EVALUATION_H_
#define STATE_ESTIMATION_EVALUATION_FILTER_EVALUATION_H_

#include <iostream>
#include <vector>

namespace state_estimation {

namespace evaluation {

enum class EvaluationType {
  kGT = 0,
  kEst,
};

template <typename PredictionModelState,
          typename PredictionModelControlInput,
          typename PredictionModelStateSampler,
          typename PredictionModel,
          typename ObservationModelObservation,
          typename ObservationModelState,
          typename ObservationModel>
class FilterEvaluator {
 public:
  void Init(PredictionModelStateSampler state_sampler, PredictionModel prediction_model, ObservationModel observation_model) {
    this->prediction_model_state_sampler_ = state_sampler;
    this->prediction_model_ = prediction_model;
    this->observation_model_ = observation_model;
  }

  void SetData(std::vector<double> timestamps,
               std::vector<PredictionModelState> true_states,
               std::vector<PredictionModelState> est_states,
               std::vector<PredictionModelControlInput> control_inputs,
               std::vector<ObservationModelObservation> observations,
               std::vector<bool> need_updates = std::vector<bool>()) {
    if (need_updates.size() == 0) {
      for (int i = 0; i < timestamps.size(); i++) {
        need_updates.push_back(true);
      }
    }

    assert(timestamps.size() == true_states.size());
    assert(timestamps.size() == est_states.size());
    assert(timestamps.size() == control_inputs.size());
    assert(timestamps.size() == observations.size());
    assert(timestamps.size() == need_updates.size());

    this->timestamps_ = timestamps;
    this->true_states_ = true_states;
    this->est_states_ = est_states;
    this->control_inputs_ = control_inputs;
    this->observations_ = observations;
    this->need_updates_ = need_updates;
  }

  int SizeOfData(void) {
    return this->timetamps_.size();
  }

  double CalculateMeanError(void) {
#ifdef DEBUG_FOCUSING
    std::cout << "FilterEvaluator::CalculateMeanError" << std::endl;
#endif
    double mean_error = 0.0;
    double true_x, est_x, true_y, est_y;
    for (int i = 0; i < this->true_states_.size(); i++) {
      true_x = this->true_states_.at(i).position().x();
      est_x = this->est_states_.at(i).position().x();
      true_y = this->true_states_.at(i).position().y();
      est_y = this->est_states_.at(i).position().y();
      mean_error += std::pow(std::pow(true_x - est_x, 2.0) + std::pow(true_y - est_y, 2.0), 0.5) / this->true_states_.size();
    }
    return mean_error;
  }

  double CalculateOverallProbabilityLog(EvaluationType evaluation_type) {
#ifdef DEBUG_FOCUSING
    std::cout << "FilterEvaluator::CalculateOverallProbabilityLog" << std::endl;
#endif
    std::vector<PredictionModelState> temp_states;
    switch (evaluation_type) {
      case EvaluationType::kGT:
        temp_states = this->true_states_;
        break;
      case EvaluationType::kEst:
        temp_states = this->est_states_;
        break;
    }

    double log_likelihood = 0.0;
    for (int i = 0; i < temp_states.size(); i++) {
      if (i == 0) {
        log_likelihood += this->prediction_model_state_sampler_.CalculateStateProbabilityLog(&(temp_states.at(i)));
      } else {
        double dt = this->timestamps_.at(i) - this->timestamps_.at(i - 1);
        log_likelihood += this->prediction_model_.CalculateStateTransitionProbabilityLog(&(temp_states.at(i)), &(temp_states.at(i - 1)), &(this->control_inputs_.at(i)), dt);
      }

      if (this->need_updates_.at(i)) {
        ObservationModelState observation_model_state;
        observation_model_state.FromPredictionModelState(&(temp_states.at(i)));
        log_likelihood += this->observation_model_.GetProbabilityObservationConditioningStateLog(&(this->observations_.at(i)), &(observation_model_state));
      }
    }

    return log_likelihood;
  }

  double CalculatePredictionProbabilityLog(EvaluationType evaluation_type) {
#ifdef DEBUG_FOCUSING
    std::cout << "FilterEvaluator::CalculatePredictionProbabilityLog" << std::endl;
#endif
    std::vector<PredictionModelState> temp_states;
    switch (evaluation_type) {
      case EvaluationType::kGT:
        temp_states = this->true_states_;
        break;
      case EvaluationType::kEst:
        temp_states = this->est_states_;
        break;
    }

    double log_likelihood = 0.0;
    for (int i = 0; i < temp_states.size(); i++) {
      if (i == 0) {
        log_likelihood += this->prediction_model_state_sampler_.CalculateStateProbabilityLog(&(temp_states.at(i)));
      } else {
        double dt = this->timestamps_.at(i) - this->timestamps_.at(i - 1);
        log_likelihood += this->prediction_model_.CalculateStateTransitionProbabilityLog(&(temp_states.at(i)), &(temp_states.at(i - 1)), &(this->control_inputs_.at(i)), dt);
      }

#ifdef DEBUG_FILTER_EVALUATOR_CALCULATE_PREDICTION_PROBABILITY_LOG
      std::cout << "FilterEvaluator::CalculatePredictionProbabilityLog: step_log_likelihood: " << log_likelihood << std::endl;
#endif
    }

    return log_likelihood;
  }

  double CalculateObservationProbabilityLog(EvaluationType evaluation_type) {
#ifdef DEBUG_FOCUSING
    std::cout << "FilterEvaluator::CalculateObservationProbabilityLog" << std::endl;
#endif
    std::vector<PredictionModelState> temp_states;
    switch (evaluation_type) {
      case EvaluationType::kGT:
        temp_states = this->true_states_;
        break;
      case EvaluationType::kEst:
        temp_states = this->est_states_;
        break;
    }

    double log_likelihood = 0.0;
    for (int i = 0; i < temp_states.size(); i++) {
      if (this->need_updates_.at(i)) {
        ObservationModelState observation_model_state;
        observation_model_state.FromPredictionModelState(&(temp_states.at(i)));
        log_likelihood += this->observation_model_.GetProbabilityObservationConditioningStateLog(&(this->observations_.at(i)), &(observation_model_state));
      }
    }

    return log_likelihood;
  }

  std::vector<double> CalculateSequentialObservationProbabilityLog(EvaluationType evaluation_type) {
#ifdef DEBUG_FOCUSING
    std::cout << "FilterEvaluator::CalculateSequentialObservationProbabilityLog" << std::endl;
#endif
    std::vector<PredictionModelState> temp_states;
    switch (evaluation_type) {
      case EvaluationType::kGT:
        temp_states = this->true_states_;
        break;
      case EvaluationType::kEst:
        temp_states = this->est_states_;
        break;
    }

    std::vector<double> sequential_observation_log_probs;
    double log_likelihood = 0.0;
    for (int i = 0; i < temp_states.size(); i++) {
      if (this->need_updates_.at(i)) {
        ObservationModelState observation_model_state;
        observation_model_state.FromPredictionModelState(&(temp_states.at(i)));
        log_likelihood += this->observation_model_.GetProbabilityObservationConditioningStateLog(&(this->observations_.at(i)), &(observation_model_state));
      }
      sequential_observation_log_probs.push_back(log_likelihood);
    }

    return sequential_observation_log_probs;
  }

  void OutputPredictionProbabilityDetail(void) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(10);
    std::cout << "FilterEvaluator::OutputPredictionProbabilityDetail" << std::endl;
    std::cout << "  number_of_true_states: " << this->true_states_.size() << std::endl;
    std::cout << "  number_of_estimated_states: " << this->est_states_.size() << std::endl;
    for (int i = 0; i < this->true_states_.size(); i++) {
      std::cout << "    timestamp: " << this->timestamps_.at(i) << std::endl;
      if (i == 0) {
        std::vector<std::pair<std::string, double>> state_t_named_values;
        this->true_states_.at(i).GetAllNamedValues(&state_t_named_values);
        std::cout << "    state_t_state_gt:     ";
        for (int j = 0; j < state_t_named_values.size(); j++) {
          std::cout << " " << state_t_named_values.at(j).first << ": " << state_t_named_values.at(j).second;
        }
        std::cout << std::endl;
        this->est_states_.at(i).GetAllNamedValues(&state_t_named_values);
        std::cout << "    state_t_state_est:     ";
        for (int j = 0; j < state_t_named_values.size(); j++) {
          std::cout << " " << state_t_named_values.at(j).first << ": " << state_t_named_values.at(j).second;
        }
        std::cout << std::endl;
        std::vector<std::pair<std::string, double>> control_input_named_values;
        this->control_inputs_.at(i).GetAllNamedValues(&control_input_named_values);
        std::cout << "    control_input:     ";
        for (int j = 0; j < control_input_named_values.size(); j++) {
          std::cout << " " << control_input_named_values.at(j).first << ": " << control_input_named_values.at(j).second;
        }
        std::cout << std::endl;
        std::cout << "    state_transition_probability_log_gt: " << this->prediction_model_state_sampler_.CalculateStateProbabilityLog(&(this->true_states_.at(i))) << std::endl;
        std::cout << "    state_transition_probability_log_est: " << this->prediction_model_state_sampler_.CalculateStateProbabilityLog(&(this->est_states_.at(i))) << std::endl;
        continue;
      }
      std::vector<std::pair<std::string, double>> state_tminus_named_values;
      this->true_states_.at(i-1).GetAllNamedValues(&state_tminus_named_values);
      std::vector<std::pair<std::string, double>> state_t_named_values;
      this->true_states_.at(i).GetAllNamedValues(&state_t_named_values);
      std::cout << "    state_tminus_state_gt:";
      for (int j = 0; j < state_tminus_named_values.size(); j++) {
        std::cout << " " << state_tminus_named_values.at(j).first << ": " << state_tminus_named_values.at(j).second;
      }
      std::cout << std::endl;
      std::cout << "    state_t_state_gt:     ";
      for (int j = 0; j < state_t_named_values.size(); j++) {
        std::cout << " " << state_t_named_values.at(j).first << ": " << state_t_named_values.at(j).second;
      }
      std::cout << std::endl;
      this->est_states_.at(i-1).GetAllNamedValues(&state_tminus_named_values);
      this->est_states_.at(i).GetAllNamedValues(&state_t_named_values);
      std::cout << "    state_tminus_state_est:";
      for (int j = 0; j < state_tminus_named_values.size(); j++) {
        std::cout << " " << state_tminus_named_values.at(j).first << ": " << state_tminus_named_values.at(j).second;
      }
      std::cout << std::endl;
      std::cout << "    state_t_state_est:     ";
      for (int j = 0; j < state_t_named_values.size(); j++) {
        std::cout << " " << state_t_named_values.at(j).first << ": " << state_t_named_values.at(j).second;
      }
      std::cout << std::endl;
      std::vector<std::pair<std::string, double>> control_input_named_values;
      this->control_inputs_.at(i).GetAllNamedValues(&control_input_named_values);
      std::cout << "    control_input:     ";
      for (int j = 0; j < control_input_named_values.size(); j++) {
        std::cout << " " << control_input_named_values.at(j).first << ": " << control_input_named_values.at(j).second;
      }
      std::cout << std::endl;
      double dt = this->timestamps_.at(i) - this->timestamps_.at(i - 1);
      std::cout << "    state_transition_probability_log_gt: " << this->prediction_model_.CalculateStateTransitionProbabilityLog(&(this->true_states_.at(i)), &(this->true_states_.at(i - 1)), &(this->control_inputs_.at(i)), dt) << std::endl;
      std::cout << "    state_transition_probability_log_est: " << this->prediction_model_.CalculateStateTransitionProbabilityLog(&(this->est_states_.at(i)), &(this->est_states_.at(i - 1)), &(this->control_inputs_.at(i)), dt) << std::endl;
    }
  }

  void OutputObservationProbabilityDetail(void) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(10);
    std::cout << "FilterEvaluator::OutputObservationProbabilityDetail" << std::endl;
    std::cout << "  number_of_true_states: " << this->true_states_.size() << std::endl;
    std::cout << "  number_of_estimated_states: " << this->est_states_.size() << std::endl;
    for (int i = 0; i < this->true_states_.size(); i++) {
      std::cout << "    observation_prob_t:";
      ObservationModelState observation_model_state;
      if (this->need_updates_.at(i)) {
        observation_model_state.FromPredictionModelState(&(this->true_states_.at(i)));
        std::cout << " gt: " << this->observation_model_.GetProbabilityObservationConditioningStateLog(&(this->observations_.at(i)), &(observation_model_state));
        observation_model_state.FromPredictionModelState(&(this->est_states_.at(i)));
        std::cout << " est: " << this->observation_model_.GetProbabilityObservationConditioningStateLog(&(this->observations_.at(i)), &(observation_model_state));
      } else {
        std::cout << " gt: " << "0.0";
        std::cout << " est: " << "0.0";
      }
      std::cout << std::endl;
    }
  }

  std::vector<double> timestamps(void) {
    return this->timestmaps_;
  }

  std::vector<PredictionModelState> true_states(void) {
    return this->true_states_;
  }

  std::vector<PredictionModelState> est_states(void) {
    return this->est_states_;
  }

  std::vector<PredictionModelControlInput> control_inputs(void) {
    return this->control_inputs_;
  }

  std::vector<ObservationModelObservation> observations(void) {
    return this->observations_;
  }

  FilterEvaluator(void) {
    this->prediction_model_state_sampler_ = PredictionModelStateSampler();
    this->prediction_model_ = PredictionModel();
    this->observation_model_ = ObservationModel();
    this->timestamps_ = std::vector<double>();
    this->true_states_ = std::vector<PredictionModelState>();
    this->est_states_ = std::vector<PredictionModelState>();
    this->control_inputs_ = std::vector<PredictionModelControlInput>();
    this->observations_ = std::vector<ObservationModelObservation>();
    this->need_updates_ = std::vector<bool>();
  }

  ~FilterEvaluator() {}

 private:
  PredictionModel prediction_model_;
  PredictionModelStateSampler prediction_model_state_sampler_;
  ObservationModel observation_model_;
  std::vector<double> timestamps_;
  std::vector<PredictionModelState> true_states_;
  std::vector<PredictionModelState> est_states_;
  std::vector<PredictionModelControlInput> control_inputs_;
  std::vector<ObservationModelObservation> observations_;
  std::vector<bool> need_updates_;
};

}  // namespace evaluation

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_EVALUATION_FILTER_EVALUATION_H_
