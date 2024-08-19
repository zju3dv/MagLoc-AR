/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-01 21:21:36
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-20 17:21:25
 */
#include "distribution/probability_mapper_cov.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "distribution/gaussian_distribution.h"
#include "variable/position.h"

namespace state_estimation {

namespace distribution {

std::unordered_map<std::string, double> ProbabilityMapperCov::LookupMeans(std::string state_key) const {
  std::unordered_map<std::string, double> named_means;
  if ((this->state_keys_).find(state_key) != (this->state_keys_).end()) {
    const NamedMultivariateGaussian* feature_distribution = (*this->distribution_map_ptr_)[state_key];
    std::unordered_map<std::string, int> feature_key_to_index_dict = feature_distribution->variable_name_to_index_dict();
    for (auto it = feature_key_to_index_dict.begin(); it != feature_key_to_index_dict.end(); it++) {
      named_means.insert(std::pair<std::string, double>(it->first, feature_distribution->mean()(it->second)));
    }
  }
  return named_means;
}

std::unordered_map<std::string, double>
ProbabilityMapperCov::CalculateProbabilityStatesConditioningFeatureVector(
    std::vector<std::pair<std::string, double>> feature_vector,
    double dynamic_map_offset) const {
  std::unordered_map<std::string, double> state2probability_precise;
  double probability_feature_vector_conditioning_state;
  double probability_feature_vector = 0.0;

  std::string current_state_key;
  for (auto it = this->state_keys_.begin(); it != this->state_keys_.end(); it++) {
    current_state_key = *it;
    probability_feature_vector_conditioning_state =
        this->CalculateProbabilityFeatureVectorConditioningState(feature_vector, current_state_key);
    state2probability_precise.insert(
        std::pair<std::string, double>(current_state_key, probability_feature_vector_conditioning_state));
    probability_feature_vector += probability_feature_vector_conditioning_state;
  }

#ifdef DEBUG_PROBABILITY_MAPPER_MARGINAL_OBSERVATION_PROBABILITY
  if (probability_feature_vector <= 0) {
    std::cout << "DEBUG::ProbabilityMapCov::CalculateProbabilityPositionsConditioningFeatureVector: zero probability." << std::endl;
  }
#endif

  std::unordered_map<std::string, double> state2probability;
  if (probability_feature_vector > 0) {
    for (auto it = state2probability_precise.begin(); it != state2probability_precise.end(); it++) {
      state2probability.insert(std::pair<std::string, double>(it->first, static_cast<double>(it->second / probability_feature_vector)));
    }
  }

  return state2probability;
}

double ProbabilityMapperCov::CalculateProbabilityFeatureVectorConditioningState(
    std::vector<std::pair<std::string, double>> feature_vector,
    std::string state_key,
    double dynamic_map_offset) const {
  return std::exp(this->CalculateProbabilityFeatureVectorConditioningStateLog(feature_vector, state_key, dynamic_map_offset));
}

double ProbabilityMapperCov::CalculateProbabilityFeatureVectorConditioningStateLog(
    std::vector<std::pair<std::string, double>> feature_vector,
    std::string state_key,
    double dynamic_map_offset) const {
  double log_feature_vector_likelihood;
  if ((this->state_keys_).find(state_key) == (this->state_keys_).end()) {
    // invalid position.
    // to be done.
    log_feature_vector_likelihood = std::log(0.0);
#ifdef DEBUG_PROBABILITY_MAPPER_INVALID_STATE
    std::cout << "ProbabilityMapperCov::CalculateProbabilityFeatureVectorConditioningStateLog: invalid state, " << state_key << std::endl;
#endif
  } else if (feature_vector.size() == 0) {
    // without any information, uniform guess by default.
    log_feature_vector_likelihood = 0.0;
#ifdef DEBUG_PROBABILITY_MAPPER_EMPTY_FEATURE_VECTOR
    std::cout << "ProbabilityMapperCov::CalculateProbabilityFeatureVectorConditioningStateLog: empty feature vector." << std::endl;
#endif
  } else {
    log_feature_vector_likelihood = 0.0;
    std::string feature_key;
    double feature_value;
    const NamedMultivariateGaussian* feature_distribution = (*this->distribution_map_ptr_)[state_key];

    // pick corresponding feature_distributions
    // (features that contained in the map but omitted in the feature vector are not included in the probability calculation),
    // and count the number of valid features (i.e. included in the map).
    // the counting is separated from zero-centering.
    // valid feature: value != 0.0 and key in map.
    int number_of_valid_features = 0;
    std::unordered_map<std::string, int> feature_key_to_index_dict = feature_distribution->variable_name_to_index_dict();
    for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
      feature_key = feature_vector[feature_i].first;
      feature_value = feature_vector[feature_i].second;
      if ((feature_value != 0.0) && (feature_key_to_index_dict.find(feature_key) != feature_key_to_index_dict.end())) {
        number_of_valid_features += 1;
      }
    }

    // set the pre-defined offset value for map means.
    double value_offset_in_map = this->static_map_offset_ + dynamic_map_offset;

    // if required, do the zero-centering.
    // features with values 0.0 are recognized as fail-to-percept features
    // and they are omitted in the centering.
    // only those features with valid keys and non-zero values are included in the computation of the center-value.
    // when removing the center-value, only skip those zero-value features.
    // the reason to remove center-value for those invalid features is that
    // they are also included in the probability calculation.
    if (this->zero_centering_) {
      double sum_of_valid_features_in_vector = 0.0;
      double sum_of_valid_features_in_map = 0.0;
      for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
        feature_key = feature_vector[feature_i].first;
        feature_value = feature_vector[feature_i].second;
        if ((feature_value != 0.0) && (feature_key_to_index_dict.find(feature_key) != feature_key_to_index_dict.end())) {
          sum_of_valid_features_in_vector += feature_value;
          sum_of_valid_features_in_map += feature_distribution->mean()(feature_key_to_index_dict[feature_key]);
        }
      }
      double mean_of_valid_features_in_vector =
          sum_of_valid_features_in_vector / number_of_valid_features;
      value_offset_in_map =
          sum_of_valid_features_in_map / number_of_valid_features;

      // zero-centering the feature vector
      for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
        if (feature_value != 0.0) {
          // feature_vector[feature_i].second -= mean_of_valid_features_in_vector;
          feature_vector[feature_i].second -= (mean_of_valid_features_in_vector - value_offset_in_map);
        }
      }
    }

    // calculate probability using multivariate gaussian distribution
    std::vector<std::pair<std::string, double>> solid_feature_vector;
    for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
      feature_key = feature_vector[feature_i].first;
      feature_value = feature_vector[feature_i].second;
      if (feature_value == 0.0) {
        continue;
      }
      if (feature_key_to_index_dict.find(feature_key) != feature_key_to_index_dict.end()) {
        solid_feature_vector.push_back(std::pair<std::string, double>(feature_key, feature_value));
      } else {
        // the default feature_value_variance should be set to the average variance in the map. TBD
        UnivariateGaussian univariate_gaussian(0.0, 4.0);
        log_feature_vector_likelihood += std::log(univariate_gaussian.QuantizedProbability(feature_value, 1.0));
      }
    }
    log_feature_vector_likelihood += std::log(feature_distribution->QuantizedProbability(solid_feature_vector, 1.0));
  }

  if ((std::abs(this->probability_map_smooth_factor_) < 1e-200) && (std::abs(std::exp(log_feature_vector_likelihood)) < 1e-200)) {
    log_feature_vector_likelihood = 0.0;
  } else {
    log_feature_vector_likelihood *= this->probability_map_smooth_factor_;
  }

  return log_feature_vector_likelihood;
}

void ProbabilityMapperCov::OutputProbabilityStatesConditioningFeatureVector(
    std::vector<std::pair<std::string, double>> feature_vector,
    double dynamic_map_offset) const {
  std::unordered_map<std::string, double>
      state2probability_state_conditioning_feature_vector =
          this->CalculateProbabilityStatesConditioningFeatureVector(feature_vector, dynamic_map_offset);

  std::vector<std::string> state_keys;
  for (auto it = this->state_keys_.begin(); it != this->state_keys_.end(); it++) {
    state_keys.push_back(*it);
  }

  sort(state_keys.begin(), state_keys.end());

  char line[100];
  std::string temp_state_key;

  for (int i = 0; i < state_keys.size(); i++) {
    temp_state_key = state_keys[i];
    variable::Position temp_position;
    temp_position.FromKey(temp_state_key);
    snprintf(
        line,
        sizeof(line),
        "%f,%f,%.50f",
        temp_position.x(),
        temp_position.y(),
        state2probability_state_conditioning_feature_vector[temp_state_key]);
    std::cout << line << std::endl;
  }
}

int ProbabilityMapperCov::Terminate(void) {
  this->distribution_map_ptr_ = nullptr;
  return 0;
}

}  // namespace distribution

}  // namespace state_estimation
