/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:15:46
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-02-16 14:56:38
 */
#include "distribution/probability_mapper_2d.h"

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

const double kFeatureVectorLogLikelihoodMin = std::log(1e-300);

int ProbabilityMapper2D::GetValidFeatureStatistics(
    std::vector<std::pair<std::string, double>> feature_vector,
    std::string state_key,
    double* valid_feature_map_mean,
    double* valid_feature_client_mean) const {
  // valid feature: features that both have non-zero values in the feature_vector and included in the distribution_map[state_key].
  int number_of_valid_features = 0;
  *valid_feature_map_mean = 0.0;
  *valid_feature_client_mean = 0.0;
  if (this->state_keys_.find(state_key) != this->state_keys_.end()) {
    const std::unordered_map<std::string, std::vector<double>>* feature_distribution = (*this->distribution_map_ptr_)[state_key];
    std::string feature_key;
    double feature_value;
    for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
      feature_key = feature_vector[feature_i].first;
      feature_value = feature_vector[feature_i].second;
      if ((feature_value != 0.0) && (feature_distribution->find(feature_key) != feature_distribution->end())) {
        number_of_valid_features += 1;
        *valid_feature_map_mean += feature_distribution->at(feature_key)[0];
        *valid_feature_client_mean += feature_value;
      }
    }
    *valid_feature_map_mean /= number_of_valid_features;
    *valid_feature_client_mean /= number_of_valid_features;
  }
  return number_of_valid_features;
}

int ProbabilityMapper2D::ShiftFeatureVector(std::vector<std::pair<std::string, double>>* feature_vector, double shift_value) const {
  int num_shifted_features = 0;
  for (int i = 0; i < feature_vector->size(); i++) {
    if (feature_vector->at(i).second != 0.0) {
      feature_vector->at(i).second += shift_value;
      num_shifted_features += 1;
    }
  }
  return num_shifted_features;
}

std::unordered_map<std::string, double> ProbabilityMapper2D::LookupMeans(std::string state_key) const {
  std::unordered_map<std::string, double> named_means;
  if ((this->state_keys_).find(state_key) != (this->state_keys_).end()) {
    const std::unordered_map<std::string, std::vector<double>>*
        feature_distribution = (*this->distribution_map_ptr_)[state_key];
    for (auto it = feature_distribution->begin(); it != feature_distribution->end(); it++) {
      named_means.insert(std::pair<std::string, double>(it->first, it->second.at(0)));
    }
  }
  return named_means;
}

std::unordered_map<std::string, double>
ProbabilityMapper2D::CalculateProbabilityStatesConditioningFeatureVector(
    std::vector<std::pair<std::string, double>> feature_vector,
    double dynamic_map_offset) const {
  int log_probability_lifting_threshold = 6;
  bool need_normalization = false;
  std::unordered_map<std::string, double> state2probability_log;
  double log_probability_feature_vector_conditioning_state;
  double max_log_probability_feature_vector_conditioning_state = 1.0;
  double epsilon = 1e-8;
  std::string current_state_key;
  for (auto it = this->state_keys_.begin();
       it != this->state_keys_.end(); it++) {
    current_state_key = *it;
    log_probability_feature_vector_conditioning_state =
        this->CalculateProbabilityFeatureVectorConditioningStateLog(
            feature_vector, current_state_key, dynamic_map_offset);
    if ((max_log_probability_feature_vector_conditioning_state > epsilon) ||
        (log_probability_feature_vector_conditioning_state > max_log_probability_feature_vector_conditioning_state)) {
      max_log_probability_feature_vector_conditioning_state = log_probability_feature_vector_conditioning_state;
    }
    state2probability_log.insert(
        std::pair<std::string, double>(
            current_state_key,
            log_probability_feature_vector_conditioning_state));
  }

  double probability_feature_vector = 0.0;
  for (auto it = state2probability_log.begin(); it != state2probability_log.end(); it++) {
    if (feature_vector.size() > log_probability_lifting_threshold) {
      it->second -= max_log_probability_feature_vector_conditioning_state;
    }
    probability_feature_vector += std::exp(it->second);
  }

  std::unordered_map<std::string, double> state2probability;
  if (probability_feature_vector > 0.0) {
    for (auto it = state2probability_log.begin(); it != state2probability_log.end(); it++) {
      if (need_normalization) {
        state2probability.insert(std::pair<std::string, double>(it->first, std::exp(it->second) / probability_feature_vector));
      } else {
        state2probability.insert(std::pair<std::string, double>(it->first, std::exp(it->second)));
      }
    }
  }

#ifdef DEBUG_PROBABILITY_MAPPER_MARGINAL_OBSERVATION_PROBABILITY
  if (probability_feature_vector <= 0.0) {
    std::cout << "DEBUG::ProbabilityMap2D::CalculateProbabilityPositionsConditioningFeatureVector: zero probability." << std::endl;
  }
#endif

  return state2probability;
}

double ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningState(
    std::vector<std::pair<std::string, double>> feature_vector,
    std::string state_key,
    double dynamic_map_offset) const {
  return std::exp(this->CalculateProbabilityFeatureVectorConditioningStateLog(feature_vector, state_key, dynamic_map_offset));
}

double ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningStateLog(
    std::vector<std::pair<std::string, double>> feature_vector,
    std::string state_key,
    double dynamic_map_offset) const {
    int mapping_type = 0;
#ifdef DEBUG_FOCUSING
  std::cout << "ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningStateLog" << std::endl;
#endif
  double log_feature_vector_likelihood;
  if ((this->state_keys_).find(state_key) == (this->state_keys_).end()) {
    // invalid position.
    log_feature_vector_likelihood = std::log(0.0);
#ifdef DEBUG_PROBABILITY_MAPPER_INVALID_STATE
    std::cout << "ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningStateLog: invalid state, " << state_key << std::endl;
#endif
  } else if (mapping_type == 1) {
    // get valid feature statistics.
    // valid_feature: feature_value in the feature_vector is larger than 0.0 and the feature_key is in the distribution_map entry of state_key.
    double valid_feature_map_mean = 0.0;
    double valid_feature_client_mean = 0.0;
    int number_of_valid_features = this->GetValidFeatureStatistics(feature_vector, state_key, &valid_feature_map_mean, &valid_feature_client_mean);

    // user can manually set the map_offset using the this->map_offset_ attribute.
    double value_offset_in_map = this->static_map_offset_ + dynamic_map_offset;

    // if zero-centering, the offset value is automatically set according to both map values and feature_vector values.
    // if (this->zero_centering_) {
    if (this->zero_centering_ && (number_of_valid_features > this->threshold_of_effective_feature_number_for_zero_centering_)) {
      value_offset_in_map = valid_feature_map_mean - valid_feature_client_mean;
    }

    if (value_offset_in_map != 0.0) {
      this->ShiftFeatureVector(&feature_vector, value_offset_in_map);
    }

    log_feature_vector_likelihood = 0.0;
    std::unordered_map<std::string, double> feature_vector_map;
    for (int i = 0; i < feature_vector.size(); i++) {
      feature_vector_map.insert(std::pair<std::string, double>(feature_vector.at(i).first, feature_vector.at(i).second));
    }
    const std::unordered_map<std::string, std::vector<double>>* feature_distribution = (*this->distribution_map_ptr_)[state_key];
    for (auto map_item = feature_distribution->begin(); map_item != feature_distribution->end(); map_item++) {
      double feature_likelihood = 0.0;
      assert(map_item->second.size() == 2);
      UnivariateGaussian univariate_gaussian(map_item->second[0], (map_item->second[1] * this->probability_map_smooth_factor_)  * (map_item->second[1] * this->probability_map_smooth_factor_));
      if (feature_vector_map.find(map_item->first) != feature_vector_map.end()) {
        feature_likelihood = univariate_gaussian.QuantizedProbability(feature_vector_map.at(map_item->first), 1.0);
      } else {
        feature_likelihood = univariate_gaussian.QuantizedProbability(0.0, 1.0);
      }
      log_feature_vector_likelihood += std::log(feature_likelihood);
    }
  } else {
    if (feature_vector.size() == 0) {
      // without any information, uniform guess by default.
      log_feature_vector_likelihood = 0.0;
#ifdef DEBUG_PROBABILITY_MAPPER_EMPTY_FEATURE_VECTOR
      std::cout << "ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningStateLog: empty feature vector." << std::endl;
#endif
    } else {
      // get valid feature statistics.
      // valid_feature: feature_value in the feature_vector is larger than 0.0 and the feature_key is in the distribution_map entry of state_key.
      double valid_feature_map_mean = 0.0;
      double valid_feature_client_mean = 0.0;
      int number_of_valid_features = this->GetValidFeatureStatistics(feature_vector, state_key, &valid_feature_map_mean, &valid_feature_client_mean);

      // user can manually set the map_offset using the this->map_offset_ attribute.
      double value_offset_in_map = this->static_map_offset_ + dynamic_map_offset;

      // if zero-centering, the offset value is automatically set according to both map values and feature_vector values.
      // if (this->zero_centering_) {
      if (this->zero_centering_ && (number_of_valid_features > this->threshold_of_effective_feature_number_for_zero_centering_)) {
        value_offset_in_map = valid_feature_map_mean - valid_feature_client_mean;
      }

      if (value_offset_in_map != 0.0) {
        this->ShiftFeatureVector(&feature_vector, value_offset_in_map);
      }

      // calculate probability in a feature-wise manner.
      const std::unordered_map<std::string, std::vector<double>>*
          feature_distribution = (*this->distribution_map_ptr_)[state_key];
      log_feature_vector_likelihood = 0.0;
      for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
        std::string feature_key = feature_vector[feature_i].first;
        double feature_value = feature_vector[feature_i].second;

        double feature_likelihood;
        if ((std::abs(feature_value) <= 1e-30) || (feature_distribution->find(feature_key) == feature_distribution->end())) {
        // if ((feature_value == 0.0) || (feature_distribution->find(feature_key) == feature_distribution->end())) {
        // if (feature_value == 0.0) {
          // no valid value received for the current feature.
          // uniformly distributed over all positions in map.
          feature_likelihood = 1.0;
        } else {
          double feature_value_mean;
          double feature_value_std;
          // if (feature_distribution->find(feature_key) == feature_distribution->end()) {
          //   // no probability model for this required feature in the current position.
          //   // there is no valid sample for this feature existing at the current position in the map.
          //   // data needs to be normalized.
          //   feature_value_mean = valid_feature_map_mean - 40.0;
          //   // the default feature_value_std should be set to the average std in map. TBD
          //   feature_value_std = 4.0;
          // } else {
          //   assert(feature_distribution->at(feature_key).size() == 2);
          //   feature_value_mean = feature_distribution->at(feature_key)[0];
          //   feature_value_std = feature_distribution->at(feature_key)[1];
          // }
          assert(feature_distribution->at(feature_key).size() == 2);
          feature_value_mean = feature_distribution->at(feature_key)[0];
          feature_value_std = feature_distribution->at(feature_key)[1] * this->probability_map_smooth_factor_;
          UnivariateGaussian univariate_gaussian(feature_value_mean, feature_value_std * feature_value_std);
          feature_likelihood = univariate_gaussian.QuantizedProbability(feature_value, 1.0);
        }
        log_feature_vector_likelihood += std::log(feature_likelihood);
#ifdef DEBUG_PROBABILITY_MAPPER_LOG_FEATURE_VECTOR_LIKELIHOOD
        std::cout.precision(50);
        std::cout << "ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningState: log_feature_vector_likelihood: " << log_feature_vector_likelihood << std::endl;
#endif
      }
#ifdef DEBUG_PROBABILITY_MAPPER_FEATURE_VECTOR_LIKELIHOOD
      std::cout.precision(50);
      std::cout << "ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningState: log_feature_vector_likelihood: " << log_feature_vector_likelihood << std::endl;
      std::cout << "ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningState: feature_vector_likelihood: " << std::exp(log_feature_vector_likelihood) << std::endl;
#endif
    }
  }

  return log_feature_vector_likelihood;
}

double ProbabilityMapper2D::CalculateNEESFeatureVectorConditioningState(
    std::vector<std::pair<std::string, double>> feature_vector,
    std::string state_key,
    int* num_features,
    double* error_mean,
    double* std_mean) const {
  double feature_vector_nees = 0.0;
  double feature_vector_error = 0.0;
  double feature_vector_std = 0.0;
  if ((this->state_keys_).find(state_key) == (this->state_keys_).end()) {
    // invalid position.
    // to be done.
    feature_vector_nees = 1000.0;
#ifdef DEBUG
    std::cout << "invalid position!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "position_key: " << position.ToKey() << std::endl;
#endif
    *num_features = 0;
    *error_mean = 0;
    *std_mean = 0;
  } else {
    std::string feature_key;
    double feature_value;
    const std::unordered_map<std::string, std::vector<double>>*
        feature_distribution = (*this->distribution_map_ptr_)[state_key];

    std::vector<std::unordered_map<std::string,
                                   std::vector<double>>::const_iterator>
        feature_key_its;

    int number_of_valid_features = 0;
    for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
      feature_key = feature_vector[feature_i].first;
      feature_value = feature_vector[feature_i].second;
      std::unordered_map<std::string, std::vector<double>>::const_iterator
          feature_key_it = feature_distribution->find(feature_key);
      feature_key_its.push_back(feature_key_it);
      if ((feature_value != 0.0) && (feature_key_it != feature_distribution->end())) {
        number_of_valid_features += 1;
      }
    }

    double value_offset_in_map = this->static_map_offset_;

    if (this->zero_centering_) {
      double sum_of_valid_features_in_vector = 0.0;
      double sum_of_valid_features_in_map = 0.0;
      for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
        feature_key = feature_vector[feature_i].first;
        feature_value = feature_vector[feature_i].second;
        if ((feature_value != 0.0) && (feature_key_its[feature_i] != feature_distribution->end())) {
          sum_of_valid_features_in_vector += feature_value;
          sum_of_valid_features_in_map += feature_key_its[feature_i]->second[0];
        }
      }
      double mean_of_valid_features_in_vector =
          sum_of_valid_features_in_vector / number_of_valid_features;
      value_offset_in_map =
          sum_of_valid_features_in_map / number_of_valid_features;

      for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
        if (feature_value != 0.0) {
          // feature_vector[feature_i].second -= mean_of_valid_features_in_vector;
          feature_vector[feature_i].second -= (mean_of_valid_features_in_vector - value_offset_in_map);
        }
      }
    }

    // calculate probability in a feature-wise manner.
    int n_features = 0;
    for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
#ifdef DEBUG
      // std::cout << "DEBUG::ProbabilityMap2D::CalculateProbabilityFeatureVectorConditioningPosition: fingerprint likelihood: " << likelihood << std::endl;
#endif
      feature_key = feature_vector[feature_i].first;
      feature_value = feature_vector[feature_i].second;

      double feature_nees;
      double feature_error;
      double feature_std;
      if (feature_value == 0.0) {
        // no valid value received for the current feature.
        // uniformly distributed over all positions in map.
        feature_nees = 0.0;
        feature_error = 0.0;
        feature_std = 0.0;
      } else {
        double feature_value_mean;
        double feature_value_std;
        if (feature_key_its[feature_i] == feature_distribution->end()) {
          // no probability model for this required feature in the current position.
          // there is no valid sample for this feature existing at the current position in the map.
          // data needs to be normalized.
          feature_value_mean = 0.0;
          // the default feature_value_std should be set to the average std in map. TBD
          feature_value_std = 2.0;
          continue;
        } else {
          // feature_value_mean =
          //     feature_key_its[feature_i]->second[0] - value_offset_in_map;
          feature_value_mean = feature_key_its[feature_i]->second[0];
          feature_value_std = feature_key_its[feature_i]->second[1];
        }
        n_features += 1;
        feature_nees = std::pow((feature_value - feature_value_mean), 2.0) / (std::pow(feature_value_std, 2.0));
        feature_error = std::abs(feature_value - feature_value_mean);
        feature_std = feature_value_std;
        // std::cout.precision(5);
        // std::cout << "feature_nees: " << feature_nees << std::endl;
      }
      feature_vector_nees += feature_nees;
      feature_vector_error += feature_error;
      feature_vector_std += feature_std;
    }

    assert(n_features == number_of_valid_features);
    feature_vector_nees /= n_features;
    feature_vector_error /= n_features;
    feature_vector_std /= n_features;
    *num_features = n_features;
    *error_mean = feature_vector_error;
    *std_mean = feature_vector_std;
  }

  return feature_vector_nees;
}

std::vector<std::pair<std::string, double>> ProbabilityMapper2D::CalculateProbabilityFeatureVectorConditioningStateSeparated(
    std::vector<std::pair<std::string, double>> feature_vector,
    std::string state_key) const {
  std::string feature_key;
  double feature_value;
  double feature_value_mean;
  double feature_value_std;
  double feature_likelihood;
  std::vector<std::pair<std::string, double>> named_feature_likelihoods;

  if ((this->state_keys_).find(state_key) == (this->state_keys_).end()) {
    // invalid position.
    // to be done.
    for (int i = 0; i < feature_vector.size(); i++) {
      named_feature_likelihoods.push_back(std::pair<std::string, double>(feature_vector.at(i).first, 0.0));
    }
#ifdef DEBUG
    std::cout << "invalid position!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "position_key: " << position.ToKey() << std::endl;
#endif
  } else {
    const std::unordered_map<std::string, std::vector<double>>*
        feature_distribution = (*this->distribution_map_ptr_)[state_key];

    std::vector<std::unordered_map<std::string,
                                   std::vector<double>>::const_iterator>
        feature_key_its;

    int number_of_valid_features = 0;
    for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
      feature_key = feature_vector[feature_i].first;
      feature_value = feature_vector[feature_i].second;
      std::unordered_map<std::string, std::vector<double>>::const_iterator
          feature_key_it = feature_distribution->find(feature_key);
      feature_key_its.push_back(feature_key_it);
      if (feature_value == 0.0) {
        continue;
      }
      if (feature_key_it != feature_distribution->end()) {
        number_of_valid_features += 1;
      }
    }

    double value_offset_in_map = this->static_map_offset_;

    if (this->zero_centering_) {
      double sum_of_valid_features_in_vector = 0.0;
      double mean_of_valid_features_in_vector = 0.0;
      double sum_of_valid_features_in_map = 0.0;
      for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
        feature_key = feature_vector[feature_i].first;
        feature_value = feature_vector[feature_i].second;
        if (feature_value == 0.0) {
          continue;
        }
        if (feature_key_its[feature_i] != feature_distribution->end()) {
          sum_of_valid_features_in_vector += feature_value;
          sum_of_valid_features_in_map += feature_key_its[feature_i]->second[0];
        }
      }
      mean_of_valid_features_in_vector =
          sum_of_valid_features_in_vector / number_of_valid_features;
      value_offset_in_map =
          sum_of_valid_features_in_map / number_of_valid_features;

      for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
        feature_key = feature_vector[feature_i].first;
        feature_value = feature_vector[feature_i].second;
        if (feature_value == 0.0) {
          continue;
        } else {
          feature_vector[feature_i].second =
              feature_value - mean_of_valid_features_in_vector;
        }
      }
    }

    // calculate probability in a feature-wise manner.
    for (int feature_i = 0; feature_i < feature_vector.size(); feature_i++) {
#ifdef DEBUG
      // std::cout << "DEBUG::ProbabilityMap2D::CalculateProbabilityFeatureVectorConditioningPosition: fingerprint likelihood: " << likelihood << std::endl;
#endif
      feature_key = feature_vector[feature_i].first;
      feature_value = feature_vector[feature_i].second;

      if (feature_value == 0.0) {
        // no valid value received for the current feature.
        // uniformly distributed over all positions in map.
        feature_likelihood = 1.0;
      } else {
        if (feature_key_its[feature_i] == feature_distribution->end()) {
          // no probability model for this required feature in the current position.
          // there is no valid sample for this feature existing at the current position in the map.
          // data needs to be normalized.
          feature_value_mean = 0.0;
          feature_value_std = 2.0;
        } else {
          feature_value_mean =
              feature_key_its[feature_i]->second[0] - value_offset_in_map;
          feature_value_std = feature_key_its[feature_i]->second[1];
        }
        UnivariateGaussian univariate_gaussian(feature_value_mean, feature_value_std * feature_value_std);
        feature_likelihood = univariate_gaussian.QuantizedProbability(feature_value, 1.0);
      }
      named_feature_likelihoods.push_back(std::pair<std::string, double>(feature_vector.at(feature_i).first, feature_likelihood));
    }
  }

  return named_feature_likelihoods;
}

void ProbabilityMapper2D::OutputProbabilityStatesConditioningFeatureVector(
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

int ProbabilityMapper2D::Terminate(void) {
  this->distribution_map_ptr_ = nullptr;
  return 0;
}

ProbabilityMapper2D::ProbabilityMapper2D(void) {
  std::set<std::string> state_keys;
  this->state_keys_ = state_keys;
  this->distribution_map_ptr_ = nullptr;
  this->probability_map_smooth_factor_ = 1.0;
  this->zero_centering_ = false;
  this->static_map_offset_ = 0.0;
}

ProbabilityMapper2D::~ProbabilityMapper2D(void) {}

}  // namespace distribution

}  // namespace state_estimation
