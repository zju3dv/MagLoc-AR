/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:15:55
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-20 15:35:16
 */
#ifndef STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_2D_H_
#define STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_2D_H_

#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distribution/distribution_map.h"
#include "distribution/probability_mapper_base.h"

namespace state_estimation {

namespace distribution {

class ProbabilityMapper2D : public ProbabilityMapperBase {
 public:
  void Init(std::set<std::string> state_keys,
            DistributionMap* distribution_map_ptr,
            double probability_map_smooth_factor,
            bool zero_centering,
            double static_map_offset = 0.0,
            int threshold_of_effective_feature_number_for_zero_centering = 5) {
    for (auto it = state_keys.begin(); it != state_keys.end(); it++) {
      assert(distribution_map_ptr->operator[](*it) != nullptr);
    }
    this->state_keys_ = state_keys;
    this->distribution_map_ptr_ = distribution_map_ptr;
    this->probability_map_smooth_factor_ = probability_map_smooth_factor;
    this->zero_centering_ = zero_centering;
    this->static_map_offset_ = static_map_offset;
    this->threshold_of_effective_feature_number_for_zero_centering_ = threshold_of_effective_feature_number_for_zero_centering;
  }

  std::set<std::string> GetStateKeys(void) const {
    return this->state_keys_;
  }

  double GetProbabilityMapSmoothFactor(void) const {
    return this->probability_map_smooth_factor_;
  }

  void SetProbabilityMapSmoothFactor(double smooth_factor) {
    this->probability_map_smooth_factor_ = smooth_factor;
  }

  double GetStaticMapOffset(void) const {
    return this->static_map_offset_;
  }

  const std::unordered_map<std::string, std::vector<double>>* GetDistributionParams(std::string state_key) const {
    const std::unordered_map<std::string, std::vector<double>>* distribution_params = nullptr;
    if (this->state_keys_.find(state_key) != this->state_keys_.end()) {
      distribution_params = (*this->distribution_map_ptr_)[state_key];
    } else {
#ifdef DEBUG_PROBABILITY_MAPPER
      std::cout << "ProbabilityMapper::GetDistributionParams: wrong state_key." << std::endl;
      std::cout << "ProbabilityMapper::GetDistributionParams: return nullptr." << std::endl;
#endif
    }
    return distribution_params;
  }

  int GetKeyCovariance(const std::vector<std::string>& key_vector, Eigen::MatrixXd* key_covariance) const {
    return this->distribution_map_ptr_->GetKeyCovariance(key_vector, key_covariance);
  }

  void SetDistributionMap(DistributionMap* distribution_map_ptr) {
    this->distribution_map_ptr_ = distribution_map_ptr;
  }

  void SetStateKeys(std::set<std::string> state_keys) {
    this->state_keys_ = state_keys;
  }

  void SetZeroCentering(bool is_zero_centered) {
    this->zero_centering_ = is_zero_centered;
  }

  void SetStaticMapOffset(double static_map_offset) {
    this->static_map_offset_ = static_map_offset;
  }

  int GetValidFeatureStatistics(
      std::vector<std::pair<std::string, double>> feature_vector,
      std::string state_key,
      double* valid_feature_map_mean,
      double* valid_feature_client_mean) const;

  int ShiftFeatureVector(std::vector<std::pair<std::string, double>>* feature_vector, double shift_value) const;

  std::unordered_map<std::string, double> LookupMeans(std::string state_key) const;

  std::unordered_map<std::string, double>
  CalculateProbabilityStatesConditioningFeatureVector(
      std::vector<std::pair<std::string, double>> feature_vector,
      double dynamic_map_offset = 0.0) const;

  double CalculateProbabilityFeatureVectorConditioningState(
      std::vector<std::pair<std::string, double>> feature_vector,
      std::string state_key,
      double dynamic_map_offset = 0.0) const;

  double CalculateProbabilityFeatureVectorConditioningStateLog(
      std::vector<std::pair<std::string, double>> feature_vector,
      std::string state_key,
      double dynamic_map_offset = 0.0) const;

  double CalculateNEESFeatureVectorConditioningState(
      std::vector<std::pair<std::string, double>> feature_vector,
      std::string state_key,
      int* num_features,
      double* error_mean,
      double* std_mean) const;

  std::vector<std::pair<std::string, double>> CalculateProbabilityFeatureVectorConditioningStateSeparated(
      std::vector<std::pair<std::string, double>> feature_vector,
      std::string state_key) const;

  void OutputProbabilityStatesConditioningFeatureVector(
      std::vector<std::pair<std::string, double>> feature_vector,
      double dynamic_map_offset = 0.0) const;

  int Terminate(void);

  ProbabilityMapper2D(void);
  ~ProbabilityMapper2D();

 private:
  std::set<std::string> state_keys_;
  DistributionMap* distribution_map_ptr_;
  double probability_map_smooth_factor_;
  bool zero_centering_;
  double static_map_offset_;
  int threshold_of_effective_feature_number_for_zero_centering_;
};

}  // namespace distribution

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_2D_H_
