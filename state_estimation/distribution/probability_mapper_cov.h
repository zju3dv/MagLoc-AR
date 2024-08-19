/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-01 19:39:34
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-20 15:54:06
 */
#ifndef STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_COV_H_
#define STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_COV_H_

#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distribution/distribution_map_multivariate_gaussian.h"
#include "distribution/gaussian_distribution.h"
#include "distribution/probability_mapper_base.h"

namespace state_estimation {

namespace distribution {

class ProbabilityMapperCov : public ProbabilityMapperBase {
 public:
  void Init(std::set<std::string> state_keys,
            DistributionMapMultivariateGaussian* distribution_map_ptr,
            double probability_map_smooth_factor,
            bool is_zero_centered,
            double static_map_offset = 0.0) {
    this->state_keys_ = state_keys;
    this->distribution_map_ptr_ = distribution_map_ptr;
    this->probability_map_smooth_factor_ = probability_map_smooth_factor;
    this->zero_centering_ = is_zero_centered;
    this->static_map_offset_ = static_map_offset;
  }

  std::set<std::string> GetStateKeys(void) const {
    return this->state_keys_;
  }

  void SetStateKeys(std::set<std::string> state_keys) {
    this->state_keys_ = state_keys;
  }

  double GetProbabilityMapSmoothFactor(void) const {
    return this->probability_map_smooth_factor_;
  }

  double GetStaticMapOffset(void) const {
    return this->static_map_offset_;
  }

  const NamedMultivariateGaussian* GetDistributionParams(std::string state_key) {
    const NamedMultivariateGaussian* named_multivariate_gaussian_ptr = nullptr;
    if (this->state_keys_.find(state_key) != this->state_keys_.end()) {
      named_multivariate_gaussian_ptr = (*this->distribution_map_ptr_)[state_key];
    } else {
      std::cout << "ProbabilityMapperCov::GetDistributionParams: wrong state_key." << std::endl;
      std::cout << "ProbabilityMapperCov::GetDistributionParams: return nullptr." << std::endl;
    }
    return named_multivariate_gaussian_ptr;
  }

  void SetDistributionMap(DistributionMapMultivariateGaussian* distribution_map_ptr) {
    this->distribution_map_ptr_ = distribution_map_ptr;
  }

  void SetZeroCentering(bool is_zero_centered) {
    this->zero_centering_ = is_zero_centered;
  }

  void SetStaticMapOffset(double static_map_offset) {
    this->static_map_offset_ = static_map_offset;
  }

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

  void OutputProbabilityStatesConditioningFeatureVector(
      std::vector<std::pair<std::string, double>> feature_vector,
      double dynamic_map_offset = 0.0) const;

  int Terminate(void);

  ProbabilityMapperCov(void) {
    std::set<std::string> state_keys;
    this->state_keys_ = state_keys;
    this->distribution_map_ptr_ = nullptr;
    this->probability_map_smooth_factor_ = 1.0;
    this->zero_centering_ = false;
    this->static_map_offset_ = 0.0;
  }

  ~ProbabilityMapperCov() {}

 private:
  std::set<std::string> state_keys_;
  DistributionMapMultivariateGaussian* distribution_map_ptr_;
  double probability_map_smooth_factor_;
  bool zero_centering_;
  double static_map_offset_;
};

}  // namespace distribution

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_COV_H_
