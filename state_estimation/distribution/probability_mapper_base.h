/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-01 19:41:49
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-20 15:34:30
 */
#ifndef STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_BASE_H_
#define STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_BASE_H_

#include <cmath>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace state_estimation {

namespace distribution {

class ProbabilityMapperBase {
 public:
  virtual std::set<std::string> GetStateKeys(void) const = 0;
  virtual void SetStateKeys(std::set<std::string> state_keys) = 0;

  virtual std::unordered_map<std::string, double> LookupMeans(std::string state_key) const = 0;

  virtual std::unordered_map<std::string, double>
  CalculateProbabilityStatesConditioningFeatureVector(
      std::vector<std::pair<std::string, double>> feature_vector,
      double dynamic_map_offset) const = 0;

  virtual double CalculateProbabilityFeatureVectorConditioningState(
      std::vector<std::pair<std::string, double>> feature_vector,
      std::string state_key,
      double dynamic_map_offset) const = 0;

  virtual double CalculateProbabilityFeatureVectorConditioningStateLog(
      std::vector<std::pair<std::string, double>> feature_vector,
      std::string state_key,
      double dynamic_map_offset) const = 0;

  virtual void OutputProbabilityStatesConditioningFeatureVector(
      std::vector<std::pair<std::string, double>> feature_vector,
      double dynamic_map_offset) const = 0;

  virtual int Terminate(void) = 0;
};

}  // namespace distribution

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_DISTRIBUTION_PROBABILITY_MAPPER_BASE_H_
