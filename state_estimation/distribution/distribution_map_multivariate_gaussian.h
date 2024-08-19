/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-02-24 20:17:19
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-01 16:38:51
 */
#ifndef STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_MULTIVARIATE_GAUSSIAN_H_
#define STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_MULTIVARIATE_GAUSSIAN_H_

#include <set>
#include <string>
#include <unordered_map>

#include "distribution/distribution_map_base.h"
#include "distribution/gaussian_distribution.h"

namespace state_estimation {

namespace distribution {

class DistributionMapMultivariateGaussian : public DistributionMapBase {
 public:
  void Init(int number_of_label_fields, int number_of_features) {
    this->number_of_label_fields_ = number_of_label_fields;
    this->number_of_features_ = number_of_features;
  }

  int Insert(std::string distribution_map_filepath);
  int Update(std::string distribution_map_filepath);

  std::set<std::string> GetAllKeys(void) {
    std::set<std::string> all_keys;
    for (auto it = this->distribution_map_.begin(); it != this->distribution_map_.end(); it++) {
      all_keys.insert(it->first);
    }
    return all_keys;
  }

  void Clear(void) {
    this->distribution_map_.clear();
  }

  int GetSize(void) {
    return this->distribution_map_.size();
  }

  const std::unordered_map<std::string, NamedMultivariateGaussian>* distribution_map(void) {
    return &(this->distribution_map_);
  }

  const NamedMultivariateGaussian*
  operator[](std::string state_key) {
    return &(this->distribution_map_[state_key]);
  }

  int number_of_label_fields(void) {
    return this->number_of_label_fields_;
  }

  int number_of_features(void) {
    return this->number_of_features_;
  }

  DistributionMapMultivariateGaussian(void) {
    this->number_of_label_fields_ = 0;
    this->number_of_features_ = 0;
    std::unordered_map<std::string, NamedMultivariateGaussian> distribution_map;
    this->distribution_map_ = distribution_map;
  }

  ~DistributionMapMultivariateGaussian() {}

 private:
  int number_of_label_fields_;
  int number_of_features_;
  std::unordered_map<std::string, NamedMultivariateGaussian> distribution_map_;
};

}  // namespace distribution

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_MULTIVARIATE_GAUSSIAN_H_
