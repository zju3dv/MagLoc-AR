/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:15:34
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-01 11:23:32
 */
#ifndef STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_H_
#define STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_H_

#include <Eigen/Eigen>

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "distribution/distribution_map_base.h"

namespace state_estimation {

namespace distribution {

class DistributionMap : public DistributionMapBase {
 public:
  void Init(int number_of_label_fields, int number_of_feature_parameters) {
    this->number_of_label_fields_ = number_of_label_fields;
    this->number_of_feature_parameters_ = number_of_feature_parameters;
  }

  int Insert(std::string distribution_map_filepath);
  int Update(std::string distribution_map_filepath);
  std::set<std::string> GetAllKeys(void);

  int InsertKeyCovariance(std::string distribution_covariance_map_filepath);

  void Clear(void) {
    this->distribution_map_.clear();
    this->key_covariance_index_lookup_table_.clear();
    this->key_covariance_ = Eigen::MatrixXd::Zero(0, 0);
  }

  int GetSize(void) {
    return this->distribution_map_.size();
  }

  const std::unordered_map<std::string, std::vector<double>>*
  operator[](std::string state_key) {
    if (this->distribution_map_.find(state_key) != this->distribution_map_.end()) {
      return &(this->distribution_map_[state_key]);
    } else {
      return nullptr;
    }
  }

  int GetKeyCovariance(const std::vector<std::string>& key_vector, Eigen::MatrixXd* key_covariance);
  int GetKeyCovarianceIndex(const std::string& key);

  bool has_key_covariance(void) {
    return this->has_key_covariance_;
  }

  DistributionMap(void);
  ~DistributionMap();

 private:
  int number_of_label_fields_;
  int number_of_feature_parameters_;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::vector<double>>>
      distribution_map_;
  std::unordered_map<std::string, int> key_covariance_index_lookup_table_;
  Eigen::MatrixXd key_covariance_;
  bool has_key_covariance_;
};

}  // namespace distribution

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_H_
