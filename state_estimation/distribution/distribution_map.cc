/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:15:25
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-01-12 17:15:59
 */
#include "distribution/distribution_map.h"

#include <Eigen/Eigen>
#include <assert.h>

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace distribution {

int DistributionMap::Insert(std::string distribution_map_filepath) {
  std::ifstream map_file(distribution_map_filepath);
  if (!map_file.is_open()) {
    std::cout << "DistributionMap::Insert: cannot open map file -- "
              << distribution_map_filepath << std::endl;
    return 0;
  }

#ifdef DEBUG
  struct timeval time;
  char out[50];
  gettimeofday(&time, NULL);
  sprintf(out, "%.3f", time.tv_sec + time.tv_usec * 1e-6);
  std::cout << "time before loading map: " << out << std::endl;
#endif

  std::string buffer((std::istreambuf_iterator<char>(map_file)),
                     std::istreambuf_iterator<char>(0));
  std::stringstream ss;
  ss.str(buffer);
  map_file.close();

  std::string line;
  std::vector<std::string> line_split;

  if (!std::getline(ss, line)) {
    std::cout << "DistributionMap::Insert: empty map file -- "
              << distribution_map_filepath << std::endl;
    return 0;
  }

  line_split.clear();
  SplitString(line, line_split, ",");

  std::vector<std::string> feature_keys;
  for (int i = 0; i < (line_split.size()); i++) {
    feature_keys.push_back(line_split[i]);
  }

  std::unordered_map<std::string, std::vector<double>> feature_distributions;
  variable::Position position;
  std::vector<double> distribution_parameters;

  while (std::getline(ss, line)) {
    //    while (std::getline(map_file, line)) {
    line_split.clear();
    //line_split = split_string(line, ",");
    SplitString(line, line_split, ",");

    position.x(std::stod(line_split[0]));
    position.y(std::stod(line_split[1]));

    if (this->number_of_label_fields_ == 3) {
      position.floor(std::stoi(line_split[2]));
    } else {
      position.floor(0);
    }

    feature_distributions.clear();
    distribution_parameters.clear();

    assert(line_split.size() == (this->number_of_feature_parameters_ * feature_keys.size() + this->number_of_label_fields_));
    for (int i = 0; i < (feature_keys.size()); i++) {
      for (int j = 0; j < this->number_of_feature_parameters_; j++) {
        distribution_parameters.push_back(std::stod(line_split[i * this->number_of_feature_parameters_ + this->number_of_label_fields_ + j]));
      }
      feature_distributions.insert(std::pair<std::string, std::vector<double>>(feature_keys[i], distribution_parameters));
      distribution_parameters.clear();
    }

    this->distribution_map_.insert(std::pair<std::string, std::unordered_map<std::string, std::vector<double>>>(position.ToKey(), feature_distributions));
  }

  //    map_file.close();

#ifdef DEBUG
  gettimeofday(&time, NULL);
  sprintf(out, "%.3f", time.tv_sec + time.tv_usec * 1e-6);
  std::cout << "time after loading map: " << out << std::endl;
#endif

  return 1;
}

int DistributionMap::Update(std::string distribution_map_filepath) {
  return 0;
}

std::set<std::string> DistributionMap::GetAllKeys(void) {
  std::set<std::string> all_keys;
  for (auto it = this->distribution_map_.begin(); it != this->distribution_map_.end(); it++) {
    all_keys.insert(it->first);
  }
  return all_keys;
}

int DistributionMap::InsertKeyCovariance(std::string distribution_covariance_map_filepath) {
  std::ifstream covariance_map_file(distribution_covariance_map_filepath);
  if (!covariance_map_file.is_open()) {
    std::cout << "DistributionMap::InsetKeyCovariance: cannot open map file -- "
              << distribution_covariance_map_filepath << std::endl;
    return 0;
  }

  std::string buffer((std::istreambuf_iterator<char>(covariance_map_file)), std::istreambuf_iterator<char>(0));
  std::stringstream ss;
  ss.str(buffer);
  covariance_map_file.close();

  std::string line;
  std::vector<std::string> line_split;

  if (!std::getline(ss, line)) {
    std::cout << "DistributionMap::InsertKeyCovariance: empty map file -- "
              << distribution_covariance_map_filepath << std::endl;
    return 0;
  }

  line_split.clear();
  SplitString(line, line_split, ",");

  assert(line_split.size() == 1);
  int number_of_keys = std::stoi(line_split[0]);
  assert(this->GetSize() == number_of_keys);

  this->key_covariance_index_lookup_table_.clear();
  this->key_covariance_ = Eigen::MatrixXd::Zero(number_of_keys, number_of_keys);

  this->has_key_covariance_ = true;

  variable::Position position;
  int current_index = 0;
  while (std::getline(ss, line)) {
    line_split.clear();
    SplitString(line, line_split, ",");
    assert(line_split.size() == this->number_of_label_fields_ + number_of_keys);

    position.x(std::stod(line_split[0]));
    position.y(std::stod(line_split[1]));

    if (this->number_of_label_fields_ == 3) {
      position.floor(std::stoi(line_split[2]));
    } else {
      position.floor(0);
    }

    this->key_covariance_index_lookup_table_.insert(std::pair<std::string, int>(position.ToKey(), current_index));
    for (int i = 0; i < number_of_keys; i++) {
      this->key_covariance_(current_index, i) = std::stod(line_split[this->number_of_label_fields_ + i]);
    }

    current_index++;
  }
  assert(current_index == number_of_keys);
  return number_of_keys;
}

int DistributionMap::GetKeyCovariance(const std::vector<std::string>& key_vector, Eigen::MatrixXd* key_covariance) {
  int n_keys = key_vector.size();
  *key_covariance = Eigen::MatrixXd::Zero(n_keys, n_keys);
  int row_index;
  int col_index;
  int valid_key_count = 0;
  if (!this->has_key_covariance_) {
    valid_key_count = 0;
    return valid_key_count;
  }
  for (int i = 0; i < n_keys; i++) {
    if (this->key_covariance_index_lookup_table_.find(key_vector.at(i)) != this->key_covariance_index_lookup_table_.end()) {
      valid_key_count++;
      row_index = this->key_covariance_index_lookup_table_.at(key_vector.at(i));
      for (int j = 0; j < n_keys; j++) {
        if (this->key_covariance_index_lookup_table_.find(key_vector.at(j)) != this->key_covariance_index_lookup_table_.end()) {
          col_index = this->key_covariance_index_lookup_table_.at(key_vector.at(j));
          key_covariance->operator()(i, j) = this->key_covariance_(row_index, col_index);
        } else {
          key_covariance->operator()(i, j) = 0.0;
        }
      }
    } else {
      key_covariance->row(i).setZero();
    }
  }
  return valid_key_count;
}

int DistributionMap::GetKeyCovarianceIndex(const std::string& key) {
  if ((this->has_key_covariance_) && (this->key_covariance_index_lookup_table_.find(key) != this->key_covariance_index_lookup_table_.end())) {
    return this->key_covariance_index_lookup_table_.at(key);
  } else {
    return -1;
  }
}

DistributionMap::DistributionMap(void) {
  this->number_of_label_fields_ = 0;
  this->number_of_feature_parameters_ = 0;
  this->distribution_map_ = std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>>();
  this->key_covariance_index_lookup_table_ = std::unordered_map<std::string, int>();
  this->key_covariance_ = Eigen::MatrixXd::Zero(0, 0);
  this->has_key_covariance_ = false;
}

DistributionMap::~DistributionMap() {}

}  // namespace distribution

}  // namespace state_estimation