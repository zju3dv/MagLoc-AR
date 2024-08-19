/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:15:25
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-01 16:24:42
 */
#include "distribution/distribution_map_multivariate_gaussian.h"

#include <assert.h>

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace distribution {

int DistributionMapMultivariateGaussian::Insert(std::string distribution_map_filepath) {
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

  if (this->number_of_features_ != feature_keys.size()) {
    std::cout << "DistributionMapMultivariateGaussian::Insert: "
              << "the number of names of features in the header does not match the provided number of features."
              << std::endl;
  }
  assert(this->number_of_features_ == feature_keys.size());

  while (std::getline(ss, line)) {
    line_split.clear();
    SplitString(line, line_split, ",");

    int expect_number_of_line_items = (this->number_of_label_fields_ + this->number_of_features_ * (this->number_of_features_ + 3) / 2);
    if (expect_number_of_line_items != line_split.size()) {
      std::cout << "DistributionMapMultivariateGaussian::Insert: "
                << "the number of items in a line does not match the expected value."
                << std::endl;
    }
    assert(line_split.size() == expect_number_of_line_items);

    variable::Position position;
    position.x(std::stod(line_split[0]));
    position.y(std::stod(line_split[1]));
    if (this->number_of_label_fields_ == 3) {
      position.floor(std::stoi(line_split[2]));
    } else {
      position.floor(0);
    }

    std::unordered_map<std::string, int> variable_name_to_index_dict;
    std::vector<double> means;
    std::vector<double> covariances;
    for (int i = 0; i < this->number_of_features_; i++) {
      variable_name_to_index_dict.insert(std::pair<std::string, int>(feature_keys[i], i));
      means.push_back(std::stod(line_split[this->number_of_label_fields_ + i]));
    }
    for (int i = this->number_of_label_fields_ + this->number_of_features_; i < line_split.size(); i++) {
      covariances.push_back(std::stod(line_split[i]));
    }
    NamedMultivariateGaussian named_multivariate_gaussian(variable_name_to_index_dict, means, covariances);

    this->distribution_map_.insert(std::pair<std::string, NamedMultivariateGaussian>(position.ToKey(), named_multivariate_gaussian));
  }

#ifdef DEBUG
  gettimeofday(&time, NULL);
  sprintf(out, "%.3f", time.tv_sec + time.tv_usec * 1e-6);
  std::cout << "time after loading map: " << out << std::endl;
#endif

  return 1;
}

int DistributionMapMultivariateGaussian::Update(std::string distribution_map_filepath) {
  return 0;
}

}  // namespace distribution

}  // namespace state_estimation
