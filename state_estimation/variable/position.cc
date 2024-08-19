/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:19:30
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-02-03 10:42:13
 */
#include "variable/position.h"

#include <cmath>
#include <string>
#include <vector>

#include "util/misc.h"

namespace state_estimation {

namespace variable {

std::string Position::ToKey(void) {
  return (std::to_string(this->floor_) + "_" + std::to_string(this->x_) + "_" + std::to_string(this->y_));
}

void Position::FromKey(std::string position_key) {
  std::vector<std::string> position_key_split;
  SplitString(position_key, position_key_split, "_");
  this->floor_ = std::stoi(position_key_split[0]);
  this->x_ = std::stod(position_key_split[1]);
  this->y_ = std::stod(position_key_split[2]);
}

void Position::Round(double spatial_interval) {
  double x = std::round(this->x_ / spatial_interval) * spatial_interval;
  if (abs(x) < (spatial_interval / 2)) {
    x = 0.0;
  }
  double y = std::round(this->y_ / spatial_interval) * spatial_interval;
  if (abs(y) < (spatial_interval / 2)) {
    y = 0.0;
  }
  this->x_ = x;
  this->y_ = y;
}

Position::Position(void) {
  this->x_ = 0.0;
  this->y_ = 0.0;
  this->z_ = 0.0;
  this->floor_ = 0;
}

Position::~Position(void) {}

}  // namespace variable

}  // namespace state_estimation
