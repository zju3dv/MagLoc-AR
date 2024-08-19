/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:19:07
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-09 17:24:01
 */
#include "variable/orientation.h"

#include <Eigen/Geometry>
#include <string>
#include <vector>

#include "util/misc.h"

namespace state_estimation {

namespace variable {

std::string Orientation::ToKey(void) {
  // TODO(xuehua): the current key formatting is wrong.
  Eigen::Quaterniond q = this->q_;
  q.normalize();
  double q_array[4] = {q.w(), q.x(), q.y(), q.z()};
  int index = 0;
  for (; index < 4; index++) {
    if (q_array[index] != 0.0) {
      break;
    }
  }
  std::string orientation_key;
  if (index < 4 && q_array[index] < 0.0) {
    for (int i = 0; i < 4; i++) {
      orientation_key += std::to_string(-q_array[i]);
    }
  } else {
    for (int i = 0; i < 4; i++) {
      orientation_key += std::to_string(q_array[i]);
    }
  }
  return orientation_key;
}

void Orientation::FromKey(std::string orientation_key) {
  std::vector<std::string> orientation_key_split;
  SplitString(orientation_key, orientation_key_split, "_");
  double q_array[4];
  for (int i = 0; i < 4; i++) {
    q_array[i] = std::stod(orientation_key_split[i]);
  }
  Eigen::Quaterniond q(q_array);
  this->q_ = q;
}

Orientation::Orientation(void) {
  Eigen::Quaterniond q(0.0, 0.0, 0.0, 0.0);
  this->q_ = q;
}

Orientation::~Orientation() {}

}  // namespace variable

}  // namespace state_estimation
