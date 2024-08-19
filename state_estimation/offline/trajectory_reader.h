/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-02-03 21:07:04
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-02-03 22:03:13
 */
#ifndef STATE_ESTIMATION_OFFLINE_TRAJECTORY_READER_H_
#define STATE_ESTIMATION_OFFLINE_TRAJECTORY_READER_H_

#include <string>
#include <vector>

#include "variable/position.h"

namespace state_estimation {

namespace offline {

class TrajectoryReader {
 public:
  int Init(std::string trajectory_path, int with_timestamps);

  std::vector<variable::Position> trajectory(void) {
    return this->trajectory_;
  }

  std::vector<double> timestamps(void) {
    return this->timestamps_;
  }

  int with_timestamps(void) {
    return this->with_timestamps_;
  }

  TrajectoryReader(void) {
    std::vector<double> timestamps;
    this->timestamps_ = timestamps;
    std::vector<variable::Position> trajectory;
    this->trajectory_ = trajectory;
    this->with_timestamps_ = 0;
  }

  ~TrajectoryReader() {}

 private:
  std::vector<double> timestamps_;
  std::vector<variable::Position> trajectory_;
  int with_timestamps_;
};

}  // namespace offline

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OFFLINE_TRAJECTORY_READER_H_
