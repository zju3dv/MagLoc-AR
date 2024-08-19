/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-02-03 21:15:36
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-02-03 22:02:39
 */
#include "offline/trajectory_reader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace offline {

int TrajectoryReader::Init(std::string trajectory_path, int with_timestamps) {
  std::vector<double> timestamps;
  std::vector<variable::Position> trajectory;

  std::ifstream trajectory_file(trajectory_path);
  if (!trajectory_file) {
    std::cout << "Cannot load trajectory_path: " << trajectory_path << std::endl;
  }
  std::string line;
  while (getline(trajectory_file, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> trajectory_line_split;
    SplitString(line, trajectory_line_split, ",");
    if (with_timestamps) {
      if (trajectory_line_split.size() < 3) {
        continue;
      }
      double timestamp = std::stod(trajectory_line_split[0]);
      variable::Position trajectory_position;
      trajectory_position.x(std::stod(trajectory_line_split[1]));
      trajectory_position.y(std::stod(trajectory_line_split[2]));
      timestamps.push_back(timestamp);
      trajectory.push_back(trajectory_position);
    } else {
      if (trajectory_line_split.size() < 2) {
        continue;
      }
      variable::Position trajectory_position;
      trajectory_position.x(std::stod(trajectory_line_split[0]));
      trajectory_position.y(std::stod(trajectory_line_split[1]));
      trajectory.push_back(trajectory_position);
    }
  }
  trajectory_file.close();

  this->timestamps_ = timestamps;
  this->trajectory_ = trajectory;
  this->with_timestamps_ = with_timestamps;
  return 1;
}

}  // namespace offline

}  // namespace state_estimation
