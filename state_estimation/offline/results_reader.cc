/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-07-15 17:27:47
 * @LastEditTime: 2021-07-15 17:27:47
 * @LastEditors: xuehua
 */
#include "offline/results_reader.h"

#include <fstream>
#include <iostream>

#include "util/misc.h"

namespace state_estimation {

namespace offline {

static const int kTimestampIndex = 0;
static const int kGTXIndex = 1;
static const int kEstXIndex = 2;
static const int kGTYIndex = 3;
static const int kEstYIndex = 4;
static const int kDxIndex = 9;
static const int kDyIndex = 10;
static const int kNumItems = 12;
static const int kNumEuRoCResultItems = 17;

int TransitionModelTrajectoryEstimationResultsReader::Init(std::string results_filepath, int number_of_headerlines) {
  this->results_filepath = results_filepath;
  this->number_of_headerlines_ = number_of_headerlines;
  this->results_.clear();

  std::vector<std::string> results_lines = GetLinesInFile(results_filepath);
  for (int i = 0; i < results_lines.size(); i++) {
    std::vector<std::string> line_split;
    SplitString(results_lines.at(i), line_split, ",");
    if (line_split.size() < kNumItems) {
      continue;
    }

    variable::Position gt_position, est_position;
    gt_position.x(std::stod(line_split.at(kGTXIndex)));
    gt_position.y(std::stod(line_split.at(kGTYIndex)));
    est_position.x(std::stod(line_split.at(kEstXIndex)));
    est_position.y(std::stod(line_split.at(kEstYIndex)));

    TMTEResult result;
    result.timestamp = std::stod(line_split.at(kTimestampIndex));
    result.gt_position = gt_position;
    result.est_position = est_position;
    result.dx = std::stod(line_split.at(kDxIndex));
    result.dy = std::stod(line_split.at(kDyIndex));
    result.dz = 0.0;

    this->results_.push_back(result);
  }

  return this->results_.size();
}

int EuRoCResultReader::Init(const std::string& result_filepath, int number_of_headerlines) {
  this->result_filepath_ = result_filepath;
  this->number_of_headerlines_ = number_of_headerlines;
  this->sequential_results_.clear();

  std::vector<std::string> result_lines = GetLinesInFile(result_filepath);
  for (int i = 0; i < result_lines.size(); i++) {
    std::vector<std::string> line_split;
    SplitString(result_lines.at(i), line_split, ",");
    if (line_split.size() < kNumEuRoCResultItems) {
      continue;
    }
    this->sequential_results_.emplace_back(EuRoCResult());
    this->sequential_results_.back().timestamp = stod(line_split.at(0)) * 1e-9;
    this->sequential_results_.back().p(0) = stod(line_split.at(1));
    this->sequential_results_.back().p(1) = stod(line_split.at(2));
    this->sequential_results_.back().p(2) = stod(line_split.at(3));
    double q_w = stod(line_split.at(4));
    double q_x = stod(line_split.at(5));
    double q_y = stod(line_split.at(6));
    double q_z = stod(line_split.at(7));
    this->sequential_results_.back().q = Eigen::Quaterniond(q_w, q_x, q_y, q_z);
    this->sequential_results_.back().v(0) = stod(line_split.at(8));
    this->sequential_results_.back().v(1) = stod(line_split.at(9));
    this->sequential_results_.back().v(2) = stod(line_split.at(10));
    this->sequential_results_.back().bw(0) = stod(line_split.at(11));
    this->sequential_results_.back().bw(1) = stod(line_split.at(12));
    this->sequential_results_.back().bw(2) = stod(line_split.at(13));
    this->sequential_results_.back().ba(0) = stod(line_split.at(14));
    this->sequential_results_.back().ba(1) = stod(line_split.at(15));
    this->sequential_results_.back().ba(2) = stod(line_split.at(16));
  }

  return this->sequential_results_.size();
}

EuRoCResult EuRoCResultReader::GetNextResult(void) {
  EuRoCResult result;
  if (this->IsEmpty()) {
    std::cout << "EuRoCResultReader::GetNextResult::Warning: at the end of the sequence." << std::endl;
  } else {
    result = this->sequential_results_.at(this->cur_);
    this->cur_ += 1;
  }
  return result;
}

}  // namespace offline

}  // namespace state_estimation
