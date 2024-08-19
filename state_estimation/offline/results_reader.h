/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-07-15 17:00:08
 * @LastEditTime: 2021-07-15 17:00:54
 * @LastEditors: xuehua
 */
#ifndef STATE_ESTIMATION_OFFLINE_RESULTS_READER_H_
#define STATE_ESTIMATION_OFFLINE_RESULTS_READER_H_

#include <string>
#include <vector>
#include <iostream>

#include <Eigen/Eigen>

#include "offline/sequential_reader_base.h"
#include "variable/position.h"

namespace state_estimation {

namespace offline {

struct TMTEResult {
  double timestamp = 0.0;
  variable::Position gt_position;
  variable::Position est_position;
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
};

struct EuRoCResult {
  double timestamp = -1.0;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Vector3d bw = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
};

class TransitionModelTrajectoryEstimationResultsReader : public SequentialReader {
 public:
  int Init(std::string results_filepath, int number_of_headerlines = 0);

  TMTEResult GetNext(void) {
    TMTEResult result;
    if (this->IsEmpty()) {
      std::cout << "TransitionModelTrajectoryEstimationResultsReader::GetNext::Warning: GetNext but Empty." << std::endl;
    } else {
      result = this->results_.at(this->cur_);
      this->cur_ += 1;
    }
    return result;
  }

  int GetSize(void) {
    return this->results_.size();
  }

 private:
  std::string results_filepath;
  int number_of_headerlines_;
  std::vector<TMTEResult> results_;
};

class EuRoCResultReader : public SequentialReader {
 public:
  int Init(const std::string& result_filepath, int number_of_headerlines = 0);
  EuRoCResult GetNextResult(void);

  int GetSize(void) {
    return this->sequential_results_.size();
  }

 private:
  std::string result_filepath_;
  int number_of_headerlines_;
  std::vector<EuRoCResult> sequential_results_;
};

}  // namespace offline

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OFFLINE_RESULTS_READER_H_
