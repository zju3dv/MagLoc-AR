/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-09-16 17:07:39
 * @LastEditTime: 2021-09-16 17:07:39
 * @LastEditors: xuehua
 */
#ifndef STATE_ESTIMATION_OBSERVATION_MODEL_IMU_OBSERVATION_MODEL_H_
#define STATE_ESTIMATION_OBSERVATION_MODEL_IMU_OBSERVATION_MODEL_H_

#include <Eigen/Dense>
#include <string>
#include <utility>
#include <vector>

#include "observation_model/base.h"

namespace state_estimation {

namespace observation_model {

struct AccelerometerData {
  double timestamp = -1.0;
  double time_unit_in_second = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

typedef AccelerometerData GyroscopeData;

typedef AccelerometerData GravityData;

class GyroscopeObservation : public Observation {
 public:
  void Init(double buffer_duration);
  int GetObservationFromLines(const std::vector<std::string>& gyroscope_lines, double target_timestamp);
  int GetSequentialObservationFromLines(const std::vector<std::string>& gyroscope_lines);
  std::vector<std::pair<std::string, double>> GetFeatureVector(void);
  Eigen::Matrix<double, 3, 1> GetEigenVector(void);
  double timestamp(void);
  double omega_x(void);
  double omega_y(void);
  double omega_z(void);
  double buffer_duration(void);
  std::vector<double> sequential_timestamps(void);
  std::vector<Eigen::Matrix<double, 3, 1>> sequential_observations(void);

  GyroscopeObservation(void);
  ~GyroscopeObservation();

 private:
  double timestamp_;
  double omega_x_;
  double omega_y_;
  double omega_z_;
  double buffer_duration_;
  std::vector<double> sequential_timestamps_;
  std::vector<Eigen::Matrix<double, 3, 1>> sequential_observations_;
};

class AccelerometerObservation : public Observation {
 public:
  void Init(double buffer_duration);
  int GetObservationFromLines(const std::vector<std::string>& accelerometer_lines, double target_timestamp);
  int GetSequentialObservationFromLines(const std::vector<std::string>& accelerometer_lines);
  std::vector<std::pair<std::string, double>> GetFeatureVector(void);
  Eigen::Matrix<double, 3, 1> GetEigenVector(void);
  double timestamp(void);
  double acc_x(void);
  double acc_y(void);
  double acc_z(void);
  double buffer_duration(void);
  std::vector<double> sequential_timestamps(void);
  std::vector<Eigen::Matrix<double, 3, 1>> sequential_observations(void);

  AccelerometerObservation(void);
  ~AccelerometerObservation();

 private:
  double timestamp_;
  double acc_x_;
  double acc_y_;
  double acc_z_;
  double buffer_duration_;
  std::vector<double> sequential_timestamps_;
  std::vector<Eigen::Matrix<double, 3, 1>> sequential_observations_;
};

}  // namespace observation_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OBSERVATION_MODEL_IMU_OBSERVATION_MODEL_H_
