/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2022-03-07 17:04:02
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-03-09 20:56:46
 */
#ifndef STATE_ESTIMATION_SAMPLER_GAUSSIAN_3D_ORIENTATION_SAMPLER_H_
#define STATE_ESTIMATION_SAMPLER_GAUSSIAN_3D_ORIENTATION_SAMPLER_H_

#include <Eigen/Dense>
#include <random>
#include <iostream>

#include "sampler/gaussian_sampler.h"
#include "distribution/gaussian_distribution.h"
#include "util/misc.h"

namespace state_estimation {

namespace sampler {

class Gaussian3DOrientationSampler {
 public:
  void Seed(int random_seed) {
    this->mvg_sampler_.Seed(random_seed);
  }

  Eigen::Quaterniond SampleByGaussianOnAngleAxis(Eigen::Quaterniond q_mean, Eigen::Matrix3d covariance) {
    if (~IsCovarianceMatrix(covariance)) {
      std::cout << "Gaussian2DOrientationSampler::SampleByGaussianOnAngleAxis: the providede covariance matrix is illegal." << std::endl;
      assert(false);
    }
    this->mvg_sampler_.SetParams(Eigen::Vector3d::Zero(), covariance);
    Eigen::Vector3d sample_q_jitter = this->mvg_sampler_.Sample();
    return LogVector2Quaternion(sample_q_jitter) * q_mean;
  }

  double CalculateLogProbabilityByGaussianOnAngleAxis(Eigen::Quaterniond q_mean, Eigen::Matrix3d covariance, Eigen::Quaterniond sample_q) {
    Eigen::Quaterniond q_jitter = sample_q * q_mean.conjugate();
    Eigen::Vector3d q_jitter_log = Quaternion2LogVector(q_jitter);
    std::vector<double> x = {q_jitter_log(0), q_jitter_log(1), q_jitter_log(2)};
    std::vector<double> jitter_mean = {0.0, 0.0, 0.0};
    std::vector<double> jitter_covariance;
    for (int i = 0; i < 3; i++) {
      for (int j = i ; j < 3; j++) {
        jitter_covariance.push_back(covariance(i, j));
      }
    }
    distribution::MultivariateGaussian mvg_distribution(jitter_mean, jitter_covariance);
    return std::log(mvg_distribution.QuantizedProbability(x));
  }

  Gaussian3DOrientationSampler(void) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
  }

  ~Gaussian3DOrientationSampler() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
};

}  // namespace sampler

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_SAMPLER_GAUSSIAN_3D_ORIENTATION_SAMPLER_H_
