/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-10 16:22:13
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-18 14:06:58
 */
#ifndef STATE_ESTIMATION_SAMPLER_GAUSSIAN_SAMPLER_H_
#define STATE_ESTIMATION_SAMPLER_GAUSSIAN_SAMPLER_H_

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <ctime>

#include "util/misc.h"

namespace state_estimation {

namespace sampler {

class UnivariateGaussianSampler {
 public:
  void Init(double mean, double variance, int p_size = 10000) {
    if (variance < 0.0) {
      std::cout << "UnivariateGaussianSampler::Init: the provided variance is negative." << std::endl;
      assert(variance >= 0.0);
    }
    this->mean_ = mean;
    this->variance_ = variance;
    this->p_size_ = p_size;
    struct timespec tn;
    clock_gettime(CLOCK_REALTIME, &tn);
    this->generator_.seed(static_cast<uint64_t>(tn.tv_nsec));
  }

  double Sample(void) {
    // reference: https://aishack.in/tutorials/generating-multivariate-gaussian-random/
    std::uniform_real_distribution<double> uniform_sampler(0.0, 1.0);
    double sample_sum = 0.0;
    for (int i = 0; i < this->p_size_; i++) {
      sample_sum += uniform_sampler(this->generator_);
    }
    return ((sample_sum - this->p_size_ / 2.0) / (std::sqrt(this->p_size_ / 12))) *
               std::sqrt(this->variance_) +
           this->mean_;
  }

  void Seed(int random_seed) {
    this->generator_.seed(static_cast<uint64_t>(random_seed));
  }

  UnivariateGaussianSampler(void) {
    this->Init(0.0, 1.0);
  }

  UnivariateGaussianSampler(double mean, double variance) {
    this->Init(mean, variance);
  }

  ~UnivariateGaussianSampler() {}

 private:
  double mean_;
  double variance_;
  int p_size_;
  std::default_random_engine generator_;
};

class UnivariateGaussianSamplerStd {
 public:
  void Init(double mean, double variance) {
    if (variance < 0.0) {
      std::cout << "UnivariateGaussianSampler::Init: the provided variance is negative." << std::endl;
      assert(variance >= 0.0);
    }
    this->normal_distribution_ = std::normal_distribution<double>(mean, std::sqrt(variance));
    struct timespec tn;
    clock_gettime(CLOCK_REALTIME, &tn);
    this->generator_.seed(static_cast<uint64_t>(tn.tv_nsec));

  }

  void SetParams(double mean, double variance) {
    this->normal_distribution_ = std::normal_distribution<double>(mean, std::sqrt(variance));
  }

  double Sample(void) {
    return this->normal_distribution_(this->generator_);
  }

  void Seed(int random_seed) {
    this->generator_.seed(static_cast<uint64_t>(random_seed));
  }

  double mean(void) {
    return this->normal_distribution_.mean();
  }

  double stddev(void) {
    return this->normal_distribution_.stddev();
  }

  UnivariateGaussianSamplerStd(void) {
    this->Init(0.0, 1.0);
  }

  UnivariateGaussianSamplerStd(double mean, double variance) {
    this->Init(mean, variance);
  }

  ~UnivariateGaussianSamplerStd() {}

 private:
  std::normal_distribution<double> normal_distribution_;
  std::default_random_engine generator_;
};

class MultivariateGaussianSampler {
 public:
  void Init(Eigen::VectorXd means, Eigen::MatrixXd covariances) {
    this->SetParams(means, covariances);
    this->univariate_gaussian_sampler_ = UnivariateGaussianSamplerStd(0.0, 1.0);
  }

  void SetParams(Eigen::VectorXd means, Eigen::MatrixXd covariances) {
    if ((covariances.size() == this->covariances_.size()) && (covariances == this->covariances_)) {
      if (means != this->means_) {
        if (means.rows() != covariances.rows()) {
          std::cout << "MultivariateGaussianSampler::Init: the numbers of variables in means and covariance matrix do not match." << std::endl;
          assert(means.rows() == covariances.rows());
        } else {
          this->means_ = means;
        }
      }
      return;
    }
    bool is_valid = IsCovarianceMatrix(covariances);
    if (!is_valid) {
      std::cout << "MultivariateGaussianSampler::Init: the provided covariance matrix is invalid:" << std::endl;
      std::cout << covariances << "." << std::endl;
      assert(is_valid);
    }
    if (means.rows() != covariances.rows()) {
      std::cout << "MultivariateGaussianSampler::Init: the numbers of variables in means and covariance matrix do not match." << std::endl;
      assert(means.rows() == covariances.rows());
    }
    this->means_ = means;
    this->covariances_ = covariances;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariances);
    Eigen::MatrixXd eigen_vectors = eigen_solver.eigenvectors();
    Eigen::VectorXd eigen_values = eigen_solver.eigenvalues();
    for (int i = 0; i < eigen_values.size(); i++) {
      eigen_values(i) = std::sqrt(eigen_values(i));
    }
    this->cov_eigen_vectors_ = eigen_vectors;
    this->cov_eigen_values_sqrt_ = eigen_values;
    this->cov_eigen_values_sqrt_weighted_eigen_vectors_ = eigen_vectors * eigen_values.asDiagonal();
  }

  int NumberOfVariables(void) {
    return this->means_.rows();
  }

  Eigen::VectorXd Sample(void) {
    // reference: https://aishack.in/tutorials/generating-multivariate-gaussian-random/
    Eigen::VectorXd sample_vector = this->means_;
    for (int i = 0; i < sample_vector.rows(); i++) {
      sample_vector(i) = this->univariate_gaussian_sampler_.Sample();
    }
    sample_vector = this->cov_eigen_values_sqrt_weighted_eigen_vectors_ * sample_vector;
    sample_vector += this->means_;
    return sample_vector;
  }

  void Seed(int random_seed) {
    this->univariate_gaussian_sampler_.Seed(random_seed);
  }

  Eigen::VectorXd mean(void) {
    return this->means_;
  }

  Eigen::MatrixXd covariance(void) {
    return this->covariances_;
  }

  MultivariateGaussianSampler(void) {
    this->Init(Eigen::Matrix<double, 1, 1>(0.0), Eigen::Matrix<double, 1, 1>(1.0));
  }

  MultivariateGaussianSampler(Eigen::VectorXd means, Eigen::MatrixXd covariances) {
    this->Init(means, covariances);
  }

  ~MultivariateGaussianSampler() {}

 private:
  Eigen::VectorXd means_;
  Eigen::MatrixXd covariances_;
  UnivariateGaussianSamplerStd univariate_gaussian_sampler_;
  Eigen::MatrixXd cov_eigen_vectors_;
  Eigen::VectorXd cov_eigen_values_sqrt_;
  Eigen::MatrixXd cov_eigen_values_sqrt_weighted_eigen_vectors_;
};

}  // namespace sampler

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_SAMPLER_GAUSSIAN_SAMPLER_H_
