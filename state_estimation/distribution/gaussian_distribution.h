/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-02-24 20:39:59
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-02 14:29:27
 */
#ifndef STATE_ESTIMATION_DISTRIBUTION_GAUSSIAN_DISTRIBUTION_H_
#define STATE_ESTIMATION_DISTRIBUTION_GAUSSIAN_DISTRIBUTION_H_

#include <assert.h>
#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "util/misc.h"

namespace state_estimation {

namespace distribution {

static const double kDefaultQuantizationResolution = 1e-5;
static const double kEpsilon = 1e-30;
static const double kLogSqrt2Pi = 0.5 * std::log(2.0 * M_PI);

class UnivariateGaussian {
 public:
  double PDF(double x) const {
    return ((1.0 / std::sqrt(2 * M_PI * this->variance_)) *
            std::exp(-std::pow((x - this->mean_), 2.0) / (2.0 * this->variance_)));
  }

  double QuantizedProbability(double x, double quantization_resolution = kDefaultQuantizationResolution) const {
    return this->PDF(x) * quantization_resolution;
  }

  double CDF(double x) const {
    return (std::erfc(-((x - this->mean_) / std::sqrt(this->variance_)) / std::sqrt(2.0)) / 2.0);
  }

  double mean(void) const {
    return this->mean_;
  }

  double variance(void) const {
    return this->variance_;
  }

  double std(void) const {
    return std::sqrt(this->variance_);
  }

  UnivariateGaussian(void) {
    this->mean_ = 0.0;
    this->variance_ = 0.0;
  }

  UnivariateGaussian(double mean, double variance) {
    this->mean_ = mean;
    this->variance_ = variance;
  }

  ~UnivariateGaussian() {}

 private:
  double mean_;
  double variance_;
};

class MultivariateGaussian {
 public:
  double PDF(const Eigen::VectorXd& x) const {
    return std::exp(this->LogPDF(x));
  }

  double PDF(const std::vector<double>& x_vector) const {
    Eigen::VectorXd x = Vector2EigenVector(x_vector);
    return this->PDF(x);
  }

  double LogPDF(const Eigen::VectorXd& x) const {
    if (x.size() != this->mean_.size()) {
      std::cout << "MultivariateGaussian::LogPDF: the number of variables does not match." << std::endl;
      exit(0);
    }
    if (this->covariance_determinant_sqrt_ < kEpsilon) {
      if ((x - this->mean_).norm() < kEpsilon) {
        return +INFINITY;
      } else {
        return -INFINITY;
      }
    }
    return - this->mean_.size() * kLogSqrt2Pi - std::log(this->covariance_determinant_sqrt_) - 0.5 * (this->covariance_llt_.matrixL().solve(x - this->mean_)).squaredNorm();
  }

  double LogPDF(const std::vector<double>& x_vector) const {
    Eigen::VectorXd x = Vector2EigenVector(x_vector);
    return this->LogPDF(x);
  }

  double QuantizedProbability(const Eigen::VectorXd& x, double quantization_resolution = kDefaultQuantizationResolution) const {
    double pdf = this->PDF(x);
    if (pdf == +INFINITY) {
      return 1.0;
    } else {
      return pdf * std::pow(quantization_resolution, this->mean_.size());
    }
  }

  double QuantizedProbability(const std::vector<double>& x_vector, double quantization_resolution = kDefaultQuantizationResolution) const {
    Eigen::VectorXd x = Vector2EigenVector(x_vector);
    return this->QuantizedProbability(x, quantization_resolution);
  }

  MultivariateGaussian GetMultivariateGaussianByIndexVector(std::vector<int> variable_index_vector) const {
    Eigen::VectorXd mean(variable_index_vector.size());
    Eigen::MatrixXd covariance(variable_index_vector.size(), variable_index_vector.size());
    for (int i = 0; i < variable_index_vector.size(); i++) {
      int index = variable_index_vector.at(i);
      if (index >= mean.size()) {
        std::cout << "MultivariateGaussian::GetMultivariateGaussianByIndexVector: index out of range." << std::endl;
        exit(0);
      }
      mean(i) = this->mean_(index);
      for (int j = i; j < variable_index_vector.size(); j++) {
        int index_temp = variable_index_vector.at(j);
        if (index_temp >= mean.size()) {
          std::cout << "MultivariateGaussian::GetMultivariateGaussianByIndexVector: index out of range." << std::endl;
          exit(0);
        }
        covariance(i, j) = this->covariance_(index, index_temp);
        if (i != j) {
          covariance(j, i) = covariance(i, j);
        }
      }
    }
    MultivariateGaussian multivariate_gaussian(mean, covariance);
    return multivariate_gaussian;
  }

  UnivariateGaussian GetUnivariateGaussianByIndex(int variable_index) const {
    if (variable_index > (this->mean_.size() - 1)) {
      std::cout << "MultivariateGaussian::GetUnivariateGaussianByIndex: index exceeds the number of variables." << std::endl;
    }
    UnivariateGaussian univariate_gaussian(this->mean_(variable_index), this->covariance_(variable_index, variable_index));
    return univariate_gaussian;
  }

  Eigen::VectorXd mean(void) const {
    return this->mean_;
  }

  Eigen::MatrixXd covariance(void) const {
    return this->covariance_;
  }

  MultivariateGaussian(void) {
    this->mean_ = Eigen::VectorXd();
    this->covariance_ = Eigen::MatrixXd();
    this->covariance_llt_ = Eigen::LLT<Eigen::MatrixXd>();
    this->covariance_determinant_sqrt_ = 0.0;
  }

  MultivariateGaussian(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) {
    if (covariance.cols() != covariance.rows()) {
      std::cout << "MultivariateGaussian: the covariance matrix is not a square matrix." << std::endl;
      exit(0);
    }
    if (mean.size() != covariance.rows()) {
      std::cout << "MultivariateGaussian: the size of means and the size of covariances do not match." << std::endl;
      exit(0);
    }
    this->mean_ = mean;
    this->covariance_ = covariance;
    this->covariance_llt_ = covariance.llt();
    if ((this->covariance_llt_.info() != Eigen::Success) && (std::abs(this->covariance_.determinant()) >= kEpsilon)) {
      std::cout << "MultivariateGaussian::MultivariateGaussian: cholesky decomposition failed and the determinant of the covariance matrix does not equal to zero!" << std::endl;
      exit(0);
    }
    this->covariance_determinant_sqrt_ = this->covariance_llt_.matrixL().determinant();
  }

  MultivariateGaussian(const std::vector<double>& mean_vector, const std::vector<double>& covariance_compact_vector)
    : MultivariateGaussian{Vector2EigenVector(mean_vector), CompactVectorToCovarianceMatrix(covariance_compact_vector, mean_vector.size())} {
  }

  ~MultivariateGaussian() {}

 protected:
  Eigen::VectorXd mean_;
  Eigen::MatrixXd covariance_;
  Eigen::LLT<Eigen::MatrixXd> covariance_llt_;
  double covariance_determinant_sqrt_;
};

class NamedMultivariateGaussian : public MultivariateGaussian {
 public:
  double PDF(std::vector<double> X) const {
    return MultivariateGaussian::PDF(X);
  }

  double PDF(std::vector<std::pair<std::string, double>> named_X) const {
    std::vector<int> index_vector;
    std::vector<double> X;
    for (int i = 0; i < named_X.size(); i++) {
      std::string variable_name = named_X.at(i).first;
      if (this->variable_name_to_index_dict_.find(variable_name) != this->variable_name_to_index_dict_.end()) {
        int current_index = this->variable_name_to_index_dict_.at(variable_name);
        index_vector.push_back(current_index);
        X.push_back(named_X.at(i).second);
      } else {
        std::cout << "NamedMultivariateGaussian::PDF: the variable name does not exist: " << variable_name << std::endl;
      }
    }
    MultivariateGaussian multivariate_gaussian = this->GetMultivariateGaussianByIndexVector(index_vector);
    return multivariate_gaussian.PDF(X);
  }

  double QuantizedProbability(std::vector<double> X, double quantization_resolution = kDefaultQuantizationResolution) const {
    return MultivariateGaussian::QuantizedProbability(X, quantization_resolution);
  }

  double QuantizedProbability(std::vector<std::pair<std::string, double>> named_X, double quantization_resolution = kDefaultQuantizationResolution) const {
    std::vector<int> index_vector;
    std::vector<double> X;
    for (int i = 0; i < named_X.size(); i++) {
      std::string variable_name = named_X.at(i).first;
      if (this->variable_name_to_index_dict_.find(variable_name) != this->variable_name_to_index_dict_.end()) {
        int current_index = this->variable_name_to_index_dict_.at(variable_name);
        index_vector.push_back(current_index);
        X.push_back(named_X.at(i).second);
      } else {
        std::cout << "NamedMultivariateGaussian::QuantizedProbability: the variable name does not exist: " << variable_name << std::endl;
      }
    }
    MultivariateGaussian multivariate_gaussian = this->GetMultivariateGaussianByIndexVector(index_vector);
    return multivariate_gaussian.QuantizedProbability(X, quantization_resolution);
  }

  NamedMultivariateGaussian GetNamedMultivariateGaussianByNameVector(std::vector<std::string> variable_names) const {
    std::unordered_map<std::string, int> variable_name_to_index_dict;
    std::vector<int> index_vector;
    for (int i = 0; i < variable_names.size(); i++) {
      std::string current_variable_name = variable_names[i];
      if (this->variable_name_to_index_dict_.find(current_variable_name) != this->variable_name_to_index_dict_.end()) {
        int current_index = this->variable_name_to_index_dict_.at(current_variable_name);
        index_vector.push_back(current_index);
        variable_name_to_index_dict.insert(std::pair<std::string, int>(current_variable_name, index_vector.size() - 1));
      } else {
        std::cout << "NamedMultivariateGaussian::GetNamedMultivariateGaussianByNameVector: variable name does not exist: "
                  << current_variable_name << std::endl;
      }
    }
    MultivariateGaussian multivariate_gaussian = this->GetMultivariateGaussianByIndexVector(index_vector);
    NamedMultivariateGaussian named_multivariate_gaussian(variable_name_to_index_dict, multivariate_gaussian);
    return named_multivariate_gaussian;
  }

  MultivariateGaussian GetMultivariateGaussianByNameVector(std::vector<std::string> variable_names) const {
    std::vector<int> index_vector;
    for (int i = 0; i < variable_names.size(); i++) {
      std::string current_variable_name = variable_names[i];
      if (this->variable_name_to_index_dict_.find(current_variable_name) != this->variable_name_to_index_dict_.end()) {
        int current_index = this->variable_name_to_index_dict_.at(current_variable_name);
        index_vector.push_back(current_index);
      } else {
        std::cout << "NamedMultivariateGaussian::GetMultivariateGaussianByNameVector: variable name does not exist: "
                  << current_variable_name << std::endl;
      }
    }
    MultivariateGaussian multivariate_gaussian = this->GetMultivariateGaussianByIndexVector(index_vector);
    return multivariate_gaussian;
  }

  std::unordered_map<std::string, int> variable_name_to_index_dict(void) const {
    return this->variable_name_to_index_dict_;
  }

  NamedMultivariateGaussian(void) : MultivariateGaussian{} {
    std::unordered_map<std::string, int> variable_name_to_index_dict;
    this->variable_name_to_index_dict_ = variable_name_to_index_dict;
  }

  NamedMultivariateGaussian(std::unordered_map<std::string, int> variable_name_to_index_dict,
                            std::vector<double> means,
                            std::vector<double> covariances) : MultivariateGaussian{means, covariances} {
    if (variable_name_to_index_dict.size() != means.size()) {
      std::cout << "NamedMultivariateGaussian::Constructor: sizes of the provided variable_name_to_index_dict and means do not match."
                << std::endl;
    }
    assert(variable_name_to_index_dict.size() == means.size());
    this->variable_name_to_index_dict_ = variable_name_to_index_dict;
  }

  NamedMultivariateGaussian(std::unordered_map<std::string, int> variable_name_to_index_dict,
                            MultivariateGaussian multivariate_gaussian)
      : NamedMultivariateGaussian{variable_name_to_index_dict,
                                  EigenVector2Vector(multivariate_gaussian.mean()),
                                  CovarianceMatrixToCompactVector(multivariate_gaussian.covariance())} {
  }

  ~NamedMultivariateGaussian() {}

 private:
  std::unordered_map<std::string, int> variable_name_to_index_dict_;
};

}  // namespace distribution

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_DISTRIBUTION_GAUSSIAN_DISTRIBUTION_H_
