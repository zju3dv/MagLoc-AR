/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:20:03
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-02-07 15:44:07
 */
#ifndef STATE_ESTIMATION_UTIL_MISC_H_
#define STATE_ESTIMATION_UTIL_MISC_H_

#include <Eigen/Dense>

#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace state_estimation {

void OrientationDecompositionRzRxy(const Eigen::Quaterniond& q, Eigen::Quaterniond* q_z, Eigen::Quaterniond* q_xy);

Eigen::Matrix3d CalculateRotationMatrixAlongAxisX(const double& omega_x);

Eigen::Matrix3d CalculateRotationMatrixAlongAxisY(const double& omega_y);

Eigen::Matrix3d CalculateRotationMatrixAlongAxisZ(const double& omega_z);

Eigen::Quaterniond CalculateOrientationFromYawAndGravity(double yaw, Eigen::Vector3d gravity_s);

Eigen::Vector3d Spherical2Cartesian(Eigen::Vector3d r_theta_phi);

Eigen::Vector3d Cartesian2Spherical(Eigen::Vector3d xyz);

Eigen::Vector3d Quaternion2LogVector(Eigen::Quaterniond q);

Eigen::Quaterniond LogVector2Quaternion(Eigen::Vector3d log_vector);

std::vector<double> EigenVector2Vector(const Eigen::VectorXd& eigen_vector);

Eigen::VectorXd Vector2EigenVector(const std::vector<double>& double_vector);

Eigen::MatrixXd CalculateCovariance(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sample_matrix, Eigen::Matrix<double, Eigen::Dynamic, 1> weights = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(1, 1));

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> VectorTo2DMatrixC(std::vector<T> data_vector, int n_rows) {
  if (data_vector.size() == 0 || n_rows <= 0 || (data_vector.size() % n_rows) != 0) {
    std::cout << "state_estimation::VectorTo2DMatrixC: size mismatch." << std::endl;
    assert(!(data_vector.size() == 0 || n_rows <= 0 || (data_vector.size() % n_rows) != 0));
  }
  assert(data_vector.size() % n_rows == 0);
  int n_cols = data_vector.size() / n_rows;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_rows, n_cols);
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      matrix(i, j) = data_vector.at(i * n_cols + j);
    }
  }

  return matrix;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CompactVectorToCovarianceMatrix(const std::vector<double>& data_vector, int matrix_size);

std::vector<double> CovarianceMatrixToCompactVector(Eigen::MatrixXd covariance);

void NormalizeWeights(std::vector<double> &weights_vector);

double UniqueZeroRound(double original_value, double round_interval);

Eigen::Matrix<double, 3, 3> CalculateRzFromOrientation(Eigen::Quaterniond q);
double GetAngleByAxisFromAngleAxis(Eigen::AngleAxisd angle_axis, Eigen::Matrix<double, 3, 1> axis);

Eigen::Quaterniond QuaternionGeometricMean(const std::vector<Eigen::Quaterniond> &qs, const std::vector<double> &weights);
double QuaternionGeometricVariance(const std::vector<Eigen::Quaterniond> &qs, const std::vector<double> &weights);
Eigen::Matrix<double, 3, 3> QuaternionAngleAxisCovariance(const std::vector<Eigen::Quaterniond> &qs, const std::vector<double> &weights);

double GetDoubleVectorSum(std::vector<double> input);
double GetDoubleVectorMean(std::vector<double> input);

std::string DoubleToString(double value, int precision);
std::string ToLower(std::string strA);

// trim string
void TrimString(std::string& s);

// split string
void SplitString(const std::string& s,
                 std::vector<std::string>& v,
                 const std::string& c);

// join string
void JoinString(const std::vector<std::string>& v, const std::string& c, std::string* s);

// file io utils
std::vector<std::string> GetLinesInFile(std::string filepath, int skip_lines = 0);
std::vector<std::string> GetFolderNamesInFolder(std::string folderpath);

// Print the MemoryUsage
void PrintPhysicalMemoryUsage();

// Print second-order heading with underscores to `std::cout`.
void PrintHeading2(const std::string& heading);

std::vector<std::pair<std::string, double>>
FindTopKLargestHeap(const std::vector<std::pair<std::string, double>>& input,
                    const int& k);

// get closest timestamp-included record line from lines.
double GetTimeClosestLineIndex(const std::vector<std::string>& lines,
                               int timestamp_index,
                               int number_of_items_in_line,
                               double target_timestamp,
                               int* line_index);

// check the covariance matrix
bool IsCovarianceMatrix(Eigen::MatrixXd covariance_matrix);

// Get the permutation of all sub_vectors
template <typename Element>
void Permutate(std::vector<std::vector<Element>>* permutated_vector_ptr, const std::vector<std::vector<Element>>& variable_vectors, int variable_index) {
  if (variable_index >= variable_vectors.size()) {
    return;
  }
  std::vector<Element> temp_vector;
  int permutated_vector_size = permutated_vector_ptr->size();
  if (permutated_vector_size > 0) {
    for (int i = 1; i < variable_vectors.at(variable_index).size(); i++) {
      for (int j = 0; j < permutated_vector_size; j++) {
        temp_vector = permutated_vector_ptr->at(j);
        temp_vector.push_back(variable_vectors.at(variable_index).at(i));
        permutated_vector_ptr->push_back(temp_vector);
      }
    }
    if (variable_vectors.at(variable_index).size() > 0) {
      for (int i = 0; i < permutated_vector_size; i++) {
        permutated_vector_ptr->at(i).push_back(variable_vectors.at(variable_index).at(0));
      }
    }
  } else {
    for (int i = 0; i < variable_vectors.at(variable_index).size(); i++) {
      permutated_vector_ptr->push_back(std::vector<Element>({variable_vectors.at(variable_index).at(i)}));
    }
  }
  Permutate(permutated_vector_ptr, variable_vectors, variable_index + 1);
}

class MyTimer {
 public:
  void Start(void);
  void Close(void);
  double TimePassed(void);
  std::string TimePassedStr(void);

  MyTimer(void) {
    this->start_time_ = 0.0;
    this->status_ = 0;
  }

  ~MyTimer() {}

 private:
  double start_time_;
  int status_;  // 1 for running
};

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_UTIL_MISC_H_
