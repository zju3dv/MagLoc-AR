/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:19:49
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-02-07 15:52:23
 */
#include "util/misc.h"

#include <dirent.h>
#include <Eigen/Dense>
#include <string.h>
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>

namespace state_estimation {

const double kZeroEpsilon = 1e-30;

void OrientationDecompositionRzRxy(const Eigen::Quaterniond& q, Eigen::Quaterniond* q_z, Eigen::Quaterniond* q_xy) {
  Eigen::Vector3d z_after = Eigen::Vector3d({0.0, 0.0, 1.0});
  Eigen::Vector3d z_before = q.conjugate() * z_after;
  *q_xy = Eigen::Quaterniond::FromTwoVectors(z_before, z_after).normalized();
  *q_z = (q * (*q_xy).conjugate()).normalized();
}

Eigen::Matrix3d CalculateRotationMatrixAlongAxisX(const double& omega_x) {
  Eigen::Matrix3d Rx = Eigen::Matrix3d::Zero();
  Rx(0, 0) = 1.0;
  Rx(1, 1) = std::cos(omega_x);
  Rx(1, 2) = -std::sin(omega_x);
  Rx(2, 1) = -Rx(1, 2);
  Rx(2, 2) = Rx(1, 1);
  return Rx;
}

Eigen::Matrix3d CalculateRotationMatrixAlongAxisY(const double& omega_y) {
  Eigen::Matrix3d Ry = Eigen::Matrix3d::Zero();
  Ry(1, 1) = 1.0;
  Ry(2, 2) = std::cos(omega_y);
  Ry(0, 0) = Ry(2, 2);
  Ry(0, 2) = std::sin(omega_y);
  Ry(2, 0) = -Ry(0, 2);
  return Ry;
}

Eigen::Matrix3d CalculateRotationMatrixAlongAxisZ(const double& omega_z) {
  Eigen::Matrix3d Rz = Eigen::Matrix3d::Zero();
  Rz(2, 2) = 1.0;
  Rz(0, 0) = std::cos(omega_z);
  Rz(0, 1) = -std::sin(omega_z);
  Rz(1, 0) = -Rz(0, 1);
  Rz(1, 1) = Rz(0, 0);
  return Rz;
}

Eigen::Quaterniond CalculateOrientationFromYawAndGravity(double yaw, Eigen::Vector3d gravity_s) {
  Eigen::Quaterniond q_sgs = Eigen::Quaterniond::FromTwoVectors(gravity_s, Eigen::Vector3d({0.0, 0.0, 1.0}));
  Eigen::AngleAxisd angleaxis_wsg(yaw, Eigen::Vector3d({0.0, 0.0, 1.0}));
  Eigen::Quaterniond q_wsg(angleaxis_wsg);
  return (q_wsg.normalized() * q_sgs.normalized()).normalized();
}

Eigen::Vector3d Spherical2Cartesian(Eigen::Vector3d r_theta_phi) {
  double r, theta, phi, x, y, z;
  r = r_theta_phi(0);
  theta = r_theta_phi(1);
  phi = r_theta_phi(2);
  x = r * std::sin(theta) * std::cos(phi);
  y = r * std::sin(theta) * std::sin(phi);
  z = r * std::cos(theta);
  return Eigen::Vector3d(x, y, z);
}

Eigen::Vector3d Cartesian2Spherical(Eigen::Vector3d xyz) {
  double r, theta, phi, x, y, z;
  x = xyz(0);
  y = xyz(1);
  z = xyz(2);
  r = std::sqrt(x * x + y * y + z * z);
  theta = std::acos(z / r);
  phi = std::atan2(y, x);
  return Eigen::Vector3d(r, theta, phi);
}

Eigen::Vector3d Quaternion2LogVector(Eigen::Quaterniond q) {
  Eigen::AngleAxisd angleaxis(q);
  return angleaxis.angle() * angleaxis.axis();
}

Eigen::Quaterniond LogVector2Quaternion(Eigen::Vector3d log_vector) {
  double log_norm = log_vector.norm();
  if (log_norm < kZeroEpsilon) {
    return Eigen::Quaterniond::Identity();
  } else {
    return Eigen::Quaterniond(Eigen::AngleAxisd(log_norm, log_vector / log_norm));
  }
}

std::vector<double> EigenVector2Vector(const Eigen::VectorXd& eigen_vector) {
  int n_variables = eigen_vector.size();
  std::vector<double> double_vector;
  for (int i = 0; i < n_variables; i++) {
    double_vector.push_back(eigen_vector(i));
  }
  return double_vector;
}

Eigen::VectorXd Vector2EigenVector(const std::vector<double>& double_vector) {
  int n_variables = double_vector.size();
  Eigen::VectorXd eigen_vector = Eigen::VectorXd::Zero(n_variables);
  for (int i = 0; i < n_variables; i++) {
    eigen_vector(i) = double_vector.at(i);
  }
  return eigen_vector;
}

Eigen::MatrixXd CalculateCovariance(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sample_matrix, Eigen::Matrix<double, Eigen::Dynamic, 1> weights) {
  // Columns of the sample_matrix represents different variables;
  // rows of the sample_matrix represents different samples;
  // the function calculates the covariance between different variables.
  if (weights.size() == 1) {
    weights = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(sample_matrix.rows(), 1);
    weights = weights / weights.sum();
  }
  assert(sample_matrix.rows() == weights.rows());
  assert(std::abs(weights.sum() - 1.0) < 1e-30);
  for (int i = 0; i < sample_matrix.cols(); i++) {
    sample_matrix.col(i) = (sample_matrix.col(i).array() - sample_matrix.col(i).mean()) * weights.array().sqrt();
  }
  return sample_matrix.transpose() * sample_matrix;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CompactVectorToCovarianceMatrix(const std::vector<double>& data_vector, int matrix_size) {
  assert(data_vector.size() == (matrix_size + 1) * matrix_size / 2);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(matrix_size, matrix_size);
  int cur = 0;
  for (int i = 0; i < matrix_size; i++) {
    for (int j = i; j < matrix_size; j++) {
      matrix(i, j) = data_vector.at(cur);
      if (i != j) {
        matrix(j, i) = data_vector.at(cur);
      }
      cur++;
    }
  }
  return matrix;
}

std::vector<double> CovarianceMatrixToCompactVector(Eigen::MatrixXd covariance) {
  int n_variables = covariance.cols();
  std::vector<double> compact_vector;
  for (int i = 0; i < n_variables; i++) {
    for (int j = i; j < n_variables; j++) {
      compact_vector.push_back(covariance(i, j));
    }
  }
  return compact_vector;
}

void NormalizeWeights(std::vector<double> &weights_vector) {
  double weights_sum = 0.0;
  for (int i = 0; i < weights_vector.size(); i++) {
    weights_sum += weights_vector.at(i);
  }
  assert(weights_sum > 0.0);
  for (int i = 0; i < weights_vector.size(); i++) {
    weights_vector.at(i) /= weights_sum;
  }
}

double UniqueZeroRound(double original_value, double round_interval) {
  double rounded_value = std::round(original_value / round_interval) * round_interval;
  if (std::abs(rounded_value) < (round_interval / 2.0)) {
    rounded_value = 0.0;
  }
  return rounded_value;
}

Eigen::Matrix<double, 3, 3> CalculateRzFromOrientation(Eigen::Quaterniond q) {
  Eigen::Matrix<double, 3, 1> z_vector = {0.0, 0.0, 1.0};
  Eigen::Matrix<double, 3, 3> R_zyx = q.normalized().toRotationMatrix();
  Eigen::Matrix<double, 3, 1> R_zyx_z = R_zyx.transpose() * z_vector;
  Eigen::Quaterniond q_yx = Eigen::Quaterniond::FromTwoVectors(R_zyx_z, z_vector);
  q_yx.normalize();
  Eigen::Matrix<double, 3, 3> R_z = R_zyx * q_yx.toRotationMatrix().transpose();
  return R_z;
}

double GetAngleByAxisFromAngleAxis(Eigen::AngleAxisd angle_axis, Eigen::Matrix<double, 3, 1> axis) {
  if (angle_axis.angle() < 1e-10) {
    return 0.0;
  }
  Eigen::Matrix<double, 3, 1> angle_axis_axis = angle_axis.axis();
  angle_axis_axis.normalize();
  axis.normalize();

  double angular_distance = angle_axis_axis.transpose() * axis;
  assert(1.0 - std::abs(angular_distance) < 1e-5);

  if (angular_distance >= 0.0) {
    return angle_axis.angle();
  } else {
    return -angle_axis.angle();
  }
}

Eigen::Quaterniond QuaternionGeometricMean(const std::vector<Eigen::Quaterniond> &qs, const std::vector<double> &weights) {
  // Calculate the mean of quaternions in the sense that minimize the summation of (sin(theta_i))^2,
  // where theta_i is the angular distance between the average quaternion and the i-th sample quaternion.
  double n = qs.size();
  Eigen::Matrix<double, Eigen::Dynamic, 1> weights_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(n, 1);
  if (weights.size() > 0) {
    assert(weights.size() == n);
    for (int i = 0; i < weights.size(); i++) {
      weights_vector(i) = weights.at(i);
    }
  }
  weights_vector = weights_vector.array() / weights_vector.sum();
  Eigen::Matrix<double, 4, Eigen::Dynamic> qs_matrix;
  qs_matrix.resize(4, n);
  Eigen::Quaterniond current_q;
  for (int i = 0; i < n; i++) {
    current_q = qs.at(i);
    current_q.normalize();
    qs_matrix(0, i) = current_q.w() * std::sqrt(weights_vector(i));
    qs_matrix(1, i) = current_q.x() * std::sqrt(weights_vector(i));
    qs_matrix(2, i) = current_q.y() * std::sqrt(weights_vector(i));
    qs_matrix(3, i) = current_q.z() * std::sqrt(weights_vector(i));
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 4, 4>> es(qs_matrix * qs_matrix.transpose());
  Eigen::Matrix<double, 4, 1> eigen_values = es.eigenvalues();
  double max_eigen_value;
  int max_eigen_value_index;
  for (int i = 0; i < eigen_values.size(); i++) {
    if (i > 0 && eigen_values(i) < max_eigen_value) {
      continue;
    }
    max_eigen_value = eigen_values(i);
    max_eigen_value_index = i;
  }
  Eigen::Matrix<double, 4, 1> sev = es.eigenvectors().col(max_eigen_value_index);
  Eigen::Quaterniond eigen_q(sev(0), sev(1), sev(2), sev(3));

  eigen_q.normalize();

  return eigen_q;
}

double QuaternionGeometricVariance(const std::vector<Eigen::Quaterniond> &qs, const std::vector<double> &weights) {
  // Calculate the variance of quaternions in the sense that the mean is minimizing the summation of (sin(theta_i))^2,
  // where theta_i is the angular distance between the average quaternion and the i-th sample quaternion.
  // the variance is calculated as the summation of (sin(theta_i))^2.
  double n = qs.size();
  Eigen::Matrix<double, Eigen::Dynamic, 1> weights_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(n, 1);
  if (weights.size() > 0) {
    assert(weights.size() == n);
    for (int i = 0; i < weights.size(); i++) {
      weights_vector(i) = weights.at(i);
    }
  }
  weights_vector = weights_vector.array() / weights_vector.sum();
  Eigen::Matrix<double, 4, Eigen::Dynamic> qs_matrix;
  qs_matrix.resize(4, n);
  Eigen::Quaterniond current_q;
  for (int i = 0; i < n; i++) {
    current_q = qs.at(i);
    current_q.normalize();
    qs_matrix(0, i) = current_q.w();
    qs_matrix(1, i) = current_q.x();
    qs_matrix(2, i) = current_q.y();
    qs_matrix(3, i) = current_q.z();
  }

  Eigen::Quaterniond mean_q = QuaternionGeometricMean(qs, weights);
  mean_q.normalize();
  Eigen::Matrix<double, 4, 1> mean_q_vector {mean_q.w(), mean_q.x(), mean_q.y(), mean_q.z()};

  return (1.0 - (qs_matrix.transpose() * mean_q_vector).array().pow(2.0)).matrix().transpose() * weights_vector;
}

Eigen::Matrix<double, 3, 3> QuaternionAngleAxisCovariance(const std::vector<Eigen::Quaterniond> &qs, const std::vector<double> &weights) {
  // Calculate the quaternion covariance matrix in the form of angle-axis.
  // TODO(xuehua): correct the calculation when angle_axis are represented with opposite directions.
  double n = qs.size();
  Eigen::Matrix<double, Eigen::Dynamic, 1> weights_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(n, 1);
  if (weights.size() > 0) {
    assert(weights.size() == n);
    for (int i = 0; i < weights.size(); i++) {
      weights_vector(i) = weights.at(i);
    }
  }
  weights_vector = weights_vector.array() / weights_vector.sum();
  Eigen::Quaterniond mean_q = QuaternionGeometricMean(qs, weights);
  Eigen::Matrix<double, 3, Eigen::Dynamic> angle_axises = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, n);
  for (int i = 0; i < n; i++) {
    Eigen::AngleAxisd angle_axis(mean_q * qs.at(i).conjugate());
    angle_axises(0, i) = angle_axis.angle() * angle_axis.axis()(0) * std::sqrt(weights_vector(i));
    angle_axises(1, i) = angle_axis.angle() * angle_axis.axis()(1) * std::sqrt(weights_vector(i));
    angle_axises(2, i) = angle_axis.angle() * angle_axis.axis()(2) * std::sqrt(weights_vector(i));
  }
  return angle_axises * angle_axises.transpose();
}

std::string DoubleToString(double value, int precision) {
  std::ostringstream ss;
  ss.setf(std::ios::fixed, std::ios::floatfield);
  ss.precision(precision);
  ss << value;
  return ss.str();
}

std::string ToLower(std::string strA) {
  transform(strA.begin(), strA.end(), strA.begin(), ::tolower);
  return strA;
}

double GetDoubleVectorSum(std::vector<double> input) {
  double sum = std::accumulate(std::begin(input), std::end(input), 0.0);
  return sum;
}

double GetDoubleVectorMean(std::vector<double> input) {
  if (input.size() == 0) {
    return 0;
  }
  double sum = std::accumulate(std::begin(input), std::end(input), 0.0);
  double mean = sum / input.size();
  return mean;
}

void TrimString(std::string& s) {
  while (!s.empty() &&
         (s[s.size() - 1] == '\r' ||
          s[s.size() - 1] == '\n' ||
          s[s.size() - 1] == ' ')) {
    s.erase(s.size() - 1);
  }
  while (!s.empty() &&
         (s[0] == '\r' ||
          s[0] == '\n' ||
          s[0] == ' ')) {
    s.erase(0, 1);
  }
}

void SplitString(const std::string& s,
                 std::vector<std::string>& v,
                 const std::string& c) {
  std::string s_copy = s;
  TrimString(s_copy);
  std::string::size_type pos1, pos2;
  pos2 = s_copy.find(c);
  pos1 = 0;
  v.clear();
  while (std::string::npos != pos2) {
    v.push_back(s_copy.substr(pos1, pos2 - pos1));
    pos1 = pos2 + c.size();
    pos2 = s_copy.find(c, pos1);
  }
  if (pos1 != s_copy.length())
    v.push_back(s_copy.substr(pos1));
}

void JoinString(const std::vector<std::string>& v, const std::string& c, std::string* s) {
  s->clear();
  for (int i = 0; i < v.size(); i++) {
    *s += v.at(i);
    if (i < (v.size() - 1)) {
      *s += c;
    }
  }
}

std::vector<std::string> GetLinesInFile(std::string filepath, int skip_lines) {
  std::vector<std::string> file_lines;
  std::ifstream the_file(filepath);
#ifdef VERBOSE
  if (!the_file) {
    std::cout << "state_estimation::GetLinesInFile: cannot load filepath: " << filepath << std::endl;
  }
#endif
  if (the_file) {
    std::string line;
    while (getline(the_file, line)) {
      if (line.empty()) {
        continue;
      }
      file_lines.push_back(line);
    }
    the_file.close();
  }
  return file_lines;
}

std::vector<std::string> GetFolderNamesInFolder(std::string folder_path) {
  std::vector<std::string> folder_names;
  DIR* dir = opendir(folder_path.c_str());
  if (dir == NULL) {
    std::cout << "opendir error! Path: " << folder_path << std::endl;
  }
  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_type == DT_DIR) {  // It's dir
      std::string folder_name = entry->d_name;
      if (folder_name != "." && folder_name != "..") {
        folder_names.push_back(folder_name);
      }
    } else {
      continue;
    }
  }
  closedir(dir);
  return folder_names;
}

void PrintPhysicalMemoryUsage() {
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];
  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      int len = static_cast<int>(strlen(line));
      const char* p = line;
      for (; std::isdigit(*p) == 0; ++p) {
      }
      line[len - 3] = 0;
      result = atoi(p);
      break;
    }
  }
  fclose(file);
  PrintHeading2("Memory used: " + std::to_string(result) + " kB");
}

void PrintHeading2(const std::string& heading) {
  std::cout << std::endl
            << heading << std::endl;
  std::cout << std::string(std::min<int>(heading.size(), 78), '-') << std::endl;
}

std::vector<std::pair<std::string, double>>
FindTopKLargestHeap(const std::vector<std::pair<std::string, double>>& input,
                    const int& k) {
  if (k > input.size()) {
    throw std::runtime_error("please input corrected K");
  }
  struct cmp {
    const double eps = 1e-200;
    bool operator()(std::pair<std::string, double> a,
                    std::pair<std::string, double> b) {
      return (a.second - b.second) > eps;
    }
  };

  std::vector<std::pair<std::string, double>> res;

  std::priority_queue<std::pair<std::string, double>,
                      std::vector<std::pair<std::string, double>>,
                      cmp>
      q;
  const double eps = 1e-200;

  for (int i = 0; i < input.size(); i++) {
    while (i < k) {
      i++;
      q.push(input[i]);
    }

    if (input[i].second - q.top().second > eps) {
      q.pop();
      q.push(input[i]);
    } else {
      continue;
    }
  }
  while (!q.empty()) {
    res.push_back(q.top());
    q.pop();
  }

  return res;
}

double GetTimeClosestLineIndex(
    const std::vector<std::string>& lines,
    int timestamp_index,
    int number_of_items_in_line,
    double target_timestamp,
    int* line_index) {
  if (lines.size() == 0) {
    *line_index = 0;
    return -1;
  }
  double previous_time_difference_abs = 1e50;
  for (int i = lines.size() - 1; i >= 0; i--) {
    std::string line = lines[i];
    std::vector<std::string> line_split;
    SplitString(line, line_split, ",");
    if (line_split.size() < number_of_items_in_line) {
      continue;
    } else {
      double timestamp = std::stoll(line_split[timestamp_index]);
      if (abs(timestamp - target_timestamp) >
          previous_time_difference_abs) {
        *line_index = i + 1;
        return previous_time_difference_abs;
      } else {
        previous_time_difference_abs = std::abs(timestamp - target_timestamp);
      }
    }
  }
  *line_index = 0;
  if (previous_time_difference_abs > 1e40) {
    return -1;
  } else {
    return previous_time_difference_abs;
  }
}

bool IsCovarianceMatrix(Eigen::MatrixXd covariance_matrix) {
#ifdef DEBUG_FOCUSING
  std::cout << "IsCovarianceMatrix(Eigen::MatrixXd covariance_matrix)" << std::endl;
#endif
  bool flag = true;
  if (covariance_matrix.rows() == covariance_matrix.cols()) {
    for (int i = 0; i < covariance_matrix.rows(); i++) {
      for (int j = i; j < covariance_matrix.cols(); j++) {
        if (std::abs(covariance_matrix(i, j) - covariance_matrix(j, i)) > 1e-8) {
          flag = false;
        }
      }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance_matrix);
    Eigen::VectorXd eigen_values = eigen_solver.eigenvalues();
    for (int i = 0; i < eigen_values.size(); i++) {
      if (eigen_values(i) < 0.0) {
        flag = false;
        break;
      }
    }
  } else {
    flag = false;
  }
#ifdef DEBUG_ISCOVARIANCEMATRIX
  if (!flag) {
    std::cout << "DEBUG_ISCOVARIANCEMATRIX: negative for covariance matrix check: " << std::endl;
    std::cout << covariance_matrix << std::endl;
  }
#endif
  return flag;
}

void MyTimer::Start(void) {
  struct timeval time;
  gettimeofday(&time, NULL);
  this->start_time_ = time.tv_sec + time.tv_usec * 1e-6;
  this->status_ = 1;
}

void MyTimer::Close(void) {
  this->start_time_ = 0.0;
  this->status_ = 0;
}

double MyTimer::TimePassed(void) {
  if (this->status_ == 0) {
    std::cout << "MyTimer::TimePassed: the timer has not been started yet." << std::endl;
    return 0.0;
  } else {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec + time.tv_usec * 1e-6 - this->start_time_);
  }
}

std::string MyTimer::TimePassedStr(void) {
  char buff[100];
  snprintf(buff, sizeof(buff), "%.10f", this->TimePassed());
  return std::string(buff);
}

}  // namespace state_estimation
