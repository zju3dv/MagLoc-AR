//
// Created by SENSETIME\zhaolinsheng on 2021/7/19.
//
#include "util.h"
namespace pdr {

int32_t test_add(int32_t a, int32_t b) {
  return a + b;
}

double quart_to_rpy(double w, double x, double y, double z) {
  double r, p, yaw;
  r = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  //pitch
  if (std::abs(2 * (w * y - z * x)) >= 1)
    p = copysign(M_PI / 2, 2 * (w * y - z * x));
  else
    p = asin(2 * (w * y - z * x));
  yaw = atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y));
  return yaw;
}


std::vector<std::string>
string_split_by_char(const std::string &target_string, const char &split_char) {
  std::vector<int32_t> token_indices;
  for (int32_t i = 0; i < target_string.size(); ++i) {
    if (target_string[i] == split_char) {
      token_indices.push_back(i);
    }
  }
  token_indices.push_back(target_string.size());
  // fill in tokens
  std::vector<std::string> res;
  for (int32_t i = 0, l = 0; i < token_indices.size(); ++i) {
    res.emplace_back(target_string.substr(l, token_indices[i] - l));
    l = token_indices[i] + 1;
  }
  return res;
}

void split_string(const std::string &s,
                  std::vector<std::string> &v,
                  const std::string &c) {
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  v.clear();
  while (std::string::npos != pos2) {
    v.push_back(s.substr(pos1, pos2 - pos1));
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length())
    v.push_back(s.substr(pos1));
}

void vector_to_csv(std::vector<double> data, std::string CSV_PATH) {
  std::ofstream OpenFile(CSV_PATH);
  if (OpenFile.fail()) {
    std::cout << "打开文件错误!" << std::endl;
  }
  for (int i = 0; i < data.size(); i++) {
    OpenFile << data[i] << "\n";
  }
  OpenFile.close();
}


void vector_to_csv(std::vector<std::string> data, std::string CSV_PATH) {
  std::ofstream OpenFile(CSV_PATH);
  if (OpenFile.fail()) {
    std::cout << "打开文件错误!" << std::endl;
  }
  for (int i = 0; i < data.size(); i++) {
    OpenFile << data[i] << "\n";
  }
  OpenFile.close();
}

void
vector_to_csv(std::vector<std::vector<double>> data, std::string CSV_PATH) {
  std::ofstream OpenFile(CSV_PATH);
  if (OpenFile.fail()) {
    std::cout << "打开文件错误!" << std::endl;
  }

  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].size() - 1; ++j) {
      OpenFile << data[i][j] << ',';
    }
    OpenFile << data[i][data[i].size() - 1] << "\n";
  }

  OpenFile.close();
}

double compute_deque_mean(std::deque<double> input) {
  if (input.size() == 0) {
    return 0;
  }
  double sum = std::accumulate(std::begin(input), std::end(input), 0.0);
  double mean = sum / input.size();
  return mean;
}

double compute_deque_std(std::deque<double> input) {
  if (input.size() == 0) {
    return 0;
  }
  double mean = compute_deque_mean(input);
  double std_sum = 0;
  for (int i = 0; i < input.size(); i++) {
    std_sum += pow(input[i] - mean, 2);
  }
  return sqrt(std_sum / input.size());

}

void print_double_vector(std::vector<double> input) {
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i] << std::endl;
  }
}

double get_orientation_from_matrix(Eigen::Matrix3d rotationMatrix) {
  std::vector<double> res;
  double yaw, pitch, roll;
//rotationMatrix.data();
  std::vector<double> flat_R;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      //cout<< rotationMatrix(i,j)<<endl;
      flat_R.push_back(rotationMatrix(i, j));
    }
  }

  yaw = atan2(flat_R[1], flat_R[4]);
  pitch = asin(-flat_R[7]);
  roll = atan2(-flat_R[6], flat_R[8]);
  res.push_back(yaw);
  res.push_back(pitch);
  res.push_back(roll);
  return yaw;
}

}//namespace pdr
