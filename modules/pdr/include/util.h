//
// Created by SENSETIME\zhaolinsheng on 2021/7/19.
//
#ifndef SENSE_PDR_UTIL_H
#define SENSE_PDR_UTIL_H

#include <string>
#include <vector>
#include <vector>
#include <deque>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <numeric>
#include <algorithm>
#include "Eigen/Dense"
//#define PI 3.1415926535897932384626433832795

//using namespace std;
//using namespace Eigen;
namespace pdr {

int test_add(int a, int b);
double quart_to_rpy(double w, double x, double y, double z);
std::vector<std::string>
string_split_by_char(const std::string &target_string, const char &split_char);
void split_string(const std::string &s,
                  std::vector<std::string> &v,
                  const std::string &c);
void vector_to_csv(std::vector<double> data, std::string CSV_PATH);
void vector_to_csv(std::vector<std::string> data, std::string CSV_PATH);

void vector_to_csv(std::vector<std::vector<double>> data, std::string CSV_PATH);
double compute_deque_mean(std::deque<double> input);
double compute_deque_std(std::deque<double> input);
void print_double_vector(std::vector<double> data);
double get_orientation_from_matrix(Eigen::Matrix3d rotationMatrix);

}

#endif //SENSE_PDR_UTIL_H
