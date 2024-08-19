/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:19:39
 * @Last Modified by:   xuehua
 * @Last Modified time: 2020-12-24 16:19:39
 */
#ifndef STATE_ESTIMATION_VARIABLE_POSITION_H_
#define STATE_ESTIMATION_VARIABLE_POSITION_H_

#include <iostream>
#include <string>

#include "variable/base.h"

namespace state_estimation {

namespace variable {

class Position : public Variable {
 public:
  std::string ToKey(void);
  void FromKey(std::string position_key);
  void Round(double spatial_interval);

  double x(void) const {
    return this->x_;
  }

  void x(double x) {
    this->x_ = x;
  }

  double y(void) const {
    return this->y_;
  }

  void y(double y) {
    this->y_ = y;
  }

  double z(void) const {
    return this->z_;
  }

  void z(double z) {
    this->z_ = z;
  }


  int floor(void) const {
    return this->floor_;
  }

  void floor(int floor) {
    this->floor_ = floor;
  }

  Position operator+(const Position& position) {
    Position temp_position;
    temp_position.x(this->x_ + position.x());
    temp_position.y(this->y_ + position.y());
    temp_position.z(this->z_ + position.z());
    temp_position.floor(this->floor_);
    return temp_position;
  }

  Position operator-(const Position& position) {
    Position temp_position;
    temp_position.x(this->x_ - position.x());
    temp_position.y(this->y_ - position.y());
    temp_position.z(this->z_ - position.z());
    temp_position.floor(this->floor_);
    return temp_position;
  }

  Position operator*(const Position& position) {
    Position temp_position;
    temp_position.x(this->x_ * position.x());
    temp_position.y(this->y_ * position.y());
    temp_position.z(this->z_ * position.z());
    temp_position.floor(this->floor_);
    return temp_position;
  }

  Position operator/(const Position& position) {
    double epsilon = 1e-6;
    Position temp_position;
    if ((std::abs(position.x()) < epsilon) || (std::abs(position.y()) < epsilon) || (std::abs(position.z()) < epsilon)) {
      std::cout << "Position::operator/: divided by zero." << std::endl;
    }
    temp_position.x(this->x_ / position.x());
    temp_position.y(this->y_ / position.y());
    temp_position.z(this->z_ / position.z());
    temp_position.floor(this->floor_);
    return temp_position;
  }

  Position operator*(const double scalar_value) {
    Position temp_position;
    temp_position.x(this->x_ * scalar_value);
    temp_position.y(this->y_ * scalar_value);
    temp_position.z(this->z_ * scalar_value);
    temp_position.floor(this->floor_);
    return temp_position;
  }

  Position operator/(const double scalar_value) {
    Position temp_position;
    temp_position.x(this->x_ / scalar_value);
    temp_position.y(this->y_ / scalar_value);
    temp_position.z(this->z_ / scalar_value);
    temp_position.floor(this->floor_);
    return temp_position;
  }

  Position(void);
  ~Position();

 private:
  double x_;
  double y_;
  double z_;
  int floor_;
};

}  // namespace variable

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_VARIABLE_POSITION_H_
