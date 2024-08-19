/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:57
 * @Last Modified by:   xuehua
 * @Last Modified time: 2020-12-24 16:18:57
 */
#ifndef STATE_ESTIMATION_VARIABLE_BASE_H_
#define STATE_ESTIMATION_VARIABLE_BASE_H_

#include <string>

namespace state_estimation {

namespace variable {

class Variable {
 public:
  // Key is used to map the value of the variable to
  // its correpsonding probability according to specified distribution.
  virtual std::string ToKey(void) = 0;
  virtual void FromKey(std::string variable_key) = 0;

  ~Variable(void) {}
};

}  // namespace variable

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_VARIABLE_BASE_H_
