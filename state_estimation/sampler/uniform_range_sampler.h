/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:28
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-01-18 16:07:11
 */
#ifndef STATE_ESTIMATION_SAMPLER_UNIFORM_RANGE_SAMPLER_H_
#define STATE_ESTIMATION_SAMPLER_UNIFORM_RANGE_SAMPLER_H_

#include <iostream>
#include <random>
#include <ctime>

namespace state_estimation {

namespace sampler {

template <typename UniformDistribution, typename DataType>
class UniformRangeSampler {
 public:
  void Init(DataType min_value, DataType max_value) {
    if (max_value < min_value) {
      std::cout << "UniformRangeSampler::Init: "
                << "The provided min_value is larger than the max_value."
                << "Sampler Initialization fails."
                << std::endl;
      return;
    }
    this->min_value_ = min_value;
    this->max_value_ = max_value;
    this->uniform_distribution_ =
        UniformDistribution(min_value, max_value);
  }

  void SetRange(DataType min_value, DataType max_value) {
    this->Init(min_value, max_value);
  }

  DataType min_value(void) {
    return this->min_value_;
  }

  DataType max_value(void) {
    return this->max_value_;
  }

  DataType Sample(void) {
    return this->uniform_distribution_(this->generator_);
  }

  void Seed(int random_seed) {
    this->generator_.seed(static_cast<uint64_t>(random_seed));
  }

  UniformRangeSampler(void) {
    struct timespec tn;
    clock_gettime(CLOCK_REALTIME, &tn);
    this->generator_.seed(static_cast<uint64_t>(tn.tv_nsec));
  }
  ~UniformRangeSampler() {}

 private:
  DataType min_value_;
  DataType max_value_;
  std::default_random_engine generator_;
  UniformDistribution uniform_distribution_;
};

}  // namespace sampler

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_SAMPLER_UNIFORM_RANGE_SAMPLER_H_
