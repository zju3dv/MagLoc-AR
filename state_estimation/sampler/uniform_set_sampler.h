/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:36
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-01-18 16:07:18
 */
#ifndef STATE_ESTIMATION_SAMPLER_UNIFORM_SET_SAMPLER_H_
#define STATE_ESTIMATION_SAMPLER_UNIFORM_SET_SAMPLER_H_

#include <random>
#include <set>
#include <vector>
#include <ctime>

namespace state_estimation {

namespace sampler {

template <typename T>
class UniformSetSampler {
 public:
  void Init(std::set<T> population) {
    this->population_ = population;
    std::vector<T> population_vector;
    for (auto it = population.begin(); it != population.end(); it++) {
      population_vector.push_back(*it);
    }
    this->population_vector_ = population_vector;
    this->uniform_distribution_ =
        std::uniform_int_distribution<int>(0, population_vector.size() - 1);
  }

  T Sample(void) {
    int sample_index = this->uniform_distribution_(this->generator_);
    return this->population_vector_[sample_index];
  }

  void Seed(int random_seed) {
    this->generator_.seed(static_cast<uint64_t>(random_seed));
  }

  std::set<T> population(void) {
    return this->population_;
  }

  UniformSetSampler(void) {
    struct timespec tn;
    clock_gettime(CLOCK_REALTIME, &tn);
    this->generator_.seed(static_cast<uint64_t>(tn.tv_nsec));
  }

  ~UniformSetSampler() {}

 private:
  std::vector<T> population_vector_;
  std::set<T> population_;
  std::default_random_engine generator_;
  std::uniform_int_distribution<int> uniform_distribution_;
};

}  // namespace sampler

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_SAMPLER_UNIFORM_SET_SAMPLER_H_
