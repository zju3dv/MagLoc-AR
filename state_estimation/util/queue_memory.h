/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:18:48
 * @Last Modified by:   xuehua
 * @Last Modified time: 2020-12-24 16:18:48
 */
#ifndef STATE_ESTIMATION_UTIL_QUEUE_MEMORY_H_
#define STATE_ESTIMATION_UTIL_QUEUE_MEMORY_H_

#include <deque>
#include <iostream>

namespace state_estimation {

namespace util {

template <typename T>
class QueueMemory {
 public:
  void Init(int memory_size) {
    this->max_size_ = memory_size;
    this->current_size_ = 0;
    this->memory_.clear();
  }

  void Push(T memory_item) {
    if (this->GetMaxSize() == 0) {
      std::cout
          << "Please set a non-zero memory_size before pushing."
          << std::endl;
      return;
    }
    if (this->IsFull()) {
      this->memory_.pop_front();
      this->memory_.push_back(memory_item);
    } else {
      this->memory_.push_back(memory_item);
      this->current_size_ += 1;
    }
  }

  bool Pop(T* memory_item) {
    if (this->IsEmpty()) {
      return false;
    } else {
      *memory_item = this->memory_.back();
      this->memory_.pop_back();
      this->current_size_ -= 1;
      return true;
    }
  }

  bool PopN(T* memory_item, int number_of_last_items_to_pop) {
    // if number_of_last_items_to_pop equals one, it is the same as Pop.
    if (this->current_size_ < number_of_last_items_to_pop) {
      return false;
    } else {
      for (int i = 0; i < (number_of_last_items_to_pop - 1); i++) {
        this->memory_.pop_back();
      }
      *memory_item = this->memory_.back();
      this->memory_.pop_back();
      this->current_size_ -= number_of_last_items_to_pop;
      return true;
    }
  }

  void Clear(void) {
    this->memory_.clear();
    this->current_size_ = 0;
  }

  int GetCurrentSize(void) const {
    return this->current_size_;
  }

  int GetMaxSize(void) {
    return this->max_size_;
  }

  bool IsFull(void) {
    if (this->GetMaxSize() == this->GetCurrentSize()) {
      return true;
    } else {
      return false;
    }
  }

  bool IsEmpty(void) {
    if (this->GetCurrentSize() == 0) {
      return true;
    } else {
      return false;
    }
  }

  void GetMemory(std::deque<T>* memory_storage) {
    *memory_storage = this->memory_;
  }

  T& at(int i) {
    return this->memory_.at(i);
  }

  QueueMemory(void) {}
  ~QueueMemory() {}

 private:
  int current_size_ = 0;
  int max_size_ = 0;
  std::deque<T> memory_;
};

}  // namespace util

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_UTIL_QUEUE_MEMORY_H_
