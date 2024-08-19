/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-07-15 17:17:02
 * @LastEditTime: 2021-07-15 17:17:02
 * @LastEditors: xuehua
 */
#ifndef STATE_ESTIMATION_OFFLINE_SEQUENTIAL_READER_BASE_H_
#define STATE_ESTIMATION_OFFLINE_SEQUENTIAL_READER_BASE_H_

namespace state_estimation {

namespace offline {

class SequentialReader {
 public:
  virtual int GetSize(void) = 0;

  virtual bool IsEmpty(void) {
    return (this->cur_ == this->GetSize());
  }

  virtual int GetCurrentSize(void) {
    return (this->GetSize() - this->cur_);
  }

  virtual int SetCur(int cur) {
    if ((cur >= 0) && (cur < this->GetSize())) {
      this->cur_ = cur;
    }
    return this->cur_;
  }

 protected:
  int cur_ = 0;
};

}  // namespace offline

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OFFLINE_SEQUENTIAL_READER_BASE_H_
