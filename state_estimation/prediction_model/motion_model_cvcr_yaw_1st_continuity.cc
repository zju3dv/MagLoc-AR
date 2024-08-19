/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-09-13 20:33:52
 * @LastEditTime: 2021-09-13 20:39:35
 * @LastEditors: xuehua
 */
#include "prediction_model/motion_model_cvcr_yaw_1st_continuity.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <mutex>

#include "prediction_model/base.h"
#include "variable/position.h"
#include "distribution/gaussian_distribution.h"

namespace state_estimation {

namespace prediction_model {

void MotionModelCVCRYaw::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
#ifdef DEBUG_FOCUSING
  std::cout << "MotionModelCVCRYaw::Predict" << std::endl;
#endif
#ifdef PREDICTION_MODEL_PREDICT_RUNTIME_PROFILE
  MyTimer timer;
  timer.Start();
#endif
  MotionModelCVCRYawState* my_state_t = reinterpret_cast<MotionModelCVCRYawState*>(state_t);
  MotionModelCVCRYawState* my_state_tminus = reinterpret_cast<MotionModelCVCRYawState*>(state_tminus);
  MotionModelCVCRYawControlInput* my_control_input_t = reinterpret_cast<MotionModelCVCRYawControlInput*>(control_input_t);

  // sample the control_input vector
  Eigen::Matrix<double, 3, 1> control_vector;
  {
  std::lock_guard<std::mutex> guard(this->my_mutex_);
  this->mvg_sampler_.SetParams(my_control_input_t->means(), my_control_input_t->covariances());
  control_vector = this->mvg_sampler_.Sample();
  }

  // do the state prediction according to the control_input vector and the state_tminus.
  // define symbols of tminus state variables;
  variable::Position position_tminus = my_state_tminus->position();
  Eigen::Matrix<double, 2, 1> v_tminus = my_state_tminus->v();
  double yaw_tminus = my_state_tminus->yaw();

  // define symbols of t control variables;
  Eigen::Matrix<double, 2, 1> acc_t = {control_vector(0), control_vector(1)};
  double d_yaw_t = control_vector(2);

  // predict v
  my_state_t->v(v_tminus + acc_t * dt);

  // predict yaw
  // TODO(xuehua): currently, I assume that d_yaw_t is always in the time-forwarding sense.
  //  If dt is negative which means the system is running backward, d_yaw_t should be reversed.
  //  However, to some extent, this is not straight-forward.
  if (dt >= 0.0) {
    my_state_t->yaw(yaw_tminus + d_yaw_t);
  } else {
    my_state_t->yaw(yaw_tminus - d_yaw_t);
  }

  // predict position
  variable::Position position_t;
  position_t.x(position_tminus.x() + v_tminus(0) * dt + 0.5 * acc_t(0) * dt * dt);
  position_t.y(position_tminus.y() + v_tminus(1) * dt + 0.5 * acc_t(1) * dt * dt);
  position_t.floor(position_tminus.floor());
  my_state_t->position(position_t);

#ifdef PREDICTION_MODEL_PREDICT_RUNTIME_PROFILE
  std::cout << "MotionModelCVCR::Predict: time consumed in seconds: " << timer.TimePassedStr() << std::endl;
#endif
}

void MotionModelCVCRYaw::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  MotionModelCVCRYawState* my_state_t = reinterpret_cast<MotionModelCVCRYawState*>(state_t);
  MotionModelCVCRYawState* my_state_tminus = reinterpret_cast<MotionModelCVCRYawState*>(state_tminus);

  // do the state prediction according to the state_tminus.
  // define symbols of tminus state variables;
  variable::Position position_tminus = my_state_tminus->position();
  Eigen::Matrix<double, 2, 1> v_tminus = my_state_tminus->v();
  double yaw_tminus = my_state_tminus->yaw();

  // predict v
  my_state_t->v(v_tminus);

  // predict yaw
  my_state_t->yaw(yaw_tminus);

  // predict position
  variable::Position position_t;
  position_t.x(position_tminus.x() + v_tminus(0) * dt);
  position_t.y(position_tminus.y() + v_tminus(1) * dt);
  position_t.floor(position_tminus.floor());
  my_state_t->position(position_t);
}

double MotionModelCVCRYaw::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  MotionModelCVCRYawState* my_state_t = reinterpret_cast<MotionModelCVCRYawState*>(state_t);
  MotionModelCVCRYawState* my_state_tminus = reinterpret_cast<MotionModelCVCRYawState*>(state_tminus);
  MotionModelCVCRYawControlInput* my_control_input_t = reinterpret_cast<MotionModelCVCRYawControlInput*>(control_input_t);

  Eigen::Matrix<double, 2, 1> v_t = my_state_t->v();
  Eigen::Matrix<double, 2, 1> v_tminus = my_state_tminus->v();
  double yaw_tminus = my_state_tminus->yaw();
  double yaw_t = my_state_t->yaw();

  std::vector<double> control_vector = {((v_t - v_tminus) / dt)(0),
                                        ((v_t - v_tminus) / dt)(1),
                                        (yaw_t - yaw_tminus)};

  Eigen::Matrix<double, 3, 1> means = my_control_input_t->means();
  Eigen::Matrix<double, 3, 3> covariance = my_control_input_t->covariances();
  std::vector<double> mean_vector;
  std::vector<double> covariance_vector;
  for (int i = 0; i < means.size(); i++) {
    mean_vector.push_back(means(i));
  }
  for (int i = 0; i < covariance.rows(); i++) {
    for (int j = i; j < covariance.cols(); j++) {
      covariance_vector.push_back(covariance(i, j));
    }
  }
  distribution::MultivariateGaussian mvg_distribution(mean_vector, covariance_vector);

  return mvg_distribution.QuantizedProbability(control_vector);
}

}  // namespace prediction_model

}  // namespace state_estimation
