/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-04-27 15:03:29
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-04-27 15:40:57
 */
#include "prediction_model/motion_model_2d_displacement.h"

#include <cmath>
#include <iostream>
#include <mutex>

#include "distribution/gaussian_distribution.h"

namespace state_estimation {

namespace prediction_model {

void MotionModel2dDisplacement::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
#ifdef DEBUG_FOCUSING
  std::cout << "MotionModel2dDisplacement::Predict" << std::endl;
#endif
  MotionModel2dDisplacementState* my_state_t = reinterpret_cast<MotionModel2dDisplacementState*>(state_t);
  MotionModel2dDisplacementState* my_state_tminus = reinterpret_cast<MotionModel2dDisplacementState*>(state_tminus);
  MotionModel2dDisplacementControlInput* my_control_input_t = reinterpret_cast<MotionModel2dDisplacementControlInput*>(control_input_t);

  // sample the control_input vector
  Eigen::Matrix<double, 2, 1> dp_means;
  if (dt >= 0.0) {
    dp_means = my_control_input_t->dp_means();
  } else {
    dp_means = -my_control_input_t->dp_means();
  }
  Eigen::VectorXd control_vector;
  {
  std::lock_guard<std::mutex> guard(this->my_mutex_);
  this->mvg_sampler_.SetParams(dp_means, my_control_input_t->dp_covariances());
  control_vector = this->mvg_sampler_.Sample();
  }

  // predict positions
  variable::Position position_tminus = my_state_tminus->position();
  variable::Position position_t;
  position_t.x(position_tminus.x() + control_vector(0));
  position_t.y(position_tminus.y() + control_vector(1));
  position_t.floor(position_tminus.floor());
  my_state_t->position(position_t);
}

void MotionModel2dDisplacement::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  MotionModel2dDisplacementState* my_state_t = reinterpret_cast<MotionModel2dDisplacementState*>(state_t);
  MotionModel2dDisplacementState* my_state_tminus = reinterpret_cast<MotionModel2dDisplacementState*>(state_tminus);
  my_state_t->position(my_state_tminus->position());
}

double MotionModel2dDisplacement::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  MotionModel2dDisplacementState* my_state_t = reinterpret_cast<MotionModel2dDisplacementState*>(state_t);
  MotionModel2dDisplacementState* my_state_tminus = reinterpret_cast<MotionModel2dDisplacementState*>(state_tminus);
  MotionModel2dDisplacementControlInput* my_control_input_t = reinterpret_cast<MotionModel2dDisplacementControlInput*>(control_input_t);

  Eigen::Matrix<double, 2, 1> dp_means;
  if (dt >= 0.0) {
    dp_means = my_control_input_t->dp_means();
  } else {
    dp_means = -my_control_input_t->dp_means();
  }
  std::vector<double> dp_means_vector = {dp_means(0), dp_means(1)};
  std::vector<double> dp_covariance_vector = {my_control_input_t->dp_covariances()(0, 0), my_control_input_t->dp_covariances()(0, 1), my_control_input_t->dp_covariances()(1, 1)};
  distribution::MultivariateGaussian mvg(dp_means_vector, dp_covariance_vector);

  double dx, dy;
  dx = my_state_t->position().x() - my_state_tminus->position().x();
  dy = my_state_t->position().y() - my_state_tminus->position().y();
  std::vector<double> x = {dx, dy};

  return std::log(mvg.QuantizedProbability(x));
}

}  // namespace prediction_model

}  // namespace state_estimation
