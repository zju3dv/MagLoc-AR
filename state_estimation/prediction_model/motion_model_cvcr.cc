/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-09 10:50:51
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-17 11:32:54
 */
#include "prediction_model/motion_model_cvcr.h"

#include <Eigen/Core>
#include <iostream>
#include <random>
#include <mutex>

#include "prediction_model/base.h"
#include "util/lie_algebra.h"
#include "util/misc.h"
#include "variable/orientation.h"
#include "variable/position.h"

namespace state_estimation {

namespace prediction_model {

void MotionModelCVCR::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
#ifdef DEBUG_FOCUSING
  std::cout << "MotionModelCVCR::Predict" << std::endl;
#endif
#ifdef PREDICTION_MODEL_PREDICT_RUNTIME_PROFILE
  MyTimer timer;
  timer.Start();
#endif
  MotionModelCVCRState* my_state_t = reinterpret_cast<MotionModelCVCRState*>(state_t);
  MotionModelCVCRState* my_state_tminus = reinterpret_cast<MotionModelCVCRState*>(state_tminus);
  MotionModelCVCRControlInput* my_control_input_t = reinterpret_cast<MotionModelCVCRControlInput*>(control_input_t);

  // sample the control_input vector
  Eigen::Matrix<double, 5, 1> control_vector;
  {
  std::lock_guard<std::mutex> guard(this->my_mutex_);
  this->mvg_sampler_.SetParams(my_control_input_t->means(), my_control_input_t->covariances());
  control_vector = this->mvg_sampler_.Sample();
  }

  // do the state prediction according to the control_input vector and the state_tminus.
  // define symbols of tminus state variables;
  variable::Position position_tminus = my_state_tminus->position();
  Eigen::Matrix<double, 2, 1> v_tminus = my_state_tminus->v();
  variable::Orientation orientation_tminus = my_state_tminus->orientation();
  Eigen::AngleAxisd omega_tminus = my_state_tminus->omega();

  // define symbols of t control variables;
  Eigen::Matrix<double, 2, 1> acc_t = {control_vector(0), control_vector(1)};
  Eigen::Matrix<double, 3, 1> omega_acc_t = {control_vector(2), control_vector(3), control_vector(4)};

  // predict v
  my_state_t->v(v_tminus + acc_t * dt);

  // NOTICE: in the estimation of orientations, there are two different types of system settings.
  // 1. The prediciton of orientation is updated frame-by-frame.
  //    In this setting, we usually use the traditional perturbation model, since the orientation change between frames are very small.
  // 2. The prediciton of orientation is updated with respect to a fixed noticeable time period.
  //    In this setting, we need to model the orientation change by v_orientation and acc_orientation.

  // predict omega
  // lie-algebra
  // v_so3(t) ~= v_so3(t-1) + a_so3(t) * dt
  // v_SO3(t) ~= exp(JacobInverse(v_so3(t-1) * (a_so3(t) * dt) + v_so3(t-1)))
  // currently not using the lie-algebra form.
  Eigen::Vector3d omega_tminus_so3 = omega_tminus.angle() * omega_tminus.axis();
  Eigen::Vector3d omega_t_so3 = omega_tminus_so3 + omega_acc_t * dt;
  my_state_t->omega(Eigen::AngleAxisd(omega_t_so3.norm(), omega_t_so3 / omega_t_so3.norm()));

  // predict orientation
  // according to Taylor expansion (Second-order):
  //    Phi(t_0 + dt) = Phi(t_0) + Phi'(t_0) * dt + 0.5 * Phi"(t_0) * dt * dt
  // Phi(t_0) + Phi'(t_0) * dt can be calculated easily using v_orientation and the angle-axis representation;
  // the remaining 0.5 * Phi"(t_0) * dt * dt is a perturbation term.
  // lie-algebra
  // p_SO3_1st(t) = v_SO3(t-1) "*" dt * p_SO3(t-1))
  // p_SO3(t) ~= exp(JacobInverse(p_so3_1st(t)) * (0.5 * a_so3(t) * dt * dt) + p_so3_1st(t));
  variable::Orientation orientation_t;
  Eigen::Matrix<double, 3, 3> rotation_matrix_tminus = orientation_tminus.rotation_matrix();
  Eigen::Vector3d temp_omega_so3 = (omega_tminus_so3 + omega_t_so3) / 2.0;
  Eigen::Matrix<double, 3, 3> diff_rotation_matrix = Eigen::AngleAxisd(temp_omega_so3.norm() * dt, temp_omega_so3 / temp_omega_so3.norm()).toRotationMatrix();
  orientation_t.rotation_matrix(rotation_matrix_tminus * diff_rotation_matrix);
  my_state_t->orientation(orientation_t);

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

void MotionModelCVCR::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  MotionModelCVCRState* my_state_t = reinterpret_cast<MotionModelCVCRState*>(state_t);
  MotionModelCVCRState* my_state_tminus = reinterpret_cast<MotionModelCVCRState*>(state_tminus);

  // do the state prediction according to the state_tminus.
  // define symbols of tminus state variables;
  variable::Position position_tminus = my_state_tminus->position();
  Eigen::Matrix<double, 2, 1> v_tminus = my_state_tminus->v();
  variable::Orientation orientation_tminus = my_state_tminus->orientation();
  Eigen::AngleAxisd omega_tminus = my_state_tminus->omega();

  // predict v
  my_state_t->v(v_tminus);

  // predict omega
  my_state_t->omega(omega_tminus);

  // predict orientation
  variable::Orientation orientation_t;
  Eigen::Matrix<double, 3, 3> rotation_matrix_tminus = orientation_tminus.rotation_matrix();
  Eigen::Matrix<double, 3, 3> diff_rotation_matrix = Eigen::AngleAxisd(omega_tminus.angle() * dt, omega_tminus.axis()).toRotationMatrix();
  orientation_t.rotation_matrix(rotation_matrix_tminus * diff_rotation_matrix);
  my_state_t->orientation(orientation_t);

  // predict position
  variable::Position position_t;
  position_t.x(position_tminus.x() + v_tminus(0) * dt);
  position_t.y(position_tminus.y() + v_tminus(1) * dt);
  position_t.floor(position_tminus.floor());
  my_state_t->position(position_t);
}

double MotionModelCVCR::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  return 0.0;
}

}  // namespace prediction_model

}  // namespace state_estimation
