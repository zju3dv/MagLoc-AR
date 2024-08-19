/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2022-07-14 15:21:29
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-07-19 16:27:01
 */
#include "prediction_model/kalman_filter_model.h"

#include <cassert>

#include "util/misc.h"

namespace state_estimation {

namespace prediction_model {

void KalmanFilterModelState::Init(Eigen::VectorXd m, Eigen::MatrixXd P, std::vector<std::string> variable_names, std::vector<std::string> covariance_names) {
  assert(m.size() == P.cols());
  assert(P.cols() == P.rows());
  this->m_ = m;
  this->P_ = P;
  this->number_of_variables_ = m.size();
  this->variable_names_.clear();
  if (variable_names.size() != m.size()) {
    this->has_names_ = false;
  } else {
    this->has_names_ = true;
    this->variable_names_ = variable_names;
  }
  if (covariance_names.size() == P.size()) {
    this->covariance_names_ = covariance_names;
  }
}

void KalmanFilterModelState::Init(int number_of_variables) {
  assert(number_of_variables >= 0);
  this->number_of_variables_ = number_of_variables;
  this->m_.setZero(number_of_variables);
  this->P_.setZero(number_of_variables, number_of_variables);
  this->variable_names_.clear();
  this->has_names_ = false;
}

void KalmanFilterModelState::GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
  named_values->clear();
  std::string variable_name;
  for (int i = 0; i < this->m_.size(); i++) {
    if (this->has_names_) {
      variable_name = this->variable_names_.at(i);
    } else {
      variable_name = "KF_m_" + std::to_string(i);
    }
    named_values->emplace_back(variable_name, this->m_(i));
  }
  for (int i = 0; i < this->P_.rows(); i++) {
    for (int j = 0; j < this->P_.cols(); j++) {
      if (this->covariance_names_.size() == this->P_.size()) {
        variable_name = this->covariance_names_.at(i * this->P_.cols() + j);
      } else {
        variable_name = "KF_P_" + std::to_string(i) + std::to_string(j);
      }
      named_values->emplace_back(variable_name, this->P_(i, j));
    }
  }
}

std::string KalmanFilterModelState::ToKey(std::vector<double> discretization_resolutions) {
  return "null";
}

void KalmanFilterModelState::EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
  assert(sample_state_ptrs.size() == weights.size());
  NormalizeWeights(weights);
  Eigen::VectorXd expected_m;
  KalmanFilterModelState* my_state_ptr;
  Eigen::VectorXd current_m;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sample_matrix;
  Eigen::VectorXd weights_eigen_vector = Vector2EigenVector(weights);
  for (int i = 0; i < sample_state_ptrs.size(); i++) {
    my_state_ptr = reinterpret_cast<KalmanFilterModelState*>(sample_state_ptrs.at(i));
    current_m = my_state_ptr->m();
    if (i == 0) {
      expected_m = current_m * weights.at(i);
      sample_matrix.setZero(sample_state_ptrs.size(), current_m.size());
    } else {
      expected_m += (current_m * weights.at(i));
    }
    sample_matrix.row(i) = current_m;
  }
  this->m_ = expected_m;
  this->P_ = CalculateCovariance(sample_matrix, weights_eigen_vector);
  this->number_of_variables_ = expected_m.size();
}

void KalmanFilterModelState::Add(State* state_ptr) {
  return;
}

void KalmanFilterModelState::Subtract(State* state_ptr) {
  return;
}

void KalmanFilterModelState::Multiply_scalar(double scalar_value) {
  return;
}

void KalmanFilterModelState::Divide_scalar(double scalar_value) {
  return;
}

KalmanFilterModelState::KalmanFilterModelState(void) {
  this->m_ = Eigen::VectorXd();
  this->P_ = Eigen::MatrixXd();
  this->variable_names_ = std::vector<std::string>();
  this->covariance_names_ = std::vector<std::string>();
  this->has_names_ = false;
  this->number_of_variables_ = 0;
}

KalmanFilterModelState::~KalmanFilterModelState() {}

void KalmanFilterModelStateUncertainty::CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
  assert(sample_state_ptrs.size() == weights.size());
  NormalizeWeights(weights);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sample_matrix;
  KalmanFilterModelState* my_state_ptr;
  Eigen::VectorXd current_m;
  Eigen::VectorXd weights_eigen_vector = Vector2EigenVector(weights);
  for (int i = 0; i < sample_state_ptrs.size(); i++) {
    my_state_ptr = reinterpret_cast<KalmanFilterModelState*>(sample_state_ptrs.at(i));
    current_m = my_state_ptr->m();
    if (i == 0) {
      sample_matrix.setZero(sample_state_ptrs.size(), current_m.size());
    }
    sample_matrix.row(i) = current_m;
  }
  this->P_ = CalculateCovariance(sample_matrix, weights_eigen_vector);
  this->number_of_variables_ = sample_matrix.cols();
}

KalmanFilterModelStateUncertainty::KalmanFilterModelStateUncertainty(void) {
  this->P_ = Eigen::MatrixXd();
  this->number_of_variables_ = 0;
}

KalmanFilterModelStateUncertainty::~KalmanFilterModelStateUncertainty() {}

void KalmanFilterModelControlInput::GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
  return;
}

void KalmanFilterModelControlInput::Init(Eigen::MatrixXd A, Eigen::MatrixXd Q, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H, Eigen::MatrixXd R, Eigen::VectorXd y) {
  int n_state_variables = A.cols();
  int n_observation_variables = H.rows();
  assert(n_state_variables == A.rows());
  assert(n_state_variables == Q.cols());
  assert(n_state_variables == Q.rows());
  assert(n_state_variables == H.cols());
  assert(n_observation_variables == R.cols());
  assert(n_observation_variables == R.rows());
  assert(n_observation_variables == y.size());

  this->A_ = A;
  this->Q_ = Q;
  this->H_ = H;
  this->R_ = R;
  this->y_ = y;
  this->number_of_state_variables_ = n_state_variables;
  this->number_of_observation_variables_ = n_observation_variables;
}

KalmanFilterModelControlInput::KalmanFilterModelControlInput(void) {
  this->A_ = Eigen::MatrixXd();
  this->Q_ = Eigen::MatrixXd();
  this->H_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>();
  this->y_ = Eigen::VectorXd();
  this->R_ = Eigen::MatrixXd();
  this->number_of_state_variables_ = 0;
  this->number_of_observation_variables_ = 0;
}

KalmanFilterModelControlInput::~KalmanFilterModelControlInput() {}

void KalmanFilterModel::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction) {
  KalmanFilterModelState* my_state_t = reinterpret_cast<KalmanFilterModelState*>(state_t);
  KalmanFilterModelState* my_state_tminus = reinterpret_cast<KalmanFilterModelState*>(state_tminus);
  KalmanFilterModelControlInput* my_control_input_t = reinterpret_cast<KalmanFilterModelControlInput*>(control_input_t);

  Eigen::VectorXd m_tminus = my_state_tminus->m();
  Eigen::MatrixXd P_tminus = my_state_tminus->P();
  Eigen::MatrixXd A_tminus = my_control_input_t->A();
  Eigen::MatrixXd Q_tminus = my_control_input_t->Q();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_t = my_control_input_t->H();
  Eigen::VectorXd y_t = my_control_input_t->y();
  Eigen::MatrixXd R_t = my_control_input_t->R();

  int number_of_state_variables = my_control_input_t->number_of_state_variables();
  int number_of_observation_variables = my_control_input_t->number_of_observation_variables();
  assert(m_tminus.size() == number_of_state_variables);
  assert(P_tminus.cols() == number_of_state_variables);
  assert(A_tminus.cols() == number_of_state_variables);
  assert(Q_tminus.cols() == number_of_state_variables);
  assert(H_t.rows() == number_of_observation_variables);
  assert(H_t.cols() == number_of_state_variables);
  assert(y_t.size() == number_of_observation_variables);
  assert(R_t.cols() == number_of_observation_variables);

  Eigen::VectorXd m_t_pre = A_tminus * m_tminus;
  Eigen::MatrixXd P_t_pre = A_tminus * P_tminus * A_tminus.transpose() + Q_tminus;

  Eigen::VectorXd v_t = y_t - H_t * m_t_pre;
  Eigen::MatrixXd S_t = H_t * P_t_pre * H_t.transpose() + R_t;
  Eigen::MatrixXd K_t = P_t_pre * H_t.transpose() * S_t.inverse();
  Eigen::VectorXd m_t = m_t_pre + K_t * v_t;
  Eigen::MatrixXd P_t = P_t_pre - K_t * S_t * K_t.transpose();

  my_state_t->Init(m_t, P_t, my_state_tminus->variable_names());
}

void KalmanFilterModel::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  KalmanFilterModelState* my_state_t = reinterpret_cast<KalmanFilterModelState*>(state_t);
  KalmanFilterModelState* my_state_tminus = reinterpret_cast<KalmanFilterModelState*>(state_tminus);

  Eigen::VectorXd m_tminus = my_state_tminus->m();
  Eigen::MatrixXd P_tminus = my_state_tminus->P();

  my_state_t->Init(m_tminus, P_tminus, my_state_tminus->variable_names());
}

void KalmanFilterModel::Seed(int random_seed) {
  return;
}

double KalmanFilterModel::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  return 0.0;
}

void KalmanFilterModel::JitterState(State* state_t) {
  return;
}

KalmanFilterModel::KalmanFilterModel(void) {}
KalmanFilterModel::~KalmanFilterModel() {}

void KalmanFilterModelStateSampler::Init(Eigen::VectorXd m, Eigen::MatrixXd P) {
  assert(m.size() == P.cols());
  assert(P.cols() == P.rows());
  this->m_ = m;
  this->P_ = P;
  this->number_of_variables_ = m.size();
}

void KalmanFilterModelStateSampler::Sample(State* state_sample) {
  KalmanFilterModelState* my_state = reinterpret_cast<KalmanFilterModelState*>(state_sample);
  my_state->m(this->m_);
  my_state->P(this->P_);
}

void KalmanFilterModelStateSampler::Seed(int random_seed) {
  return;
}

double KalmanFilterModelStateSampler::CalculateStateProbabilityLog(State* state_sample) {
  return 0.0;
}

KalmanFilterModelStateSampler::KalmanFilterModelStateSampler(void) {
  this->m_ = Eigen::VectorXd();
  this->P_ = Eigen::MatrixXd();
  this->number_of_variables_ = 0;
}

KalmanFilterModelStateSampler::~KalmanFilterModelStateSampler() {}

}  // namespace prediction_model

}  // namespace state_estimation
