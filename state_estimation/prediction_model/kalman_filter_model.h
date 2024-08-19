/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2022-07-14 14:12:36
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-07-19 21:48:26
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_KALMAN_FILTER_MODEL_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_KALMAN_FILTER_MODEL_H_

#include <Eigen/Eigen>

#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "sampler/gaussian_sampler.h"

namespace state_estimation {

namespace prediction_model {

// All used symbols are referenced from the Sarkka's book.

class KalmanFilterModelState : public State {
 public:
  void Init(Eigen::VectorXd m, Eigen::MatrixXd P, std::vector<std::string> variable_names = std::vector<std::string>(), std::vector<std::string> covariance_names = std::vector<std::string>());
  void Init(int number_of_variables);
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const;
  std::string ToKey(std::vector<double> discretization_resolutions);
  void EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights);

  void Add(State* state_ptr);
  void Subtract(State* state_ptr);
  void Multiply_scalar(double scalar_value);
  void Divide_scalar(double scalar_value);

  Eigen::VectorXd m(void) const {
    return this->m_;
  }

  void m(Eigen::VectorXd m) {
    if (this->number_of_variables_ != m.size()) {
      std::cout << "KalmanFilterModelState::m(Eigen::VectorXd): "
                << "The size of provided m does not meet the number_of_variables. "
                << "Leave m unchanged." << std::endl;
      return;
    }
    this->m_ = m;
  }

  Eigen::MatrixXd P(void) const {
    return this->P_;
  }

  void P(Eigen::MatrixXd P) {
    if (this->number_of_variables_ != P.cols() || P.cols() != P.rows()) {
      std::cout << "KalmanFilterModelState::P(Eigen::MatrixXd): "
                << "The size of provided P does not meet the number_of_variables. "
                << "Leave P unchanged." << std::endl;
      return;
    }
    this->P_ = P;
  }

  std::vector<std::string> variable_names(void) const {
    return this->variable_names_;
  }

  std::vector<std::string> covariance_names(void) const {
    return this->covariance_names_;
  }

  int number_of_variables(void) {
    return this->number_of_variables_;
  }

  KalmanFilterModelState(void);
  ~KalmanFilterModelState();

 private:
  Eigen::VectorXd m_;
  Eigen::MatrixXd P_;
  std::vector<std::string> variable_names_;
  std::vector<std::string> covariance_names_;
  bool has_names_;
  int number_of_variables_;
};

class KalmanFilterModelStateUncertainty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights);

  KalmanFilterModelStateUncertainty(void);
  ~KalmanFilterModelStateUncertainty();

 private:
  Eigen::MatrixXd P_;
  int number_of_variables_;
};

class KalmanFilterModelControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const;
  void Init(Eigen::MatrixXd A, Eigen::MatrixXd Q, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H, Eigen::MatrixXd R, Eigen::VectorXd y);

  Eigen::MatrixXd A(void) {
    return this->A_;
  }

  Eigen::MatrixXd Q(void) {
    return this->Q_;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H(void) {
    return this->H_;
  }

  Eigen::VectorXd y(void) {
    return this->y_;
  }

  void y(Eigen::VectorXd y) {
    if (y.size() != this->number_of_observation_variables_) {
      std::cout << "KalmanFilterModelControlInput::y(Eigen::VectorXd): "
                << "Size of the provided y does not meet the number_of_observation_variables. "
                << "Leave y unchanged." << std::endl;
      return;
    }
    this->y_ = y;
  }

  Eigen::MatrixXd R(void) {
    return this->R_;
  }

  void R(Eigen::MatrixXd R) {
    if (R.cols() != R.rows() || R.cols() != this->number_of_observation_variables_) {
      std::cout << "KalmanFilterModelControlInput::R(Eigen::MatrixXd): "
                << "Size of the provided R does not meet the number_of_observation_variables. "
                << "Leave R unchanged." << std::endl;
      return;
    }
    this->R_ = R;
  }

  int number_of_state_variables(void) {
    return this->number_of_state_variables_;
  }

  int number_of_observation_variables(void) {
    return this->number_of_observation_variables_;
  }

  KalmanFilterModelControlInput(void);
  ~KalmanFilterModelControlInput();

 private:
  Eigen::MatrixXd A_;
  Eigen::MatrixXd Q_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_;
  Eigen::VectorXd y_;
  Eigen::MatrixXd R_;
  int number_of_state_variables_;
  int number_of_observation_variables_;
};

class KalmanFilterModel : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);
  void Seed(int random_seed);
  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);
  void JitterState(State* state_t);

  KalmanFilterModel(void);
  ~KalmanFilterModel();
};

class KalmanFilterModelStateSampler : public StateSampler {
 public:
  void Init(Eigen::VectorXd m, Eigen::MatrixXd P);
  void Sample(State* state_sample);
  void Seed(int random_seed);
  double CalculateStateProbabilityLog(State* state_sample);

  KalmanFilterModelStateSampler(void);
  ~KalmanFilterModelStateSampler();

 private:
  Eigen::VectorXd m_;
  Eigen::MatrixXd P_;
  int number_of_variables_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_KALMAN_FILTER_MODEL_H_
