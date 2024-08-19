/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2022-03-09 20:22:39
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-03-11 11:08:20
 */
#include "prediction_model/motion_model_3d_orientation.h"

#include <cmath>
#include <mutex>

#include "distribution/gaussian_distribution.h"
#include "util/misc.h"

namespace state_estimation {

namespace prediction_model {

void MotionModel3dOrientationState::EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
  assert(sample_state_ptrs.size() == weights.size());
  NormalizeWeights(weights);
  std::vector<Eigen::Quaterniond> qs;
  MotionModel3dOrientationState* my_state_ptr;
  for (int i = 0; i < sample_state_ptrs.size(); i++) {
    my_state_ptr = reinterpret_cast<MotionModel3dOrientationState*>(sample_state_ptrs.at(i));
    qs.push_back(my_state_ptr->orientation());
  }

  Eigen::Quaterniond q_mean = variable::Orientation::Mean(qs, weights);

  this->orientation_ = q_mean;
}

void MotionModel3dOrientationStateUncertainty::CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
  assert(sample_state_ptrs.size() == weights.size());
  MotionModel3dOrientationState* my_state_ptr;
  std::vector<Eigen::Quaterniond> qs;

  for (int i = 0; i < sample_state_ptrs.size(); i++) {
    my_state_ptr = reinterpret_cast<MotionModel3dOrientationState*>(sample_state_ptrs.at(i));
    qs.push_back(my_state_ptr->orientation());
  }

  this->geometric_variance_ = variable::Orientation::Variance(qs, weights);
}

void MotionModel3dOrientationDifferentialControlInput::GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
  named_values->clear();
  named_values->emplace_back("dR_log_x", this->dR_log_(0));
  named_values->emplace_back("dR_log_y", this->dR_log_(1));
  named_values->emplace_back("dR_log_z", this->dR_log_(2));
  named_values->emplace_back("dR_log_error_cov_00", this->dR_log_error_cov_(0, 0));
  named_values->emplace_back("dR_log_error_cov_01", this->dR_log_error_cov_(0, 1));
  named_values->emplace_back("dR_log_error_cov_02", this->dR_log_error_cov_(0, 2));
  named_values->emplace_back("dR_log_error_cov_10", this->dR_log_error_cov_(1, 0));
  named_values->emplace_back("dR_log_error_cov_11", this->dR_log_error_cov_(1, 1));
  named_values->emplace_back("dR_log_error_cov_12", this->dR_log_error_cov_(1, 2));
  named_values->emplace_back("dR_log_error_cov_20", this->dR_log_error_cov_(2, 0));
  named_values->emplace_back("dR_log_error_cov_21", this->dR_log_error_cov_(2, 1));
  named_values->emplace_back("dR_log_error_cov_22", this->dR_log_error_cov_(2, 2));
}

void MotionModel3dOrientationDifferential::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
#ifdef DEBUG_FOCUSING
  std::cout << "Motion3dOrientationDifferential::Predict" << std::endl;
#endif
  MotionModel3dOrientationState* my_state_t = reinterpret_cast<MotionModel3dOrientationState*>(state_t);
  MotionModel3dOrientationState* my_state_tminus = reinterpret_cast<MotionModel3dOrientationState*>(state_tminus);
  MotionModel3dOrientationDifferentialControlInput* my_control_input_t = reinterpret_cast<MotionModel3dOrientationDifferentialControlInput*>(control_input_t);

  Eigen::Vector3d sample_dR_log_error = Eigen::Vector3d::Zero();
  {
  std::lock_guard<std::mutex> guard(this->my_mutex_);
  this->mvg_sampler_.SetParams(Eigen::Vector3d::Zero(), my_control_input_t->dR_log_error_cov());
  sample_dR_log_error = this->mvg_sampler_.Sample();
  }

  Eigen::Quaterniond sample_dR_q_error = LogVector2Quaternion(sample_dR_log_error);
  Eigen::Quaterniond dR_q = LogVector2Quaternion(my_control_input_t->dR_log());

  if (dt >= 0.0) {
    my_state_t->orientation(sample_dR_q_error * dR_q * my_state_tminus->orientation());
  } else {
    my_state_t->orientation(dR_q.conjugate() * sample_dR_q_error.conjugate() * my_state_tminus->orientation());
  }
}

void MotionModel3dOrientationDifferential::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  MotionModel3dOrientationState* my_state_t = reinterpret_cast<MotionModel3dOrientationState*>(state_t);
  MotionModel3dOrientationState* my_state_tminus = reinterpret_cast<MotionModel3dOrientationState*>(state_tminus);
  my_state_t->orientation(my_state_tminus->orientation());
}

double MotionModel3dOrientationDifferential::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  // TODO(xuehua):
  // for the current prediction function, the calculation of state_transition_probability is not trivial,
  // it relies on the calculation of marginal distribution of delta-orientation-1d.
  MotionModel3dOrientationState* my_state_t = reinterpret_cast<MotionModel3dOrientationState*>(state_t);
  MotionModel3dOrientationState* my_state_tminus = reinterpret_cast<MotionModel3dOrientationState*>(state_tminus);
  MotionModel3dOrientationDifferentialControlInput* my_control_input_t = reinterpret_cast<MotionModel3dOrientationDifferentialControlInput*>(control_input_t);

  std::vector<double> dR_log_error_means = {0.0, 0.0, 0.0};
  std::vector<double> dR_log_error_cov;
  for (int i = 0; i < 3; i++) {
    for (int j = i; j < 3; j++) {
      dR_log_error_cov.push_back(my_control_input_t->dR_log_error_cov()(i, j));
    }
  }
  distribution::MultivariateGaussian mvg_distribution(dR_log_error_means, dR_log_error_cov);

  Eigen::Quaterniond q_error = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond q_dR = LogVector2Quaternion(my_control_input_t->dR_log());
  if (dt >= 0.0) {
    q_error = my_state_t->orientation() * my_state_tminus->orientation().conjugate() * q_dR.conjugate();
  } else {
    q_error = my_state_tminus->orientation() * my_state_t->orientation().conjugate() * q_dR.conjugate();
  }
  Eigen::Vector3d sample_dR_log_error = Quaternion2LogVector(q_error);
  std::vector<double> x = {sample_dR_log_error(0), sample_dR_log_error(1), sample_dR_log_error(2)};
  return std::log(mvg_distribution.QuantizedProbability(x));
}

void MotionModel3dOrientationUniformStateSampler::Sample(State* state_sample) {
  MotionModel3dOrientationState* my_state_sample = reinterpret_cast<MotionModel3dOrientationState*>(state_sample);
  my_state_sample->orientation(this->orientation_sampler_.SampleByUniformOnAngleAxis(this->reference_orientation_,
                                                                                     this->max_angular_distance_,
                                                                                     this->max_angular_distance_,
                                                                                     this->max_angular_distance_));
}

double MotionModel3dOrientationUniformStateSampler::CalculateStateProbabilityLog(State* state_sample) {
  MotionModel3dOrientationState* my_state_sample = reinterpret_cast<MotionModel3dOrientationState*>(state_sample);
  Eigen::Quaterniond sample_q = my_state_sample->orientation();
  return this->orientation_sampler_.CalculateLogProbabilityByUniformOnAngleAxis(this->reference_orientation_,
                                                                                this->max_angular_distance_,
                                                                                this->max_angular_distance_,
                                                                                this->max_angular_distance_,
                                                                                sample_q);
}

void MotionModel3dOrientationGaussianStateSampler::Sample(State* state_sample) {
  MotionModel3dOrientationState* my_state_sample = reinterpret_cast<MotionModel3dOrientationState*>(state_sample);
  my_state_sample->orientation(this->gaussian_3d_orientation_sampler_.SampleByGaussianOnAngleAxis(this->q_mean_, this->covariance_));
}

double MotionModel3dOrientationGaussianStateSampler::CalculateStateProbabilityLog(State* state_sample) {
  MotionModel3dOrientationState* my_state_sample = reinterpret_cast<MotionModel3dOrientationState*>(state_sample);
  Eigen::Quaterniond sample_q = my_state_sample->orientation();
  return this->gaussian_3d_orientation_sampler_.CalculateLogProbabilityByGaussianOnAngleAxis(this->q_mean_, this->covariance_, sample_q);
}


}  // namespace prediction_model

}  // namespace state_estimation
