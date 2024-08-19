/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-11-02 10:40:19
 * @LastEditTime: 2021-11-02 10:40:19
 * @LastEditors: xuehua
 */
#include "prediction_model/motion_model_yaw_differential.h"

#include <cmath>
#include <mutex>

#include "distribution/gaussian_distribution.h"
#include "util/misc.h"

namespace state_estimation {

namespace prediction_model {

void MotionModelYawDifferentialState::EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
  assert(sample_state_ptrs.size() == weights.size());
  NormalizeWeights(weights);
  std::vector<Eigen::Quaterniond> qs;
  MotionModelYawDifferentialState* my_state_ptr;
  Eigen::Matrix<double, 3, 1> z_vector = {0.0, 0.0, 1.0};
  for (int i = 0; i < sample_state_ptrs.size(); i++) {
    my_state_ptr = reinterpret_cast<MotionModelYawDifferentialState*>(sample_state_ptrs.at(i));
    qs.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(my_state_ptr->yaw(), z_vector)));
  }

  Eigen::AngleAxisd mean_yaw_angle_axis(variable::Orientation::Mean(qs, weights));
  assert(mean_yaw_angle_axis.axis()(0) < 1e-5);
  assert(mean_yaw_angle_axis.axis()(1) < 1e-5);
  double mean_yaw;
  if (mean_yaw_angle_axis.axis()(2) >= 0.0) {
    mean_yaw = mean_yaw_angle_axis.angle();
  } else {
    mean_yaw = -mean_yaw_angle_axis.angle();
  }

  this->yaw_ = mean_yaw;
}

void MotionModelYawDifferentialStateUncertainty::CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
  assert(sample_state_ptrs.size() == weights.size());
  MotionModelYawDifferentialState* my_state_ptr;
  std::vector<Eigen::Quaterniond> qs;
  Eigen::Matrix<double, 3, 1> z_vector = {0.0, 0.0, 1.0};

  for (int i = 0; i < sample_state_ptrs.size(); i++) {
    my_state_ptr = reinterpret_cast<MotionModelYawDifferentialState*>(sample_state_ptrs.at(i));
    qs.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(my_state_ptr->yaw(), z_vector)));
  }

  this->yaw_variance_ = variable::Orientation::Variance(qs, weights);
}

void MotionModelYawDifferentialControlInput::GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
  named_values->clear();
  named_values->emplace_back("rotation_matrix_00", this->dR_(0, 0));
  named_values->emplace_back("rotation_matrix_01", this->dR_(0, 1));
  named_values->emplace_back("rotation_matrix_02", this->dR_(0, 2));
  named_values->emplace_back("rotation_matrix_10", this->dR_(1, 0));
  named_values->emplace_back("rotation_matrix_11", this->dR_(1, 1));
  named_values->emplace_back("rotation_matrix_12", this->dR_(1, 2));
  named_values->emplace_back("rotation_matrix_20", this->dR_(2, 0));
  named_values->emplace_back("rotation_matrix_21", this->dR_(2, 1));
  named_values->emplace_back("rotation_matrix_22", this->dR_(2, 2));
  named_values->emplace_back("rotation_log_cov_00", this->dR_error_log_cov_(0, 0));
  named_values->emplace_back("rotation_log_cov_01", this->dR_error_log_cov_(0, 1));
  named_values->emplace_back("rotation_log_cov_02", this->dR_error_log_cov_(0, 2));
  named_values->emplace_back("rotation_log_cov_10", this->dR_error_log_cov_(1, 0));
  named_values->emplace_back("rotation_log_cov_11", this->dR_error_log_cov_(1, 1));
  named_values->emplace_back("rotation_log_cov_12", this->dR_error_log_cov_(1, 2));
  named_values->emplace_back("rotation_log_cov_20", this->dR_error_log_cov_(2, 0));
  named_values->emplace_back("rotation_log_cov_21", this->dR_error_log_cov_(2, 1));
  named_values->emplace_back("rotation_log_cov_22", this->dR_error_log_cov_(2, 2));
}

void MotionModelYawDifferential::Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt, bool ideal_prediction) {
#ifdef DEBUG_FOCUSING
  std::cout << "MotionModelYawDifferential::Predict" << std::endl;
#endif
  MotionModelYawDifferentialState* my_state_t = reinterpret_cast<MotionModelYawDifferentialState*>(state_t);
  MotionModelYawDifferentialState* my_state_tminus = reinterpret_cast<MotionModelYawDifferentialState*>(state_tminus);
  MotionModelYawDifferentialControlInput* my_control_input_t = reinterpret_cast<MotionModelYawDifferentialControlInput*>(control_input_t);

  Eigen::Vector3d means = Eigen::Vector3d::Zero();

  Eigen::Vector3d sample_dR_error_log = Eigen::Vector3d::Zero();
  if (ideal_prediction) {
    sample_dR_error_log = means;
  } else {
    std::lock_guard<std::mutex> guard(this->my_mutex_);
    this->mvg_sampler_.SetParams(means, my_control_input_t->dR_error_log_cov());
    sample_dR_error_log = this->mvg_sampler_.Sample();
  }
  double sample_dR_error_angle = sample_dR_error_log.norm();
  Eigen::AngleAxisd sample_dR_error_angleaxis;
  if (std::abs(sample_dR_error_angle) < 1e-8) {
    sample_dR_error_angleaxis = Eigen::AngleAxisd(0.0, Eigen::Vector3d({0.0, 0.0, 1.0}));
  } else {
    sample_dR_error_angleaxis = Eigen::AngleAxisd(sample_dR_error_angle, sample_dR_error_log / sample_dR_error_angle);
  }

  Eigen::Matrix3d R_ws_tminus = my_state_tminus->q_ws().toRotationMatrix();

  Eigen::Matrix3d R_ws_t;
  if (dt >= 0.0) {
    R_ws_t = sample_dR_error_angleaxis * my_control_input_t->dR() * R_ws_tminus;
  } else {
    R_ws_t = my_control_input_t->dR().transpose() * sample_dR_error_angleaxis.inverse() * R_ws_tminus;
  }

  Eigen::AngleAxisd angleaxis_wsg_t_predicted(R_ws_t * my_control_input_t->q_sgs().normalized().conjugate());

  // only keep the z-axis component of R_wsg_t_temp
  double yaw_t = (angleaxis_wsg_t_predicted.angle() * angleaxis_wsg_t_predicted.axis())(2);

  my_state_t->q_sgs(my_control_input_t->q_sgs());
  my_state_t->yaw(yaw_t);

  // calculate state_prediction_log_probability for the next state.
  std::vector<double> sample_dR_error_log_vector = EigenVector2Vector(sample_dR_error_log);
  std::vector<double> dR_error_log_mean_vector = EigenVector2Vector(means);
  std::vector<double> dR_error_log_covariance_vector = CovarianceMatrixToCompactVector(my_control_input_t->dR_error_log_cov());
  distribution::MultivariateGaussian mvg_local_rotation(dR_error_log_mean_vector, dR_error_log_covariance_vector);
  my_state_t->state_prediction_log_probability(my_state_tminus->state_prediction_log_probability() + std::log(mvg_local_rotation.QuantizedProbability(sample_dR_error_log_vector)));
  my_state_t->state_update_log_probability(my_state_tminus->state_update_log_probability());
}

void MotionModelYawDifferential::PredictWithoutControlInput(State* state_t, State* state_tminus, double dt) {
  MotionModelYawDifferentialState* my_state_t = reinterpret_cast<MotionModelYawDifferentialState*>(state_t);
  MotionModelYawDifferentialState* my_state_tminus = reinterpret_cast<MotionModelYawDifferentialState*>(state_tminus);
  my_state_t->yaw(my_state_tminus->yaw());
}

double MotionModelYawDifferential::CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt) {
  // TODO(xuehua):
  // for the current prediction function, the calculation of state_transition_probability is not trivial,
  // it relies on the calculation of marginal distribution of delta-orientation-1d.
  return std::log(1.0);
}

void MotionModelYawDifferential::JitterState(State* state_t) {
  MotionModelYawDifferentialState* my_state_t = reinterpret_cast<MotionModelYawDifferentialState*>(state_t);
  double yaw_t = my_state_t->yaw();
  double yaw_jitter_delta = this->yaw_jitter_uvg_sampler_.Sample();
  double yaw_jitter_mean = this->yaw_jitter_uvg_sampler_.mean();
  double yaw_jitter_stddev = this->yaw_jitter_uvg_sampler_.stddev();
  distribution::UnivariateGaussian uvg_yaw_jitter(yaw_jitter_mean, std::pow(yaw_jitter_stddev, 2.0));

  my_state_t->yaw(yaw_t + yaw_jitter_delta);
  my_state_t->state_prediction_log_probability(my_state_t->state_prediction_log_probability() + std::log(uvg_yaw_jitter.QuantizedProbability(yaw_jitter_delta)));
}

void MotionModelYawDifferentialUniformStateSampler::Sample(State* state_sample) {
  MotionModelYawDifferentialState* my_state_sample = reinterpret_cast<MotionModelYawDifferentialState*>(state_sample);
  my_state_sample->yaw(this->yaw_sampler_.Sample());
}

double MotionModelYawDifferentialUniformStateSampler::CalculateStateProbabilityLog(State* state_sample) {
  MotionModelYawDifferentialState* my_state_sample = reinterpret_cast<MotionModelYawDifferentialState*>(state_sample);
  double yaw = my_state_sample->yaw();
  double yaw_min = this->yaw_sampler_.min_value();
  double yaw_max = this->yaw_sampler_.max_value();

  double log_prob = 0.0;
  if (std::abs(yaw_max - yaw_min) < 1e-10) {
    if (std::abs(yaw_max - yaw) < 1e-10) {
      log_prob += std::log(1.0);
    } else {
      log_prob += std::log(0.0);
    }
  } else {
    if ((yaw >= yaw_min - 1e-10) && (yaw <= yaw_max + 1e-10)) {
      log_prob += std::log(1.0 / (yaw_max - yaw_min));
    } else {
      log_prob += std::log(0.0);
    }
  }

  return log_prob;
}

void MotionModelYawDifferentialGaussianStateSampler::Sample(State* state_sample) {
  MotionModelYawDifferentialState* my_state_sample = reinterpret_cast<MotionModelYawDifferentialState*>(state_sample);
  my_state_sample->yaw(this->univariate_gaussian_sampler_.Sample());
}

double MotionModelYawDifferentialGaussianStateSampler::CalculateStateProbabilityLog(State* state_sample) {
  MotionModelYawDifferentialState* my_state_sample = reinterpret_cast<MotionModelYawDifferentialState*>(state_sample);
  double yaw = my_state_sample->yaw();

  distribution::UnivariateGaussian uvg_distribution(this->yaw_mean_, this->yaw_variance_);
  return std::log(uvg_distribution.QuantizedProbability(yaw));
}


}  // namespace prediction_model

}  // namespace state_estimation
