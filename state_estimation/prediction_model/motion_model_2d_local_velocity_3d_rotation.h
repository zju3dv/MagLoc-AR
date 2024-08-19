/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-12-22 10:49:49
 * @LastEditTime: 2021-12-22 10:50:21
 * @LastEditors: xuehua
 */
#ifndef STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_LOCAL_VELOCITY_3D_ROTATION_H_
#define STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_LOCAL_VELOCITY_3D_ROTATION_H_

#include <Eigen/Dense>

#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "prediction_model/base.h"
#include "prediction_model/motion_model_3d_orientation.h"
#include "sampler/gaussian_sampler.h"
#include "sampler/uniform_range_sampler.h"
#include "sampler/uniform_set_sampler.h"
#include "variable/orientation.h"
#include "variable/position.h"
#include "util/misc.h"
#include "util/variable_name_constants.h"

namespace state_estimation {

namespace prediction_model {

class MotionModel2dLocalVelocity3dRotationState : public State {
 public:
  static const int kNumberOfStateVariables = 1 + MotionModel3dOrientationState::kNumberOfStateVariables;

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNamePositionX, this->position_.x());
    named_values->emplace(named_values->end(), util::kNamePositionY, this->position_.y());

    std::vector<std::pair<std::string, double>> orientation_state_named_values;
    this->motion_model_3d_orientation_state_.GetAllNamedValues(&orientation_state_named_values);
    for (int i = 0; i < orientation_state_named_values.size(); i++) {
      if (orientation_state_named_values.at(i).first == util::kNamePositionX) {
        continue;
      }
      if (orientation_state_named_values.at(i).first == util::kNamePositionY) {
        continue;
      }
      named_values->emplace_back(orientation_state_named_values.at(i));
    }
  }

  void EstimateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    assert(sample_state_ptrs.size() == weights.size());
    double weights_sum = 0.0;
    for (int i = 0; i < weights.size(); i++) {
      weights_sum += weights.at(i);
    }
    assert(weights_sum > 0.0);
    for (int i = 0; i < weights.size(); i++) {
      weights.at(i) /= weights_sum;
    }

    variable::Position mean_position;
    MotionModel2dLocalVelocity3dRotationState* my_state_ptr;
    std::vector<State*> orientation_sample_state_ptrs;
    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(sample_state_ptrs.at(i));
      if (i == 0) {
        mean_position = my_state_ptr->position() * weights.at(i);
      } else {
        mean_position = mean_position + my_state_ptr->position() * weights.at(i);
      }
      orientation_sample_state_ptrs.push_back(my_state_ptr->motion_model_3d_orientation_state_ptr());
    }

    MotionModel3dOrientationState orientation_estimate_state;
    orientation_estimate_state.EstimateFromSamples(orientation_sample_state_ptrs, weights);

    this->position_ = mean_position;
    this->motion_model_3d_orientation_state_ = orientation_estimate_state;
  }

  variable::Position position(void) const {
    return this->position_;
  }

  void position(variable::Position position) {
    this->position_ = position;
  }

  MotionModel3dOrientationState motion_model_3d_orientation_state(void) {
    return this->motion_model_3d_orientation_state_;
  }

  void motion_model_3d_orientation_state(MotionModel3dOrientationState motion_model_3d_orientation_state) {
    this->motion_model_3d_orientation_state_ = motion_model_3d_orientation_state;
  }

  MotionModel3dOrientationState* motion_model_3d_orientation_state_ptr(void) {
    return &(this->motion_model_3d_orientation_state_);
  }

  std::string ToKey(std::vector<double> discretization_resolution) {
    assert(discretization_resolution.size() == this->kNumberOfStateVariables);
    double position_resolution = discretization_resolution.at(0);
    variable::Position temp_position = this->position_;
    temp_position.Round(position_resolution);
    std::vector<double> orientation_discretization_resolutions = {discretization_resolution.at(1)};
    return temp_position.ToKey() + "_" + this->motion_model_3d_orientation_state_.ToKey(orientation_discretization_resolutions);
  }

  void Add(State* state_ptr) {
    MotionModel2dLocalVelocity3dRotationState* my_state_ptr = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() + temp_position.x());
    position.y(this->position_.y() + temp_position.y());
    position.floor(this->position_.floor());
    this->position_ = position;
    MotionModel3dOrientationState* orientation_state_ptr = my_state_ptr->motion_model_3d_orientation_state_ptr();
    this->motion_model_3d_orientation_state_.Add(orientation_state_ptr);
  }

  void Subtract(State* state_ptr) {
    MotionModel2dLocalVelocity3dRotationState* my_state_ptr = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_ptr);
    variable::Position temp_position = my_state_ptr->position();
    variable::Position position;
    position.x(this->position_.x() - temp_position.x());
    position.y(this->position_.y() - temp_position.y());
    position.floor(this->position_.floor());
    this->position_ = position;
    MotionModel3dOrientationState* orientation_state_ptr = my_state_ptr->motion_model_3d_orientation_state_ptr();
    this->motion_model_3d_orientation_state_.Subtract(orientation_state_ptr);
  }

  void Multiply_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() * scalar_value);
    position.y(this->position_.y() * scalar_value);
    position.floor(this->position_.floor());
    this->position_ = position;
    this->motion_model_3d_orientation_state_.Multiply_scalar(scalar_value);
  }

  void Divide_scalar(double scalar_value) {
    variable::Position position;
    position.x(this->position_.x() / scalar_value);
    position.y(this->position_.y() / scalar_value);
    position.floor(this->position_.floor());
    this->position_ = position;
    this->motion_model_3d_orientation_state_.Divide_scalar(scalar_value);
  }

  MotionModel2dLocalVelocity3dRotationState(void) {
    this->position_ = variable::Position();
    this->motion_model_3d_orientation_state_ = MotionModel3dOrientationState();
  }

  ~MotionModel2dLocalVelocity3dRotationState() {}

 private:
  variable::Position position_;  // the 2d position under the map coordinate-system
  MotionModel3dOrientationState motion_model_3d_orientation_state_;  // 3d orientation under the world coordinate system
};

class MotionModel2dLocalVelocity3dRotationStateUncertainty : public StateUncertainty {
 public:
  void CalculateFromSamples(std::vector<State*> sample_state_ptrs, std::vector<double> weights) {
    assert(sample_state_ptrs.size() == weights.size());
    Eigen::Matrix<double, Eigen::Dynamic, 2> position_2d_vectors = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sample_state_ptrs.size(), 2);
    MotionModel2dLocalVelocity3dRotationState* my_state_ptr;
    std::vector<State*> orientation_state_sample_ptrs;

    for (int i = 0; i < sample_state_ptrs.size(); i++) {
      my_state_ptr = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(sample_state_ptrs.at(i));
      position_2d_vectors(i, 0) = my_state_ptr->position().x();
      position_2d_vectors(i, 1) = my_state_ptr->position().y();
      orientation_state_sample_ptrs.push_back(my_state_ptr->motion_model_3d_orientation_state_ptr());
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> weights_vector = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(weights.size(), 1);
    for (int i = 0; i < weights.size(); i++) {
      weights_vector(i) = weights.at(i);
    }
    weights_vector = weights_vector.array() / weights_vector.sum();

    position_2d_vectors.col(0) = (position_2d_vectors.col(0).array() - position_2d_vectors.col(0).mean()) * weights_vector.array().sqrt();
    position_2d_vectors.col(1) = (position_2d_vectors.col(1).array() - position_2d_vectors.col(1).mean()) * weights_vector.array().sqrt();

    this->position_2d_covariance_ = position_2d_vectors.transpose() * position_2d_vectors;
    this->position_2d_distance_variance_ = position_2d_vectors.rowwise().squaredNorm().sum();
    this->motion_model_3d_orientation_state_uncertainty_.CalculateFromSamples(orientation_state_sample_ptrs, weights);
  }

  Eigen::Matrix<double, 2, 2> position_2d_covariance(void) {
    return this->position_2d_covariance_;
  }

  double position_2d_distance_variance(void) {
    return this->position_2d_distance_variance_;
  }

  MotionModel3dOrientationStateUncertainty motion_model_3d_orientation_state_uncertainty(void) {
    return this->motion_model_3d_orientation_state_uncertainty_;
  }

  void motion_model_3d_orientation_state_uncertainty(MotionModel3dOrientationStateUncertainty motion_model_3d_orientation_state_uncertainty) {
    this->motion_model_3d_orientation_state_uncertainty_ = motion_model_3d_orientation_state_uncertainty;
  }

  MotionModel2dLocalVelocity3dRotationStateUncertainty(void) {
    this->position_2d_covariance_ = Eigen::Matrix<double, 2, 2>::Zero(2, 2);
    this->position_2d_distance_variance_ = 0.0;
    this->motion_model_3d_orientation_state_uncertainty_ = MotionModel3dOrientationStateUncertainty();
  }

  ~MotionModel2dLocalVelocity3dRotationStateUncertainty() {}

 private:
  Eigen::Matrix<double, 2, 2> position_2d_covariance_;
  double position_2d_distance_variance_;
  MotionModel3dOrientationStateUncertainty motion_model_3d_orientation_state_uncertainty_;
};

class MotionModel2dLocalVelocity3dRotationControlInput : public ControlInput {
 public:
  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) const {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNameVLocalX, this->v_local_(0));
    named_values->emplace(named_values->end(), util::kNameVLocalY, this->v_local_(1));
    named_values->emplace(named_values->end(), util::kNameVLocalZ, this->v_local_(2));
    named_values->emplace(named_values->end(), "cov_0", this->v_local_covariance_(0, 0));
    named_values->emplace(named_values->end(), "cov_1", this->v_local_covariance_(0, 1));
    named_values->emplace(named_values->end(), "cov_2", this->v_local_covariance_(0, 2));
    named_values->emplace(named_values->end(), "cov_3", this->v_local_covariance_(1, 0));
    named_values->emplace(named_values->end(), "cov_4", this->v_local_covariance_(1, 1));
    named_values->emplace(named_values->end(), "cov_5", this->v_local_covariance_(1, 2));
    named_values->emplace(named_values->end(), "cov_6", this->v_local_covariance_(2, 0));
    named_values->emplace(named_values->end(), "cov_7", this->v_local_covariance_(2, 1));
    named_values->emplace(named_values->end(), "cov_8", this->v_local_covariance_(2, 2));

    std::vector<std::pair<std::string, double>> rotation_control_input_named_values;
    this->motion_model_3d_rotation_control_input_.GetAllNamedValues(&rotation_control_input_named_values);
    for (int i = 0; i < rotation_control_input_named_values.size(); i++) {
      named_values->emplace_back(rotation_control_input_named_values.at(i));
    }
  }

  Eigen::Vector3d v_local(void) {
    return this->v_local_;
  }

  void v_local(Eigen::Vector3d v_local) {
    this->v_local_ = v_local;
  }

  Eigen::Matrix3d v_local_covariance(void) {
    return this->v_local_covariance_;
  }

  void v_local_covariance(Eigen::Matrix3d v_local_covariance) {
    this->v_local_covariance_ = v_local_covariance;
  }

  MotionModel3dOrientationDifferentialControlInput motion_model_3d_rotation_control_input(void) {
    return this->motion_model_3d_rotation_control_input_;
  }

  void motion_model_3d_rotation_control_input(MotionModel3dOrientationDifferentialControlInput motion_model_3d_rotation_control_input) {
    this->motion_model_3d_rotation_control_input_ = motion_model_3d_rotation_control_input;
  }

  Eigen::Quaterniond orientation_sensor_pose_ws(void) {
    return this->orientation_sensor_pose_ws_;
  }

  void orientation_sensor_pose_ws(Eigen::Quaterniond orientation_sensor_pose_ws) {
    this->orientation_sensor_pose_ws_ = orientation_sensor_pose_ws;
  }

  bool INS_orientation_estimation(void) {
    return this->INS_orientation_estimation_;
  }

  void INS_orientation_estimation(bool INS_orientation_estimation) {
    this->INS_orientation_estimation_ = INS_orientation_estimation;
  }

  MotionModel2dLocalVelocity3dRotationControlInput(void) {
    this->v_local_ = Eigen::Matrix<double, 3, 1>::Zero();
    this->v_local_covariance_ = Eigen::Matrix<double, 3, 3>::Zero();
    this->motion_model_3d_rotation_control_input_ = MotionModel3dOrientationDifferentialControlInput();
    this->orientation_sensor_pose_ws_ = Eigen::Quaterniond::Identity();
    this->INS_orientation_estimation_ = true;
  }

  ~MotionModel2dLocalVelocity3dRotationControlInput() {}

 private:
  Eigen::Vector3d v_local_;
  Eigen::Matrix3d v_local_covariance_;
  MotionModel3dOrientationDifferentialControlInput motion_model_3d_rotation_control_input_;
  Eigen::Quaterniond orientation_sensor_pose_ws_;
  bool INS_orientation_estimation_;
};

class MotionModel2dLocalVelocity3dRotation : public PredictionModel {
 public:
  void Predict(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);
  void PredictWithoutControlInput(State* state_t, State* state_tminus, double dt);

  void Seed(int random_seed) {
    this->mvg_sampler_.Seed(random_seed);
    this->motion_model_3d_rotation_.Seed(random_seed);
  }

  double CalculateStateTransitionProbabilityLog(State* state_t, State* state_tminus, ControlInput* control_input_t, double dt);

  Eigen::Matrix3d R_mw(void) {
    return this->R_mw_;
  }

  void R_mw(Eigen::Matrix3d R_mw) {
    this->R_mw_ = R_mw;
  }

  MotionModel2dLocalVelocity3dRotation(Eigen::Matrix<double, 3, 3> R_mw = Eigen::Matrix<double, 3, 3>::Identity()) {
    this->mvg_sampler_ = sampler::MultivariateGaussianSampler();
    this->motion_model_3d_rotation_ = MotionModel3dOrientationDifferential();
    this->R_mw_ = R_mw;
  }

  ~MotionModel2dLocalVelocity3dRotation() {}

 private:
  sampler::MultivariateGaussianSampler mvg_sampler_;
  MotionModel3dOrientationDifferential motion_model_3d_rotation_;
  Eigen::Matrix<double, 3, 3> R_mw_;
};

template <typename MotionModel3dOrientationStateSampler>
class MotionModel2dLocalVelocity3dRotationStateSampler : public StateSampler {
 public:
  void Init(std::set<std::string> sample_space_position_keys,
            double position_griding_resolution,
            MotionModel3dOrientationStateSampler motion_model_3d_orientation_state_sampler) {
    this->position_key_sampler_.Init(sample_space_position_keys);
    this->position_griding_resolution_ = position_griding_resolution;
    this->motion_model_3d_orientation_state_sampler_ = motion_model_3d_orientation_state_sampler;
  }

  void Sample(State* state_sample) {
    MotionModel2dLocalVelocity3dRotationState* my_state_sample = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_sample);
    variable::Position temp_position;
    temp_position.FromKey(this->position_key_sampler_.Sample());
    my_state_sample->position(temp_position);
    this->motion_model_3d_orientation_state_sampler_.Sample(my_state_sample->motion_model_3d_orientation_state_ptr());
  }

  void Seed(int random_seed) {
    this->position_key_sampler_.Seed(random_seed);
    this->motion_model_3d_orientation_state_sampler_.Seed(random_seed);
  }

  double CalculateStateProbabilityLog(State* state_sample) {
    MotionModel2dLocalVelocity3dRotationState* my_state_sample = reinterpret_cast<MotionModel2dLocalVelocity3dRotationState*>(state_sample);
    variable::Position position = my_state_sample->position();
    position.Round(this->position_griding_resolution_);
    std::vector<std::string> position_key_vector = this->position_key_sampler_.population();

    double log_prob = 0.0;
    int temp_flag = 0;
    for (int i = 0; i < position_key_vector.size(); i++) {
      if (position.ToKey() == position_key_vector.at(i)) {
        temp_flag = 1;
        break;
      }
    }
    if (temp_flag) {
      log_prob += std::log(1.0 / position_key_vector.size());
    } else {
      log_prob += std::log(0.0);
    }

    log_prob += this->motion_model_3d_orientation_state_sampler_.CalculateStateProbabilityLog(my_state_sample->motion_model_3d_orientation_state_ptr());

    return log_prob;
  }

  double position_griding_resolution(void) {
    return this->position_griding_resolution_;
  }

  void position_griding_resolution(double position_griding_resolution) {
    this->position_griding_resolution_ = position_griding_resolution;
  }

  MotionModel3dOrientationStateSampler motion_model_3d_orientation_state_sampler(void) {
    return this->motion_model_3d_orientation_state_sampler_;
  }

  void motion_model_3d_orientation_state_sampler(MotionModel3dOrientationStateSampler motion_model_3d_orientation_state_sampler) {
    this->motion_model_3d_orientation_state_sampler_ = motion_model_3d_orientation_state_sampler;
  }

  MotionModel2dLocalVelocity3dRotationStateSampler(void) {
    this->position_key_sampler_ = sampler::UniformSetSampler<std::string>();
    this->position_griding_resolution_ = 1.0;
    this->motion_model_3d_orientation_state_sampler_ = MotionModel3dOrientationStateSampler();
  }

  ~MotionModel2dLocalVelocity3dRotationStateSampler() {}

 private:
  sampler::UniformSetSampler<std::string> position_key_sampler_;
  double position_griding_resolution_;
  MotionModel3dOrientationStateSampler motion_model_3d_orientation_state_sampler_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_PREDICTION_MODEL_MOTION_MODEL_2D_LOCAL_VELOCITY_3D_ROTATION_H_
