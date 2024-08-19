/*
 * Copyright (c) 2021 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-06-08 16:50:10
 * @LastEditTime: 2023-01-31 21:03:32
 * @LastEditors: xuehua xuehua@sensetime.com
 */
#ifndef STATE_ESTIMATION_OBSERVATION_MODEL_GEOMAGNETISM_OBSERVATION_MODEL_H_
#define STATE_ESTIMATION_OBSERVATION_MODEL_GEOMAGNETISM_OBSERVATION_MODEL_H_

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <list>

#include "distribution/gaussian_distribution.h"
#include "observation_model/base.h"
#include "observation_model/imu_observation_model.h"
#include "observation_model/orientation_observation_model.h"
#include "prediction_model/base.h"
#include "util/variable_name_constants.h"
#include "variable/orientation.h"
#include "variable/position.h"
#include "util/misc.h"

namespace state_estimation {

namespace observation_model {

struct GeomagnetismData {
  double timestamp = -1.0;
  double time_unit_in_second = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  int device_accuracy = 0;
};

enum class GeomagnetismFeatureVectorType {
  kOverallMagnitude = 0,
  kTwoDimensionalMagnitude,
  kThreeDimensionalVector,
};

bool GetGeomagnetismDataFromLine(const std::string &line_data, GeomagnetismData &geomagnetism_data, double time_unit_in_second = 1.0);

class GeomagnetismObservation : public Observation {
 public:
  void Init(double buffer_duration,
            double timestamp,
            double time_unit_in_second = 1.0,
            std::vector<double> R_mw_vector = std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
            GeomagnetismFeatureVectorType feature_vector_type = GeomagnetismFeatureVectorType::kOverallMagnitude);

  int GetObservationFromLines(const std::vector<std::string>& geomagnetism_lines, double time_unit_in_second);
  int GetObservationFromGeomagnetismDatas(const std::vector<GeomagnetismData> &geomagnetism_datas);

  int GetObservationFromGeomagnetismEigenVector(const Eigen::Vector3d geomagnetism_s) {
    this->feature_values_ = geomagnetism_s;
    return 1;
  }

  int GetRwsFromOrientationLines(
      const std::vector<std::string>& orientation_lines,
      double target_timestamp);
  int GetRwsFromAttitudeDatas(const std::vector<AttitudeData> &attitude_datas);
  int GetRwsFromGravityAndGeomagnetismLines(
      const std::vector<std::string>& gravity_lines,
      const std::vector<std::string>& geomagnetism_lines,
      double target_timestamp);
  int GetRwsFromGravityAndGeomagnetismDatas(const std::vector<GravityData> &gravity_datas, const std::vector<GeomagnetismData> &geomagnetism_datas);
  int GetRwsFromGT(const Eigen::Quaterniond& gt_orientation);
  int GetRwsFromGravityAndRwsgAngle(const std::vector<std::string>& gravity_lines, double R_wsg_angle, double target_timestamp);
  int GetRwsFromGravityDatasAndRwsgAngle(const std::vector<GravityData> &gravity_datas, double R_wsg_angle);
  int GetRwsFromRwsgAngle(double R_wsg_angle);

  int GetRsgsFromGravity(void);

  // R_ms = R_mw * R_ws, R_ms is the rotation from sensor-coordinates to map-coordinates;
  // R_ms = R_mw * R_wsg * R_sgs, R_sgs is the rotation from sensor-coordinates to gravity-aligned-sensor-coordinates,
  // and R_wsg is the rotation from gravity-aligned-sensor-coordinates to world-coordinates.
  // R_ms = R_msg * R_sgs.
  int GetRwsgFromGT(const Eigen::Quaterniond& gt_orientation);
  int GetRwsgFromRwsAndGravity(void);

  int GetGravityFromLines(const std::vector<std::string>& gravity_lines, double target_timestamp);
  int GetGravityFromGravityDatas(const std::vector<GravityData> &gravity_datas);
  int GetGravityFromGT(const Eigen::Quaterniond& gt_orientation);
  int GetGravityFromOrientationLines(const std::vector<std::string> orientation_lines, double target_timestamp);
  int GetGravityFromAttitudeDatas(const std::vector<AttitudeData> &attitude_datas);

  std::vector<std::pair<std::string, double>> GetFeatureVector(void);

  Eigen::Vector3d GetFeatureValuesVector(void) {
    return this->feature_values_;
  }

  double timestamp(void) {
    return this->timestamp_;
  }

  void timestamp(double timestamp, double time_unit_in_second = 1.0) {
    this->timestamp_ = timestamp;
    this->time_unit_in_second_ = time_unit_in_second;
  }

  double time_unit_in_second(void) {
    return this->time_unit_in_second_;
  }

  double buffer_duration(void) {
    return this->buffer_duration_;
  }

  Eigen::Matrix3d R_mw(void) {
    return this->R_mw_;
  }

  void R_mw(Eigen::Matrix3d R_mw) {
    this->R_mw_ = R_mw;
  }

  Eigen::Matrix3d R_ws(void) {
    return this->R_ws_;
  }

  Eigen::Matrix3d R_wsg(void) {
    return this->R_wsg_;
  }

  Eigen::Matrix3d R_sgs(void) {
    return this->R_sgs_;
  }

  GeomagnetismFeatureVectorType GetFeatureVectorType(void) {
    return this->feature_vector_type_;
  }

  void SetFeatureVectorType(GeomagnetismFeatureVectorType feature_vector_type) {
    this->feature_vector_type_ = feature_vector_type;
  }

  Eigen::Matrix<double, 3, 1> bias(void) {
    return this->bias_;
  }

  void bias(Eigen::Matrix<double, 3, 1> bias) {
    this->bias_ = bias;
  }

  Eigen::Matrix<double, 3, 1> gravity(void) {
    return this->gravity_;
  }

  void gravity(Eigen::Matrix<double, 3, 1> gravity) {
    this->gravity_ = gravity;
  }

  GeomagnetismObservation(void) {
    this->timestamp_ = -1.0;
    this->time_unit_in_second_ = 1.0;
    this->feature_values_ = Eigen::Vector3d::Zero();
    this->buffer_duration_ = 0.0;
    this->R_mw_ = Eigen::Matrix3d::Identity(3, 3);
    this->R_ws_ = Eigen::Matrix3d::Identity(3, 3);
    this->R_wsg_ = Eigen::Matrix3d::Identity(3, 3);
    this->R_sgs_ = Eigen::Matrix3d::Identity(3, 3);
    this->feature_vector_type_ = GeomagnetismFeatureVectorType::kThreeDimensionalVector;
    this->bias_ = Eigen::Matrix<double, 3, 1>::Zero(3, 1);
    this->gravity_ = Eigen::Matrix<double, 3, 1>({0.0, 0.0, 1.0});
  }

  ~GeomagnetismObservation() {}

 private:
  double timestamp_;
  double time_unit_in_second_;
  Eigen::Vector3d feature_values_;
  double buffer_duration_;
  Eigen::Matrix3d R_mw_;
  Eigen::Matrix3d R_ws_;
  Eigen::Matrix3d R_wsg_;
  Eigen::Matrix3d R_sgs_;
  GeomagnetismFeatureVectorType feature_vector_type_;
  Eigen::Matrix<double, 3, 1> bias_;
  Eigen::Matrix<double, 3, 1> gravity_;
};

class GeomagnetismObservationState : public State {
 public:
  variable::Position position(void) {
    return this->position_;
  }

  void position(variable::Position position) {
    this->position_ = position;
  }

  Eigen::Vector3d bias(void) {
    return this->bias_;
  }

  void bias(Eigen::Vector3d bias) {
    this->bias_ = bias;
  }

  Eigen::Matrix3d bias_covariance(void) {
    return this->bias_covariance_;
  }

  void bias_covariance(Eigen::Matrix3d bias_covariance) {
    if (!IsCovarianceMatrix(bias_covariance)) {
      std::cout << "GeomagnetismObservationState::bias_covariance(Eigen::Matrix3d): "
                << "the provided matrix is not a covariance matrix, leave the member variable unchanged." << std::endl;
      return;
    }
    this->bias_covariance_ = bias_covariance;
  }

  void FromPredictionModelState(const prediction_model::State* prediction_model_state_ptr) {
    std::vector<std::pair<std::string, double>> named_values;
    prediction_model_state_ptr->GetAllNamedValues(&named_values);
    variable::Position temp_position;
    Eigen::Matrix3d bias_covariance = this->bias_covariance_;
    for (int i = 0; i < named_values.size(); i++) {
      if (named_values.at(i).first == util::kNamePositionX) {
        temp_position.x(named_values.at(i).second);
      } else if (named_values.at(i).first == util::kNamePositionY) {
        temp_position.y(named_values.at(i).second);
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasX) {
        this->bias_(0) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasY) {
        this->bias_(1) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasZ) {
        this->bias_(2) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceXX) {
        bias_covariance(0, 0) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceYY) {
        bias_covariance(1, 1) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceZZ) {
        bias_covariance(2, 2) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceXY) {
        bias_covariance(0, 1) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceYX) {
        bias_covariance(1, 0) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceXZ) {
        bias_covariance(0, 2) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceZX) {
        bias_covariance(2, 0) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceYZ) {
        bias_covariance(1, 2) = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameGeomagnetismBiasCovarianceZY) {
        bias_covariance(2, 1) = named_values.at(i).second;
      }
    }
    this->bias_covariance(bias_covariance);
    this->position_ = temp_position;
  }

  std::string ToKey(void) {
    return this->position_.ToKey();
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNamePositionX, this->position_.x());
    named_values->emplace(named_values->end(), util::kNamePositionY, this->position_.y());
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasX, this->bias_(0));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasY, this->bias_(1));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasZ, this->bias_(2));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceXX, this->bias_covariance_(0, 0));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceYY, this->bias_covariance_(1, 1));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceZZ, this->bias_covariance_(2, 2));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceXY, this->bias_covariance_(0, 1));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceYX, this->bias_covariance_(1, 0));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceXZ, this->bias_covariance_(0, 2));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceZX, this->bias_covariance_(2, 0));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceYZ, this->bias_covariance_(1, 2));
    named_values->emplace(named_values->end(), util::kNameGeomagnetismBiasCovarianceZY, this->bias_covariance_(2, 1));
  }

  GeomagnetismObservationState(void) {
    this->position_ = variable::Position();
    this->bias_ = Eigen::Vector3d::Zero();
    this->bias_covariance_ = Eigen::Matrix3d::Zero();
  }

  ~GeomagnetismObservationState() {}

 private:
  variable::Position position_;
  Eigen::Vector3d bias_;
  Eigen::Matrix3d bias_covariance_;
};

class GeomagnetismObservationYawState : public GeomagnetismObservationState {
 public:
  double yaw(void) {
    return this->yaw_;
  }

  void yaw(double yaw) {
    this->yaw_ = yaw;
  }

  Eigen::Quaterniond q_ws(void) {
    return this->q_ws_;
  }

  void q_ws(const Eigen::Quaterniond& q_ws) {
    this->q_ws_ = q_ws;
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) {
    GeomagnetismObservationState::GetAllNamedValues(named_values);
    named_values->emplace(named_values->end(), util::kNameYaw, this->yaw_);
    named_values->emplace(named_values->end(), util::kNameOrientationW, this->q_ws_.w());
    named_values->emplace(named_values->end(), util::kNameOrientationX, this->q_ws_.x());
    named_values->emplace(named_values->end(), util::kNameOrientationY, this->q_ws_.y());
    named_values->emplace(named_values->end(), util::kNameOrientationZ, this->q_ws_.z());
  }

  void FromPredictionModelState(const prediction_model::State* prediction_model_state_ptr) {
    GeomagnetismObservationState::FromPredictionModelState(prediction_model_state_ptr);
    std::vector<std::pair<std::string, double>> named_values;
    prediction_model_state_ptr->GetAllNamedValues(&named_values);
    double q_w = 1.0;
    double q_x = 0.0;
    double q_y = 0.0;
    double q_z = 0.0;
    for (int i = 0; i < named_values.size(); i++) {
      if (named_values.at(i).first == util::kNameYaw) {
        this->yaw_ = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameOrientationW) {
        q_w = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameOrientationX) {
        q_x = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameOrientationY) {
        q_y = named_values.at(i).second;
      } else if (named_values.at(i).first == util::kNameOrientationZ) {
        q_z = named_values.at(i).second;
      }
    }
    this->q_ws_ = Eigen::Quaterniond(q_w, q_x, q_y, q_z).normalized();
  }

  GeomagnetismObservationYawState(void) {
    this->yaw_ = 0.0;
    this->q_ws_ = Eigen::Quaterniond::Identity();
  }

  ~GeomagnetismObservationYawState() {}

 private:
  double yaw_;
  Eigen::Quaterniond q_ws_;
};

class GeomagnetismObservationState3D : public GeomagnetismObservationState {
 public:
  variable::Orientation orientation(void) {
    return this->orientation_;
  }

  void orientation(variable::Orientation orientation) {
    this->orientation_ = orientation;
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) {
    GeomagnetismObservationState::GetAllNamedValues(named_values);
    named_values->emplace(named_values->end(), util::kNameOrientationW, this->orientation_.q().w());
    named_values->emplace(named_values->end(), util::kNameOrientationX, this->orientation_.q().x());
    named_values->emplace(named_values->end(), util::kNameOrientationY, this->orientation_.q().y());
    named_values->emplace(named_values->end(), util::kNameOrientationZ, this->orientation_.q().z());
  }

  GeomagnetismObservationState3D(void) {
    this->orientation_ = variable::Orientation();
  }

  ~GeomagnetismObservationState3D() {}

 private:
  variable::Orientation orientation_;
};

template <typename ProbabilityMapper>
class GeomagnetismObservationModel : public ObservationModel {
 public:
  void Init(ProbabilityMapper probability_mapper, double map_spatial_interval) {
    this->probability_mapper_ = probability_mapper;
    this->probability_mapper_.SetZeroCentering(false);
    this->map_spatial_interval_ = map_spatial_interval;
  }

  double GetProbabilityObservationConditioningState(Observation* observation, State* state) const {
#ifdef DEBUG_FOCUSING
    std::cout << "GeomagnetismObservationModel::GetProbabilityObservationConditioningState" << std::endl;
#endif
    GeomagnetismObservation* my_observation = reinterpret_cast<GeomagnetismObservation*>(observation);
    GeomagnetismObservationState* my_state = reinterpret_cast<GeomagnetismObservationState*>(state);

    my_observation->bias(my_state->bias());
    std::vector<std::pair<std::string, double>> feature_vector = my_observation->GetFeatureVector();

    double dynamic_offset = 0.0;
    variable::Position position = my_state->position();
    position.Round(this->map_spatial_interval_);
    return this->probability_mapper_.CalculateProbabilityFeatureVectorConditioningState(feature_vector, position.ToKey(), dynamic_offset);
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation, State* state) const {
#ifdef DEBUG_FOCUSING
    std::cout << "GeomagnetismObservationModel::GetProbabilityObservationConditioningStateLog" << std::endl;
#endif
    GeomagnetismObservation* my_observation = reinterpret_cast<GeomagnetismObservation*>(observation);
    GeomagnetismObservationState* my_state = reinterpret_cast<GeomagnetismObservationState*>(state);

    my_observation->bias(my_state->bias());
    std::vector<std::pair<std::string, double>> feature_vector = my_observation->GetFeatureVector();

    double dynamic_offset = 0.0;
    variable::Position position = my_state->position();
    position.Round(this->map_spatial_interval_);
    return this->probability_mapper_.CalculateProbabilityFeatureVectorConditioningStateLog(feature_vector, position.ToKey(), dynamic_offset);
  }

  std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const {
    GeomagnetismObservation* my_observation = reinterpret_cast<GeomagnetismObservation*>(observation);

    return this->probability_mapper_.CalculateProbabilityStatesConditioningFeatureVector(my_observation->GetFeatureVector());
  }

  GeomagnetismObservationModel(void) {
    ProbabilityMapper probability_mapper;
    this->probability_mapper_ = probability_mapper;
    this->map_spatial_interval_ = 0.0;
  }

  ~GeomagnetismObservationModel() {}

 private:
  ProbabilityMapper probability_mapper_;
  double map_spatial_interval_;
};

template <typename ProbabilityMapper>
class GeomagnetismObservationModelYaw : public ObservationModel {
 public:
  void Init(ProbabilityMapper probability_mapper, double map_spatial_interval) {
    this->probability_mapper_ = probability_mapper;
    this->probability_mapper_.SetZeroCentering(false);
    this->map_spatial_interval_ = map_spatial_interval;
    this->calculation_spatial_interval_ = map_spatial_interval;
    this->dense_relative_observation_step_length_ = 1;
  }

  void SetProbabilityMapSmoothFactor(double smooth_factor) {
    this->probability_mapper_.SetProbabilityMapSmoothFactor(smooth_factor);
  }

  int dense_relative_observation_step_length(void) {
    return this->dense_relative_observation_step_length_;
  }

  void dense_relative_observation_step_length(int dense_relative_observation_step_length) {
    this->dense_relative_observation_step_length_ = dense_relative_observation_step_length;
  }

  void Init(ProbabilityMapper probability_mapper, double map_spatial_interval, double calculation_spatial_interval, int dense_relative_observation_step_length = 1) {
    this->probability_mapper_ = probability_mapper;
    this->probability_mapper_.SetZeroCentering(false);
    this->map_spatial_interval_ = map_spatial_interval;
    this->calculation_spatial_interval_ = calculation_spatial_interval;
    this->dense_relative_observation_step_length_ = dense_relative_observation_step_length;
  }

  double GetProbabilityObservationConditioningState(Observation* observation, State* state) const {
#ifdef DEBUG_FOCUSING
    std::cout << "GeomagnetismObservationModelYaw::GetProbabilityObservationConditioningState" << std::endl;
#endif
    return std::exp(this->GetProbabilityObservationConditioningStateLog(observation, state));
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation, State* state) const {
#ifdef DEBUG_FOCUSING
    std::cout << "GeomagnetismObservationModelYaw::GetProbabilityObservationConditioningStateLog" << std::endl;
#endif
    GeomagnetismObservation* my_observation = reinterpret_cast<GeomagnetismObservation*>(observation);
    GeomagnetismObservationYawState* my_state = reinterpret_cast<GeomagnetismObservationYawState*>(state);

    Eigen::Vector3d geomagnetism_s = my_observation->GetFeatureValuesVector();
    Eigen::Vector3d gravity_s = my_observation->gravity();
    Eigen::Matrix3d R_mw = my_observation->R_mw();

    Eigen::Quaterniond q_ws = my_state->q_ws().normalized();
    Eigen::Matrix3d R_ms = R_mw * q_ws;

    Eigen::Vector3d geomagnetism_bias_m = my_state->bias();
    Eigen::Matrix3d geomagnetism_bias_P = my_state->bias_covariance();

    variable::Position position = my_state->position();
    position.Round(this->calculation_spatial_interval_);

    double std_scale_factor = this->probability_mapper_.GetProbabilityMapSmoothFactor();

    const std::unordered_map<std::string, std::vector<double>>* distribution_ptr = this->probability_mapper_.GetDistributionParams(position.ToKey());
    double geomagnetism_log_prob = 0.0;
    if (!distribution_ptr) {
      geomagnetism_log_prob = std::log(0.0);
    } else {
      Eigen::Vector3d geomagnetism_m_mean;
      geomagnetism_m_mean(0) = distribution_ptr->at("x").at(0);
      geomagnetism_m_mean(1) = distribution_ptr->at("y").at(0);
      geomagnetism_m_mean(2) = distribution_ptr->at("z").at(0);

      Eigen::Matrix3d geomagnetism_m_cov = Eigen::Matrix3d::Zero();
      geomagnetism_m_cov(0, 0) = std::pow(distribution_ptr->at("x").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
      geomagnetism_m_cov(1, 1) = std::pow(distribution_ptr->at("y").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
      geomagnetism_m_cov(2, 2) = std::pow(distribution_ptr->at("z").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);

      Eigen::Vector3d geomagnetism_s_mean = R_ms.transpose() * geomagnetism_m_mean + geomagnetism_bias_m;
      Eigen::Matrix3d geomagnetism_s_cov = R_ms.transpose() * geomagnetism_m_cov * R_ms + geomagnetism_bias_P;

      std::vector<double> geomagnetism_s_mean_vector = EigenVector2Vector(geomagnetism_s_mean);
      std::vector<double> geomagnetism_s_cov_vector = CovarianceMatrixToCompactVector(geomagnetism_s_cov);
      std::vector<double> geomagnetism_s_vector = EigenVector2Vector(geomagnetism_s);

      distribution::MultivariateGaussian mvg_distribution(geomagnetism_s_mean_vector, geomagnetism_s_cov_vector);

      geomagnetism_log_prob = mvg_distribution.LogPDF(geomagnetism_s_vector);
    }

    return geomagnetism_log_prob;
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation_tminus, State* state_tminus, Observation* observation_t, State* state_t) {
    GeomagnetismObservation* my_observation_tminus = reinterpret_cast<GeomagnetismObservation*>(observation_tminus);
    GeomagnetismObservation* my_observation_t = reinterpret_cast<GeomagnetismObservation*>(observation_t);
    GeomagnetismObservationYawState* my_state_tminus = reinterpret_cast<GeomagnetismObservationYawState*>(state_tminus);
    GeomagnetismObservationYawState* my_state_t = reinterpret_cast<GeomagnetismObservationYawState*>(state_t);

    Eigen::Vector3d geomagnetism_s_tminus = my_observation_tminus->GetFeatureValuesVector();
    Eigen::Vector3d geomagnetism_s_t = my_observation_t->GetFeatureValuesVector();

    std::vector<double> geomagnetism_s_t_minus_tminus_vector = EigenVector2Vector(geomagnetism_s_t - geomagnetism_s_tminus);

    Eigen::Matrix3d R_mw = my_observation_t->R_mw();

    variable::Position position_tminus = my_state_tminus->position();
    position_tminus.Round(this->calculation_spatial_interval_);
    variable::Position position_t = my_state_t->position();
    position_t.Round(this->calculation_spatial_interval_);

    Eigen::Matrix3d R_ms_tminus = R_mw * my_state_tminus->q_ws().normalized();
    Eigen::Matrix3d R_ms_t = R_mw * my_state_t->q_ws().normalized();

    double std_scale_factor = this->probability_mapper_.GetProbabilityMapSmoothFactor();

    double correlation = 0.0;

    const std::unordered_map<std::string, std::vector<double>>* distribution_tminus_ptr = this->probability_mapper_.GetDistributionParams(position_tminus.ToKey());
    const std::unordered_map<std::string, std::vector<double>>* distribution_t_ptr = this->probability_mapper_.GetDistributionParams(position_t.ToKey());

    std::vector<std::string> position_keys = {position_tminus.ToKey(), position_t.ToKey()};
    Eigen::MatrixXd position_covariance_matrix;
    int n_valid_keys = this->probability_mapper_.GetKeyCovariance(position_keys, &position_covariance_matrix);
    // when the two positions are same, make the correlation equal to zero, samples are independent.
    if (position_tminus.ToKey() == position_t.ToKey()) {
      n_valid_keys = 0;
    }

    double geomagnetism_prob_log = 0.0;
    if (!distribution_tminus_ptr || !distribution_t_ptr) {
      geomagnetism_prob_log = std::log(0.0);
    } else {
      assert(position_covariance_matrix.size() == 4);
      if (n_valid_keys == 2) {
        double sigma_position_tminus = std::pow(position_covariance_matrix(0, 0), 0.5);
        double sigma_position_t = std::pow(position_covariance_matrix(1, 1), 0.5);
        correlation = position_covariance_matrix(0, 1) / (sigma_position_tminus * sigma_position_t);
      }

      Eigen::Vector3d mean_tminus = Eigen::Vector3d::Zero();
      mean_tminus(0) = distribution_tminus_ptr->at("x").at(0);
      mean_tminus(1) = distribution_tminus_ptr->at("y").at(0);
      mean_tminus(2) = distribution_tminus_ptr->at("z").at(0);

      Eigen::Matrix3d covariance_tminus = Eigen::Matrix3d::Zero();
      covariance_tminus(0, 0) = std::pow(distribution_tminus_ptr->at("x").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
      covariance_tminus(1, 1) = std::pow(distribution_tminus_ptr->at("y").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
      covariance_tminus(2, 2) = std::pow(distribution_tminus_ptr->at("z").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);

      Eigen::Vector3d mean_t = Eigen::Vector3d::Zero();
      mean_t(0) = distribution_t_ptr->at("x").at(0);
      mean_t(1) = distribution_t_ptr->at("y").at(0);
      mean_t(2) = distribution_t_ptr->at("z").at(0);

      Eigen::Matrix3d covariance_t = Eigen::Matrix3d::Zero();
      covariance_t(0, 0) = std::pow(distribution_t_ptr->at("x").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
      covariance_t(1, 1) = std::pow(distribution_t_ptr->at("y").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
      covariance_t(2, 2) = std::pow(distribution_t_ptr->at("z").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);

      // the combination of x is X * (x_t, x_tminus)^T
      // the combination of y is Y * (y_t, y_tminus)^T
      // the combination of z is Z * (z_t, z_tminus)^T
      // where X, Y and Z are all 3*2 matrix;
      // the final random variable is the sum of the above three independent multivariate Gaussian variables.
      // m = X*x + Y*y + Z*z
      // then we can apply the correlation between t and tminus.
      Eigen::Vector2d x = Eigen::Vector2d::Zero();
      Eigen::Vector2d y = Eigen::Vector2d::Zero();
      Eigen::Vector2d z = Eigen::Vector2d::Zero();
      x(0) = mean_t(0);
      x(1) = mean_tminus(0);
      y(0) = mean_t(1);
      y(1) = mean_tminus(1);
      z(0) = mean_t(2);
      z(1) = mean_tminus(2);
      Eigen::Matrix<double, 3, 2> X = Eigen::Matrix<double, 3, 2>::Zero();
      Eigen::Matrix<double, 3, 2> Y = Eigen::Matrix<double, 3, 2>::Zero();
      Eigen::Matrix<double, 3, 2> Z = Eigen::Matrix<double, 3, 2>::Zero();
      X(0, 0) = R_ms_t.transpose()(0, 0);
      X(0, 1) = -R_ms_tminus.transpose()(0, 0);
      X(1, 0) = R_ms_t.transpose()(1, 0);
      X(1, 1) = -R_ms_tminus.transpose()(1, 0);
      X(2, 0) = R_ms_t.transpose()(2, 0);
      X(2, 1) = -R_ms_tminus.transpose()(2, 0);
      Y(0, 0) = R_ms_t.transpose()(0, 1);
      Y(0, 1) = -R_ms_tminus.transpose()(0, 1);
      Y(1, 0) = R_ms_t.transpose()(1, 1);
      Y(1, 1) = -R_ms_tminus.transpose()(1, 1);
      Y(2, 0) = R_ms_t.transpose()(2, 1);
      Y(2, 1) = -R_ms_tminus.transpose()(2, 1);
      Z(0, 0) = R_ms_t.transpose()(0, 2);
      Z(0, 1) = -R_ms_tminus.transpose()(0, 2);
      Z(1, 0) = R_ms_t.transpose()(1, 2);
      Z(1, 1) = -R_ms_tminus.transpose()(1, 2);
      Z(2, 0) = R_ms_t.transpose()(2, 2);
      Z(2, 1) = -R_ms_tminus.transpose()(2, 2);

      Eigen::Matrix2d x_cov = Eigen::Matrix2d::Zero();
      x_cov(0, 0) = covariance_t(0, 0);
      x_cov(1, 1) = covariance_tminus(0, 0);
      x_cov(0, 1) = correlation * std::pow(covariance_t(0, 0), 0.5) * std::pow(covariance_tminus(0, 0), 0.5);
      x_cov(1, 0) = correlation * std::pow(covariance_t(0, 0), 0.5) * std::pow(covariance_tminus(0, 0), 0.5);
      Eigen::Matrix2d y_cov = Eigen::Matrix2d::Zero();
      y_cov(0, 0) = covariance_t(1, 1);
      y_cov(1, 1) = covariance_tminus(1, 1);
      y_cov(0, 1) = correlation * std::pow(covariance_t(1, 1), 0.5) * std::pow(covariance_tminus(1, 1), 0.5);
      y_cov(1, 0) = correlation * std::pow(covariance_t(1, 1), 0.5) * std::pow(covariance_tminus(1, 1), 0.5);
      Eigen::Matrix2d z_cov = Eigen::Matrix2d::Zero();
      z_cov(0, 0) = covariance_t(2, 2);
      z_cov(1, 1) = covariance_tminus(2, 2);
      z_cov(0, 1) = correlation * std::pow(covariance_t(2, 2), 0.5) * std::pow(covariance_tminus(2, 2), 0.5);
      z_cov(1, 0) = correlation * std::pow(covariance_t(2, 2), 0.5) * std::pow(covariance_tminus(2, 2), 0.5);

      Eigen::Vector3d m = X*x + Y*y + Z*z;
      Eigen::Matrix3d m_cov = X * x_cov * X.transpose() + Y * y_cov * Y.transpose() + Z * z_cov * Z.transpose();

      std::vector<double> m_vector = EigenVector2Vector(m);
      std::vector<double> m_cov_vector = CovarianceMatrixToCompactVector(m_cov);

      distribution::MultivariateGaussian mvg_distribution(m_vector, m_cov_vector);

      geomagnetism_prob_log = mvg_distribution.LogPDF(geomagnetism_s_t_minus_tminus_vector);
    }

    return geomagnetism_prob_log;
  }

  double GetProbabilityObservationConditioningStateLog(std::list<Observation*> observation_window, std::list<State*> state_window, Observation* observation_t, State* state_t) {
    assert(observation_window.size() == state_window.size());
    assert(this->dense_relative_observation_step_length_ > 0);
    if (observation_window.size() < this->dense_relative_observation_step_length_) {
      return 0.0;
    }
    int n_relative_observations = std::floor(observation_window.size() / this->dense_relative_observation_step_length_);
    int start_index = observation_window.size() - n_relative_observations * this->dense_relative_observation_step_length_;

    double std_scale_factor = this->probability_mapper_.GetProbabilityMapSmoothFactor();

    GeomagnetismObservation* my_observation_t = reinterpret_cast<GeomagnetismObservation*>(observation_t);
    GeomagnetismObservationYawState* my_state_t = reinterpret_cast<GeomagnetismObservationYawState*>(state_t);
    Eigen::Vector3d geomagnetism_s_t = my_observation_t->GetFeatureValuesVector();

    Eigen::Matrix3d R_mw = my_observation_t->R_mw();
    Eigen::Matrix3d R_ms_t = R_mw * my_state_t->q_ws().normalized();

    variable::Position position_t = my_state_t->position();
    position_t.Round(this->calculation_spatial_interval_);
    std::string position_key_t = position_t.ToKey();

    const std::unordered_map<std::string, std::vector<double>>* distribution_t_ptr = this->probability_mapper_.GetDistributionParams(position_key_t);

    if (!distribution_t_ptr) {
      return std::log(0.0);
    }

    // prepare coefficient matrices
    Eigen::VectorXd geomagnetism_s_relative = Eigen::VectorXd::Zero(3 * n_relative_observations);
    Eigen::VectorXd x_m = Eigen::VectorXd::Zero(n_relative_observations + 1);
    Eigen::VectorXd y_m = Eigen::VectorXd::Zero(n_relative_observations + 1);
    Eigen::VectorXd z_m = Eigen::VectorXd::Zero(n_relative_observations + 1);
    x_m(0) = distribution_t_ptr->at("x").at(0);
    y_m(0) = distribution_t_ptr->at("y").at(0);
    z_m(0) = distribution_t_ptr->at("z").at(0);
    Eigen::MatrixXd x_covariance_m = Eigen::MatrixXd::Zero(n_relative_observations + 1, n_relative_observations + 1);
    Eigen::MatrixXd y_covariance_m = Eigen::MatrixXd::Zero(n_relative_observations + 1, n_relative_observations + 1);
    Eigen::MatrixXd z_covariance_m = Eigen::MatrixXd::Zero(n_relative_observations + 1, n_relative_observations + 1);
    x_covariance_m(0, 0) = std::pow(distribution_t_ptr->at("x").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
    y_covariance_m(0, 0) = std::pow(distribution_t_ptr->at("y").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
    z_covariance_m(0, 0) = std::pow(distribution_t_ptr->at("z").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
    // Eigen::MatrixXd X = Eigen::MatrixXd::Zero(3 * n_relative_observations, n_relative_observations + 1);
    // Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(3 * n_relative_observations, n_relative_observations + 1);
    // Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(3 * n_relative_observations, n_relative_observations + 1);
    Eigen::VectorXd X_compact = Eigen::VectorXd::Zero(3 * n_relative_observations);
    Eigen::VectorXd Y_compact = Eigen::VectorXd::Zero(3 * n_relative_observations);
    Eigen::VectorXd Z_compact = Eigen::VectorXd::Zero(3 * n_relative_observations);
    std::list<Observation*>::iterator observation_iter = observation_window.begin();
    std::list<State*>::iterator state_iter = state_window.begin();
    std::vector<std::string> position_keys = {position_key_t};
    int current_index = 0;
    int next_index = start_index;
    GeomagnetismObservation* my_observation_temp = nullptr;
    GeomagnetismObservationYawState* my_state_temp = nullptr;
    Eigen::Vector3d geomagnetism_s_temp = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R_ms_temp = Eigen::Matrix3d::Identity();
    variable::Position position_temp;
    std::string position_key_temp;
    const std::unordered_map<std::string, std::vector<double>>* distribution_temp_ptr = nullptr;
    for (int i = 0; i < n_relative_observations; i++) {
      while (current_index < next_index) {
        observation_iter++;
        state_iter++;
        current_index++;
      }
      next_index += this->dense_relative_observation_step_length_;
      my_observation_temp = reinterpret_cast<GeomagnetismObservation*>(*observation_iter);
      my_state_temp = reinterpret_cast<GeomagnetismObservationYawState*>(*state_iter);

      geomagnetism_s_temp = my_observation_temp->GetFeatureValuesVector();
      R_ms_temp = R_mw * my_state_temp->q_ws().normalized();

      position_temp = my_state_temp->position();
      position_temp.Round(this->calculation_spatial_interval_);
      position_key_temp = position_temp.ToKey();
      position_keys.emplace_back(position_key_temp);
      distribution_temp_ptr = this->probability_mapper_.GetDistributionParams(position_key_temp);
      if (!distribution_temp_ptr) {
        return std::log(0.0);
      }
      x_m(n_relative_observations - i) = distribution_temp_ptr->at("x").at(0);
      y_m(n_relative_observations - i) = distribution_temp_ptr->at("y").at(0);
      z_m(n_relative_observations - i) = distribution_temp_ptr->at("z").at(0);
      x_covariance_m(n_relative_observations - i, n_relative_observations - i) = std::pow(distribution_temp_ptr->at("x").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
      y_covariance_m(n_relative_observations - i, n_relative_observations - i) = std::pow(distribution_temp_ptr->at("y").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
      z_covariance_m(n_relative_observations - i, n_relative_observations - i) = std::pow(distribution_temp_ptr->at("z").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);

      geomagnetism_s_relative.block(3 * (n_relative_observations - 1 - i), 0, 3, 1) = geomagnetism_s_t - geomagnetism_s_temp;

      // X.block(3 * (n_relative_observations - 1 - i), 0, 3, 1) = R_ms_t.row(0).transpose();
      // Y.block(3 * (n_relative_observations - 1 - i), 0, 3, 1) = R_ms_t.row(1).transpose();
      // Z.block(3 * (n_relative_observations - 1 - i), 0, 3, 1) = R_ms_t.row(2).transpose();

      // X.block(3 * (n_relative_observations - 1 - i), n_relative_observations - i, 3, 1) = -R_ms_temp.row(0).transpose();
      // Y.block(3 * (n_relative_observations - 1 - i), n_relative_observations - i, 3, 1) = -R_ms_temp.row(1).transpose();
      // Z.block(3 * (n_relative_observations - 1 - i), n_relative_observations - i, 3, 1) = -R_ms_temp.row(2).transpose();
      X_compact.block(3 * (n_relative_observations - 1 - i), 0, 3, 1) = -R_ms_temp.row(0).transpose();
      Y_compact.block(3 * (n_relative_observations - 1 - i), 0, 3, 1) = -R_ms_temp.row(1).transpose();
      Z_compact.block(3 * (n_relative_observations - 1 - i), 0, 3, 1) = -R_ms_temp.row(2).transpose();

      // observation_iter++;
      // state_iter++;
    }

    // fill covariance matrix
    Eigen::MatrixXd position_covariance_matrix;
    int n_valid_keys = this->probability_mapper_.GetKeyCovariance(position_keys, &position_covariance_matrix);
    if (n_valid_keys == position_keys.size()) {
      for (int i = 0; i < position_keys.size(); i++) {
        for (int j = i + 1; j < position_keys.size(); j++) {
          if (position_keys.at(i) == position_keys.at(j)) {
            x_covariance_m(n_relative_observations - i, n_relative_observations - j) = 0.0;
            x_covariance_m(n_relative_observations - j, n_relative_observations - i) = 0.0;
            y_covariance_m(n_relative_observations - i, n_relative_observations - j) = 0.0;
            y_covariance_m(n_relative_observations - j, n_relative_observations - i) = 0.0;
            z_covariance_m(n_relative_observations - i, n_relative_observations - j) = 0.0;
            z_covariance_m(n_relative_observations - j, n_relative_observations - i) = 0.0;
          } else {
            x_covariance_m(n_relative_observations - i, n_relative_observations - j) = position_covariance_matrix(i, j);
            x_covariance_m(n_relative_observations - j, n_relative_observations - i) = position_covariance_matrix(i, j);
            y_covariance_m(n_relative_observations - i, n_relative_observations - j) = position_covariance_matrix(i, j);
            y_covariance_m(n_relative_observations - j, n_relative_observations - i) = position_covariance_matrix(i, j);
            z_covariance_m(n_relative_observations - i, n_relative_observations - j) = position_covariance_matrix(i, j);
            z_covariance_m(n_relative_observations - j, n_relative_observations - i) = position_covariance_matrix(i, j);
          }
        }
      }
    }

    // calculate relative_mean and relative_cov
    // -- relative_mean = X * x_m + Y * y_m + Z * z_m
    // -- relative_cov = X * x_cov * X.transpose() + Y * y_cov * Y.transpose() + Z * z_cov * Z.transpose()
    Eigen::VectorXd geomagnetism_s_relative_mean(3 * n_relative_observations, 1);
    // trying to remove X, Y and Z since constructing such large matrices is time-critical.
    // Eigen::Vector3d temp = x_m(0) * X.block(0, 0, 3, 1) + y_m(0) * Y.block(0, 0, 3, 1) + z_m(0) * Z.block(0, 0, 3, 1);
    Eigen::Vector3d temp = x_m(0) * R_ms_t.row(0).transpose() + y_m(0) * R_ms_t.row(1).transpose() + z_m(0) * R_ms_t.row(2).transpose();

    int row_index;
    for (int i = 0; i < n_relative_observations; i++) {
      row_index = 3 * i;
      // geomagnetism_s_relative_mean.block(row_index, 0, 3, 1) = temp + x_m(i + 1) * X.block(row_index, i + 1, 3, 1) + y_m(i + 1) * Y.block(row_index, i + 1, 3, 1) + z_m(i + 1) * Z.block(row_index, i + 1, 3, 1);
      geomagnetism_s_relative_mean.block(row_index, 0, 3, 1) = temp + x_m(i + 1) * X_compact.block(row_index, 0, 3, 1) + y_m(i + 1) * Y_compact.block(row_index, 0, 3, 1) + z_m(i + 1) * Z_compact.block(row_index, 0, 3, 1);
    }

    Eigen::MatrixXd geomagnetism_s_relative_covariance(3 * n_relative_observations, 3 * n_relative_observations);

    for (int i = 0; i < n_relative_observations; i++) {
      for (int j = i; j < n_relative_observations; j++) {
        // calculating for the cov(3 * i:3 * (i + 1), 3 * j:3 * (j + 1))
        // for (int p = 3 * i; p < 3 * (i + 1); p++) {
        //   for (int q = 3 * j; q < 3 * (j + 1); q++) {
        //     // calculating for the cov(p, q);
        //     // for the p-th row of X, Y and Z, the non-zero column indices are 0 and i + 1
        //     // for the q-th row of X, Y and Z, the non-zero column indices are 0 and j + 1
        //     geomagnetism_s_relative_covariance(p, q) = X(p, 0) * X(q, 0) * x_covariance_m(0, 0) +
        //                                                X(p, 0) * X(q, j + 1) * x_covariance_m(0, j + 1) +
        //                                                X(p, i + 1) * X(q, 0) * x_covariance_m(i + 1, 0) +
        //                                                X(p, i + 1) * X(q, j + 1) * x_covariance_m(i + 1, j + 1) +
        //                                                Y(p, 0) * Y(q, 0) * y_covariance_m(0, 0) +
        //                                                Y(p, 0) * Y(q, j + 1) * y_covariance_m(0, j + 1) +
        //                                                Y(p, i + 1) * Y(q, 0) * y_covariance_m(i + 1, 0) +
        //                                                Y(p, i + 1) * Y(q, j + 1) * y_covariance_m(i + 1, j + 1) +
        //                                                Z(p, 0) * Z(q, 0) * z_covariance_m(0, 0) +
        //                                                Z(p, 0) * Z(q, j + 1) * z_covariance_m(0, j + 1) +
        //                                                Z(p, i + 1) * Z(q, 0) * z_covariance_m(i + 1, 0) +
        //                                                Z(p, i + 1) * Z(q, j + 1) * z_covariance_m(i + 1, j + 1);
        //     if (i != j) {
        //       geomagnetism_s_relative_covariance(q, p) = geomagnetism_s_relative_covariance(p, q);
        //     }
        //   }
        // }

        // trying to remove X, Y and Z since constructing such large matrices is time-critical
        for (int p = 0; p < 3; p++) {
          for (int q = 0; q < 3; q++) {
            geomagnetism_s_relative_covariance(p + 3 * i, q + 3 * j) = R_ms_t(0, p) * R_ms_t(0, q) * x_covariance_m(0, 0) +
                                                                       R_ms_t(0, p) * X_compact(q + 3 * j) * x_covariance_m(0, j + 1) +
                                                                       X_compact(p + 3 * i) * R_ms_t(0, q) * x_covariance_m(i + 1, 0) +
                                                                       X_compact(p + 3 * i) * X_compact(q + 3 * j) * x_covariance_m(i + 1, j + 1) +
                                                                       R_ms_t(1, p) * R_ms_t(1, q) * y_covariance_m(0, 0) +
                                                                       R_ms_t(1, p) * Y_compact(q + 3 * j) * y_covariance_m(0, j + 1) +
                                                                       Y_compact(p + 3 * i) * R_ms_t(1, q) * y_covariance_m(i + 1, 0) +
                                                                       Y_compact(p + 3 * i) * Y_compact(q + 3 * j) * y_covariance_m(i + 1, j + 1) +
                                                                       R_ms_t(2, p) * R_ms_t(2, q) * z_covariance_m(0, 0) +
                                                                       R_ms_t(2, p) * Z_compact(q + 3 * j) * z_covariance_m(0, j + 1) +
                                                                       Z_compact(p + 3 * i) * R_ms_t(2, q) * z_covariance_m(i + 1, 0) +
                                                                       Z_compact(p + 3 * i) * Z_compact(q + 3 * j) * z_covariance_m(i + 1, j + 1);
            if (i != j) {
              geomagnetism_s_relative_covariance(q + 3 * j, p + 3 * i) = geomagnetism_s_relative_covariance(p + 3 * i, q + 3 * j);
            }
          }
        }
      }
    }

    distribution::MultivariateGaussian mvg_distribution(geomagnetism_s_relative_mean, geomagnetism_s_relative_covariance);

    return mvg_distribution.LogPDF(geomagnetism_s_relative) / n_relative_observations;
  }

  std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const {
    GeomagnetismObservation* my_observation = reinterpret_cast<GeomagnetismObservation*>(observation);

    return this->probability_mapper_.CalculateProbabilityStatesConditioningFeatureVector(my_observation->GetFeatureVector());
  }

  Eigen::Vector3d CalculateGeomagnetismBias(GeomagnetismObservation observation, GeomagnetismObservationYawState gt_state, Eigen::Quaterniond gt_q_ms) {
    // assume that the observation already have the gt R_ws;
    variable::Position position = gt_state.position();
    position.Round(this->map_spatial_interval_);
    std::unordered_map<std::string, double> named_means = this->probability_mapper_.LookupMeans(position.ToKey());
    Eigen::Vector3d bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d map_vector_global = Eigen::Vector3d::Zero();
    Eigen::Vector3d sensor_vector_local = observation.GetFeatureValuesVector();
    if (named_means.find("x") != named_means.end()) {
      map_vector_global(0) = named_means.at("x");
    }
    if (named_means.find("y") != named_means.end()) {
      map_vector_global(1) = named_means.at("y");
    }
    if (named_means.find("z") != named_means.end()) {
      map_vector_global(2) = named_means.at("z");
    }
    Eigen::Vector3d map_vector_local = gt_q_ms.conjugate() * map_vector_global;
    bias = sensor_vector_local - map_vector_local;
    return bias;
  }

  Eigen::Vector3d CalculateGeomagnetismBias(GeomagnetismObservation observation, GeomagnetismObservationYawState observation_state) {
    Eigen::Matrix3d R_mw = observation.R_mw();
    Eigen::Vector3d geomagnetism_s = observation.GetFeatureValuesVector();

    Eigen::Matrix3d R_ms = R_mw * observation_state.q_ws().normalized();

    variable::Position position = observation_state.position();
    position.Round(this->map_spatial_interval_);
    std::unordered_map<std::string, double> named_means = this->probability_mapper_.LookupMeans(position.ToKey());
    Eigen::Vector3d map_vector_global = Eigen::Vector3d::Zero();
    for (auto it = named_means.begin(); it != named_means.end(); it++) {
      if (it->first == "x") {
        map_vector_global(0) = it->second;
      } else if (it->first == "y") {
        map_vector_global(1) = it->second;
      } else if (it->first == "z") {
        map_vector_global(2) = it->second;
      }
    }

    Eigen::Vector3d map_vector_local = R_ms.transpose() * map_vector_global;
    Eigen::Vector3d bias = geomagnetism_s - map_vector_local;
    return bias;
  }

  int CalculateGeomagnetismBiasMeanAndCovariance(Eigen::Vector3d &geomagnetism_bias_mean, Eigen::Matrix3d &geomagnetism_bias_covariance, GeomagnetismObservation observation, GeomagnetismObservationYawState observation_state) {
    Eigen::Matrix3d R_mw = observation.R_mw();
    Eigen::Vector3d geomagnetism_s = observation.GetFeatureValuesVector();

    Eigen::Matrix3d R_ms = R_mw * observation_state.q_ws().normalized();

    variable::Position position = observation_state.position();
    position.Round(this->map_spatial_interval_);
    const std::unordered_map<std::string, std::vector<double>>* distribution_ptr = this->probability_mapper_.GetDistributionParams(position.ToKey());
    if (!distribution_ptr) {
      return 0;
    }
    double std_scale_factor = this->probability_mapper_.GetProbabilityMapSmoothFactor();
    Eigen::Vector3d geomagnetism_m_mean;
    geomagnetism_m_mean(0) = distribution_ptr->at("x").at(0);
    geomagnetism_m_mean(1) = distribution_ptr->at("y").at(0);
    geomagnetism_m_mean(2) = distribution_ptr->at("z").at(0);
    Eigen::Matrix3d geomagnetism_m_covariance = Eigen::Matrix3d::Zero();
    geomagnetism_m_covariance(0, 0) = std::pow(distribution_ptr->at("x").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
    geomagnetism_m_covariance(1, 1) = std::pow(distribution_ptr->at("y").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);
    geomagnetism_m_covariance(2, 2) = std::pow(distribution_ptr->at("z").at(1) * std_scale_factor + this->additive_noise_std_, 2.0);

    geomagnetism_bias_mean = geomagnetism_s - R_ms.transpose() * geomagnetism_m_mean;
    geomagnetism_bias_covariance = R_ms.transpose() * geomagnetism_m_covariance * R_ms;
    return 1;
  }

  ProbabilityMapper probability_mapper(void) {
    return this->probability_mapper_;
  }

  void probability_mapper(const ProbabilityMapper& probability_mapper) {
    this->probability_mapper_ = probability_mapper_;
    this->probability_mapper_.SetZeroCentering(false);
  }

  const ProbabilityMapper* probability_mapper_ptr(void) {
    return &(this->probability_mapper_);
  }

  double map_spatial_interval(void) {
    return this->map_spatial_interval_;
  }

  double calculation_spatial_interval(void) {
    return this->calculation_spatial_interval_;
  }

  double additive_noise_std(void) {
    return this->additive_noise_std_;
  }

  void additive_noise_std(double additive_noise_std) {
    this->additive_noise_std_ = additive_noise_std;
  }

  void JitterState(State* state) {
    return;
  }

  GeomagnetismObservationModelYaw(void) {
    ProbabilityMapper probability_mapper;
    this->probability_mapper_ = probability_mapper;
    this->map_spatial_interval_ = 0.0;
    this->additive_noise_std_ = 0.0;
    this->dense_relative_observation_step_length_ = 1;
  }

  ~GeomagnetismObservationModelYaw() {}

 private:
  ProbabilityMapper probability_mapper_;
  double map_spatial_interval_;
  double calculation_spatial_interval_;
  double additive_noise_std_;
  int dense_relative_observation_step_length_;
};

void AlignFeatureVector(std::vector<std::pair<std::string, double>>* named_feature_vector, variable::Orientation orientation);

template <typename ProbabilityMapper>
class GeomagnetismObservationModel3D : public ObservationModel {
 public:
  void Init(ProbabilityMapper probability_mapper, double map_spatial_interval) {
    this->probability_mapper_ = probability_mapper;
    this->probability_mapper_.SetZeroCentering(false);
    this->map_spatial_interval_ = map_spatial_interval;
  }

  double GetProbabilityObservationConditioningState(Observation* observation, State* state) const {
    GeomagnetismObservation* my_observation =
        reinterpret_cast<GeomagnetismObservation*>(observation);
    GeomagnetismObservationState3D* my_state =
        reinterpret_cast<GeomagnetismObservationState3D*>(state);

    // align geomagnetism record to my_state.orientation
    // get the raw observation. the observation must be retrieved using GetObservationFromLines.
    std::vector<std::pair<std::string, double>> feature_vector = my_observation->GetFeatureVector();
    AlignFeatureVector(&feature_vector, my_state->orientation());

    // compute the probability according to position
    variable::Position position = my_state->position();
    position.Round(this->map_spatial_interval_);
    return this->probability_mapper_.CalculateProbabilityFeatureVectorConditioningState(feature_vector, position.ToKey());
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation, State* state) const {
    GeomagnetismObservation* my_observation =
        reinterpret_cast<GeomagnetismObservation*>(observation);
    GeomagnetismObservationState3D* my_state =
        reinterpret_cast<GeomagnetismObservationState3D*>(state);

    // align geomagnetism record to my_state.orientation
    // get the raw observation. the observation must be retrieved using GetObservationFromLines.
    std::vector<std::pair<std::string, double>> feature_vector = my_observation->GetFeatureVector();
    AlignFeatureVector(&feature_vector, my_state->orientation());

    // compute the probability according to position
    variable::Position position = my_state->position();
    position.Round(this->map_spatial_interval_);
    return this->probability_mapper_.CalculateProbabilityFeatureVectorConditioningStateLog(feature_vector, position.ToKey());
  }

  std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const {
    GeomagnetismObservation* my_observation =
        reinterpret_cast<GeomagnetismObservation*>(observation);

    std::vector<std::pair<std::string, double>> feature_vector = my_observation->GetFeatureVector();
    return this->probability_mapper_.CalculateProbabilityStatesConditioningFeatureVector(my_observation->GetFeatureVector());
  }

  GeomagnetismObservationModel3D(void) {
    ProbabilityMapper probability_mapper;
    this->probability_mapper_ = probability_mapper;
    this->map_spatial_interval_ = 0.0;
  }

  ~GeomagnetismObservationModel3D() {}

 private:
  ProbabilityMapper probability_mapper_;
  double map_spatial_interval_;
};

}  // namespace observation_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OBSERVATION_MODEL_GEOMAGNETISM_OBSERVATION_MODEL_H_
