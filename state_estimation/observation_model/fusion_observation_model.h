/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-04-12 16:17:18
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-07-19 21:13:33
 */
#ifndef STATE_ESTIMATION_OBSERVATION_MODEL_FUSION_OBSERVATION_MODEL_H_
#define STATE_ESTIMATION_OBSERVATION_MODEL_FUSION_OBSERVATION_MODEL_H_

#include <Eigen/Dense>

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "observation_model/base.h"
#include "observation_model/imu_observation_model.h"
#include "observation_model/orientation_observation_model.h"
#include "observation_model/bluetooth_observation_model.h"
#include "observation_model/geomagnetism_observation_model.h"
#include "observation_model/wifi_observation_model.h"
#include "prediction_model/base.h"
#include "distribution/gaussian_distribution.h"
#include "util/variable_name_constants.h"
#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace observation_model {

class FusionObservation : public Observation {
 public:
  void Init(double buffer_duration,
            double timestamp,
            double time_unit_in_second = 1.0,
            std::vector<double> R_mw_vector = std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
            GeomagnetismFeatureVectorType feature_vector_type = GeomagnetismFeatureVectorType::kOverallMagnitude) {
    this->buffer_duration_ = buffer_duration;
    this->timestamp(timestamp, time_unit_in_second);
    this->bluetooth_observation_.Init(buffer_duration, timestamp, time_unit_in_second);
    this->wifi_observation_.Init(buffer_duration);
    this->geomagnetism_observation_.Init(buffer_duration, timestamp, time_unit_in_second, R_mw_vector, feature_vector_type);
    this->orientation_observation_.Init(buffer_duration, this->geomagnetism_observation_.R_mw());
  }

  void timestamp(double timestamp, double time_unit_in_second) {
    this->timestamp_ = timestamp;
    this->time_unit_in_second_ = time_unit_in_second;
  }

  Eigen::Matrix3d R_mw(void) {
    return this->geomagnetism_observation_.R_mw();
  }

  Eigen::Matrix3d R_ws(void) {
    return this->geomagnetism_observation_.R_ws();
  }

  GeomagnetismFeatureVectorType GetGeomagnetismFeatureVectorType(void) {
    return this->geomagnetism_observation_.GetFeatureVectorType();
  }

  void SetGeomagnetismFeatureVectorType(GeomagnetismFeatureVectorType geomag_feature_vector_type) {
    this->geomagnetism_observation_.SetFeatureVectorType(geomag_feature_vector_type);
  }

  std::vector<std::pair<std::string, double>> GetWifiFeatureVector(void) {
    return this->wifi_observation_.GetFeatureVector();
  }

  std::vector<std::pair<std::string, double>> GetBluetoothFeatureVector(void) {
    return this->bluetooth_observation_.GetFeatureVector();
  }

  std::vector<std::pair<std::string, double>> GetGeomagnetismFeatureVector(void) {
    return this->geomagnetism_observation_.GetFeatureVector();
  }

  BluetoothObservation bluetooth_observation(void) {
    return this->bluetooth_observation_;
  }

  WifiObservation wifi_observation(void) {
    return this->wifi_observation_;
  }

  GeomagnetismObservation geomagnetism_observation(void) {
    return this->geomagnetism_observation_;
  }

  void geomagnetism_observation(GeomagnetismObservation geomagnetism_observation) {
    this->geomagnetism_observation_ = geomagnetism_observation;
  }

  GeomagnetismObservation* geomagnetism_observation_ptr(void) {
    return &(this->geomagnetism_observation_);
  }

  OrientationObservation orientation_observation(void) {
    return this->orientation_observation_;
  }

  void orientation_observation(OrientationObservation orientation_observation) {
    this->orientation_observation_ = orientation_observation;
  }

  Eigen::Vector3d gravity(void) {
    return this->geomagnetism_observation_.gravity();
  }

  void gravity(Eigen::Vector3d gravity) {
    this->geomagnetism_observation_.gravity(gravity);
    this->geomagnetism_observation_.GetRsgsFromGravity();
  }

  int GetObservationFromLines(const std::vector<std::string>& bluetooth_lines,
                              const std::vector<std::string>& wifi_lines,
                              const std::vector<std::string>& geomagnetism_lines,
                              const std::vector<std::string>& gravity_lines,
                              double time_unit_in_second);
  int GetObservationFromData(const std::vector<BeaconSignalData>& bluetooth_data,
                             const std::vector<WifiSignalData>& wifi_data,
                             const std::vector<GeomagnetismData>& geomagnetism_data,
                             const std::vector<GravityData>& gravity_data);
  std::vector<std::pair<std::string, double>> GetFeatureVector(void);

  FusionObservation(void) {
    this->timestamp_ = -1.0;
    this->time_unit_in_second_ = 1.0;
    this->buffer_duration_ = 0.0;
    this->bluetooth_observation_ = BluetoothObservation();
    this->wifi_observation_ = WifiObservation();
    this->geomagnetism_observation_ = GeomagnetismObservation();
    this->orientation_observation_ = OrientationObservation();
  }

  ~FusionObservation() {}

 private:
  double timestamp_;
  double time_unit_in_second_;
  double buffer_duration_;
  BluetoothObservation bluetooth_observation_;
  WifiObservation wifi_observation_;
  GeomagnetismObservation geomagnetism_observation_;
  OrientationObservation orientation_observation_;
};

class FusionObservationState : public State {
 public:
  void FromPredictionModelState(const prediction_model::State* prediction_model_state_ptr) {
    this->bluetooth_observation_state_.FromPredictionModelState(prediction_model_state_ptr);
    this->wifi_observation_state_.FromPredictionModelState(prediction_model_state_ptr);
    this->geomagnetism_observation_state_.FromPredictionModelState(prediction_model_state_ptr);
    this->orientation_observation_state_.FromPredictionModelState(prediction_model_state_ptr);
  }

  void position(const variable::Position& position) {
    this->bluetooth_observation_state_.position(position);
    this->wifi_observation_state_.position(position);
    this->geomagnetism_observation_state_.position(position);
  }

  void bluetooth_dynamic_offset(double bluetooth_dynamic_offset) {
    this->bluetooth_observation_state_.observation_dynamic_offset(bluetooth_dynamic_offset);
  }

  void wifi_dynamic_offset(double wifi_dynamic_offset) {
    this->wifi_observation_state_.observation_dynamic_offset(wifi_dynamic_offset);
  }

  void geomagnetism_bias(const Eigen::Vector3d& geomagnetism_bias) {
    this->geomagnetism_observation_state_.bias(geomagnetism_bias);
  }

  void yaw(double yaw) {
    this->geomagnetism_observation_state_.yaw(yaw);
    this->orientation_observation_state_.yaw(yaw);
  }

  void q_ws(const Eigen::Quaterniond& q_ws) {
    this->geomagnetism_observation_state_.q_ws(q_ws);
  }

  std::string ToKey(void) {
    std::vector<std::string> sub_keys;
    sub_keys.push_back(this->bluetooth_observation_state_.ToKey());
    sub_keys.push_back(this->wifi_observation_state_.ToKey());
    sub_keys.push_back(this->geomagnetism_observation_state_.ToKey());
    std::string key;
    JoinString(sub_keys, "_", &key);
    return key;
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values);

  BluetoothObservationState bluetooth_observation_state(void) {
    return this->bluetooth_observation_state_;
  }

  void bluetooth_observation_state(BluetoothObservationState bluetooth_observation_state) {
    this->bluetooth_observation_state_ = bluetooth_observation_state;
  }

  WifiObservationState wifi_observation_state(void) {
    return this->wifi_observation_state_;
  }

  void wifi_observation_state(WifiObservationState wifi_observation_state) {
    this->wifi_observation_state_ = wifi_observation_state;
  }

  GeomagnetismObservationYawState geomagnetism_observation_state(void) {
    return this->geomagnetism_observation_state_;
  }

  void geomagnetism_observation_state(GeomagnetismObservationYawState geomagnetism_observation_state) {
    this->geomagnetism_observation_state_ = geomagnetism_observation_state;
  }

  GeomagnetismObservationYawState* geomagnetism_observation_state_ptr(void) {
    return &(this->geomagnetism_observation_state_);
  }

  OrientationObservationYawState orientation_observation_state(void) {
    return this->orientation_observation_state_;
  }

  void orientation_observation_state(OrientationObservationYawState orientation_observation_state) {
    this->orientation_observation_state_ = orientation_observation_state;
  }

  FusionObservationState(void) {
    this->bluetooth_observation_state_ = BluetoothObservationState();
    this->wifi_observation_state_ = WifiObservationState();
    this->geomagnetism_observation_state_ = GeomagnetismObservationYawState();
    this->orientation_observation_state_ = OrientationObservationYawState();
  }

  ~FusionObservationState() {}

 private:
  BluetoothObservationState bluetooth_observation_state_;
  WifiObservationState wifi_observation_state_;
  GeomagnetismObservationYawState geomagnetism_observation_state_;
  OrientationObservationYawState orientation_observation_state_;
};

template <typename BluetoothProbabilityMapper,
          typename WifiProbabilityMapper,
          typename GeomagnetismProbabilityMapper>
class FusionObservationModel : public ObservationModel {
 public:
  void Init(BluetoothProbabilityMapper bluetooth_probability_mapper,
            WifiProbabilityMapper wifi_probability_mapper,
            GeomagnetismProbabilityMapper geomagnetism_probability_mapper,
            double bluetooth_map_spatial_interval,
            double wifi_map_spatial_interval,
            double geomagnetism_map_spatial_interval,
            bool bluetooth_is_zero_centered,
            bool wifi_is_zero_centered,
            double bluetooth_log_probability_weight,
            double wifi_log_probability_weight,
            double geomagnetism_log_probability_weight,
            bool use_orientation_sensor_constraint,
            double orientation_sensor_constraint_abs_yaw_diff_rad) {
    this->bluetooth_observation_model_.Init(bluetooth_probability_mapper, bluetooth_map_spatial_interval, bluetooth_is_zero_centered);
    this->wifi_observation_model_.Init(wifi_probability_mapper, wifi_map_spatial_interval, wifi_is_zero_centered);
    this->geomagnetism_observation_model_.Init(geomagnetism_probability_mapper, geomagnetism_map_spatial_interval);
    this->orientation_observation_model_.Init(-orientation_sensor_constraint_abs_yaw_diff_rad, orientation_sensor_constraint_abs_yaw_diff_rad);
    this->bluetooth_log_probability_weight_ = bluetooth_log_probability_weight;
    this->wifi_log_probability_weight_ = wifi_log_probability_weight;
    this->geomagnetism_log_probability_weight_ = geomagnetism_log_probability_weight;
    this->use_orientation_sensor_constraint_ = use_orientation_sensor_constraint;
  }

  void SetGeomagnetismScaleFactor(double geomagnetism_scale_factor) {
    this->geomagnetism_observation_model_.SetProbabilityMapSmoothFactor(geomagnetism_scale_factor);
  }

  void SetGeomagnetismDenseRelativeObservationStepLength(int dense_relative_observation_step_length) {
    assert(dense_relative_observation_step_length > 0);
    this->geomagnetism_observation_model_.dense_relative_observation_step_length(dense_relative_observation_step_length);
  }

  void SetGeomagnetismAdditiveNoiseStd(double geomagnetism_additive_noise_std) {
    assert(geomagnetism_additive_noise_std > -1.0e-6);
    this->geomagnetism_observation_model_.additive_noise_std(geomagnetism_additive_noise_std);
  }

  double GetProbabilityObservationConditioningState(Observation* observation, State* state) const {
    return std::exp(this->GetProbabilityObservationConditioningStateLog(observation, state));
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation, State* state) const {
    FusionObservation* my_observation = reinterpret_cast<FusionObservation*>(observation);
    FusionObservationState* my_state = reinterpret_cast<FusionObservationState*>(state);

    BluetoothObservation bluetooth_observation = my_observation->bluetooth_observation();
    WifiObservation wifi_observation = my_observation->wifi_observation();
    GeomagnetismObservation geomagnetism_observation = my_observation->geomagnetism_observation();
    OrientationObservation orientation_observation = my_observation->orientation_observation();
    BluetoothObservationState bluetooth_observation_state = my_state->bluetooth_observation_state();
    WifiObservationState wifi_observation_state = my_state->wifi_observation_state();
    GeomagnetismObservationYawState geomagnetism_observation_state = my_state->geomagnetism_observation_state();
    OrientationObservationYawState orientation_observation_state = my_state->orientation_observation_state();

    double bluetooth_prob_log = this->bluetooth_observation_model_.GetProbabilityObservationConditioningStateLog(&bluetooth_observation, &bluetooth_observation_state);
    double wifi_prob_log = this->wifi_observation_model_.GetProbabilityObservationConditioningStateLog(&wifi_observation, &wifi_observation_state);
    double geomagnetism_prob_log = this->geomagnetism_observation_model_.GetProbabilityObservationConditioningStateLog(&geomagnetism_observation, &geomagnetism_observation_state);
    double orientation_prob_log = this->orientation_observation_model_.GetProbabilityObservationConditioningStateLog(&orientation_observation, &orientation_observation_state);
#ifdef DEBUG_PARTICLE_FILTER_UPDATE_WEIGHT
    std::cout << "FusionObservationModel::GetProbabilityObservationConditioningStateLog:" << std::endl;
    std::cout << "  bluetooth_prob_log: " << bluetooth_prob_log << std::endl;
    std::cout << "  wifi_prob_log: " << wifi_prob_log << std::endl;
    std::cout << "  geomagnetism_prob_log: " << geomagnetism_prob_log << std::endl;
#endif

    double log_probability = std::log(std::pow(std::exp(wifi_prob_log), this->wifi_log_probability_weight_)) +
                             std::log(std::pow(std::exp(bluetooth_prob_log), this->bluetooth_log_probability_weight_)) +
                             std::log(std::pow(std::exp(geomagnetism_prob_log), this->geomagnetism_log_probability_weight_));

    if (this->use_orientation_sensor_constraint_) {
      log_probability += orientation_prob_log;
    }

    return log_probability;
  }

  std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const {
    // TBD this is currently logically incorrect.
    return std::unordered_map<std::string, double>();
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation_tminus, State* state_tminus, Observation* observation_t, State* state_t) {
    FusionObservation* my_observation_tminus = reinterpret_cast<FusionObservation*>(observation_tminus);
    FusionObservationState* my_state_tminus = reinterpret_cast<FusionObservationState*>(state_tminus);
    FusionObservation* my_observation_t = reinterpret_cast<FusionObservation*>(observation_t);
    FusionObservationState* my_state_t = reinterpret_cast<FusionObservationState*>(state_t);

    BluetoothObservation bluetooth_observation = my_observation_t->bluetooth_observation();
    WifiObservation wifi_observation = my_observation_t->wifi_observation();
    GeomagnetismObservation geomagnetism_observation_tminus = my_observation_tminus->geomagnetism_observation();
    GeomagnetismObservation geomagnetism_observation_t = my_observation_t->geomagnetism_observation();
    OrientationObservation orientation_observation = my_observation_t->orientation_observation();

    BluetoothObservationState bluetooth_observation_state = my_state_t->bluetooth_observation_state();
    WifiObservationState wifi_observation_state = my_state_t->wifi_observation_state();
    GeomagnetismObservationYawState geomagnetism_observation_state_tminus = my_state_tminus->geomagnetism_observation_state();
    GeomagnetismObservationYawState geomagnetism_observation_state_t = my_state_t->geomagnetism_observation_state();
    OrientationObservationYawState orientation_observation_state = my_state_t->orientation_observation_state();

    double bluetooth_prob_log = this->bluetooth_observation_model_.GetProbabilityObservationConditioningStateLog(&bluetooth_observation, &bluetooth_observation_state);
    double wifi_prob_log = this->wifi_observation_model_.GetProbabilityObservationConditioningStateLog(&wifi_observation, &wifi_observation_state);
    double geomagnetism_prob_log = this->geomagnetism_observation_model_.GetProbabilityObservationConditioningStateLog(&geomagnetism_observation_tminus, &geomagnetism_observation_state_tminus, &geomagnetism_observation_t, &geomagnetism_observation_state_t);
    double orientation_prob_log = this->orientation_observation_model_.GetProbabilityObservationConditioningStateLog(&orientation_observation, &orientation_observation_state);

    double log_probability = std::log(std::pow(std::exp(wifi_prob_log), this->wifi_log_probability_weight_)) +
                             std::log(std::pow(std::exp(bluetooth_prob_log), this->bluetooth_log_probability_weight_)) +
                             std::log(std::pow(std::exp(geomagnetism_prob_log), this->geomagnetism_log_probability_weight_));

    if (this->use_orientation_sensor_constraint_) {
      log_probability += orientation_prob_log;
    }

    return log_probability;
  }

  double GetProbabilityObservationConditioningStateLog(std::list<Observation*> observation_window, std::list<State*> state_window, Observation* observation_t, State* state_t) {
    // The general observation model distribution:
    //  p(y_t|x_0, x_1,..., x_t, y_0, y_1,..., y_t-1)
    // To make it practical and tractable:
    //  p(y_t|x_t-k, x_t-k-1,..., x_t, y_t-k, y_t-k-1,..., y_t-1)
    assert(observation_window.size() == state_window.size());

    FusionObservation* my_observation_t = reinterpret_cast<FusionObservation*>(observation_t);
    FusionObservationState* my_state_t = reinterpret_cast<FusionObservationState*>(state_t);

    BluetoothObservation bluetooth_observation = my_observation_t->bluetooth_observation();
    WifiObservation wifi_observation = my_observation_t->wifi_observation();
    GeomagnetismObservation geomagnetism_observation_t = my_observation_t->geomagnetism_observation();
    OrientationObservation orientation_observation = my_observation_t->orientation_observation();

    BluetoothObservationState bluetooth_observation_state = my_state_t->bluetooth_observation_state();
    WifiObservationState wifi_observation_state = my_state_t->wifi_observation_state();
    GeomagnetismObservationYawState geomagnetism_observation_state_t = my_state_t->geomagnetism_observation_state();
    OrientationObservationYawState orientation_observation_state = my_state_t->orientation_observation_state();

    double bluetooth_prob_log = this->bluetooth_observation_model_.GetProbabilityObservationConditioningStateLog(&bluetooth_observation, &bluetooth_observation_state);
    double wifi_prob_log = this->wifi_observation_model_.GetProbabilityObservationConditioningStateLog(&wifi_observation, &wifi_observation_state);

    std::list<Observation*> geomagnetism_observation_window;
    std::list<State*> geomagnetism_observation_state_window;

    auto observation_iter = observation_window.begin();
    auto observation_state_iter = state_window.begin();
    while (observation_iter != observation_window.end()) {
      FusionObservation* my_observation_current = reinterpret_cast<FusionObservation*>(*observation_iter);
      FusionObservationState* my_observation_state_current = reinterpret_cast<FusionObservationState*>(*observation_state_iter);
      geomagnetism_observation_window.push_back(my_observation_current->geomagnetism_observation_ptr());
      geomagnetism_observation_state_window.push_back(my_observation_state_current->geomagnetism_observation_state_ptr());
      observation_iter++;
      observation_state_iter++;
    }

    double geomagnetism_prob_log = this->geomagnetism_observation_model_.GetProbabilityObservationConditioningStateLog(geomagnetism_observation_window, geomagnetism_observation_state_window, &geomagnetism_observation_t, &geomagnetism_observation_state_t);

    double orientation_prob_log = this->orientation_observation_model_.GetProbabilityObservationConditioningStateLog(&orientation_observation, &orientation_observation_state);

    double log_probability = std::log(std::pow(std::exp(wifi_prob_log), this->wifi_log_probability_weight_)) +
                             std::log(std::pow(std::exp(bluetooth_prob_log), this->bluetooth_log_probability_weight_)) +
                             std::log(std::pow(std::exp(geomagnetism_prob_log), this->geomagnetism_log_probability_weight_));

    if (this->use_orientation_sensor_constraint_) {
      log_probability += orientation_prob_log;
    }

    return log_probability;
  }

  double CalculateBluetoothOffset(FusionObservation observation, FusionObservationState gt_state) {
    BluetoothObservation bluetooth_observation = observation.bluetooth_observation();
    BluetoothObservationState bluetooth_observation_state = gt_state.bluetooth_observation_state();

    return this->bluetooth_observation_model_.CalculateBluetoothOffset(bluetooth_observation, bluetooth_observation_state);
  }

  double CalculateWifiOffset(FusionObservation observation, FusionObservationState gt_state) {
    WifiObservation wifi_observation = observation.wifi_observation();
    WifiObservationState wifi_observation_state = gt_state.wifi_observation_state();

    return this->wifi_observation_model_.CalculateWifiOffset(wifi_observation, wifi_observation_state);
  }

  Eigen::Vector3d CalculateGeomagnetismBias(FusionObservation observation, FusionObservationState gt_state, Eigen::Quaterniond gt_q_ms) {
    GeomagnetismObservation geomagnetism_observation = observation.geomagnetism_observation();
    GeomagnetismObservationYawState geomagnetism_observation_state = gt_state.geomagnetism_observation_state();

    return this->geomagnetism_observation_model_.CalculateGeomagnetismBias(geomagnetism_observation, geomagnetism_observation_state, gt_q_ms);
  }

  int CalculateGeomagnetismBiasMeanAndCovariance(Eigen::Vector3d &geomagnetism_bias_mean, Eigen::Matrix3d &geomagnetism_bias_covariance, FusionObservation observation, FusionObservationState observation_state) {
    GeomagnetismObservation geomagnetism_observation = observation.geomagnetism_observation();
    GeomagnetismObservationYawState geomagnetism_observation_state = observation_state.geomagnetism_observation_state();
    return this->geomagnetism_observation_model_.CalculateGeomagnetismBiasMeanAndCovariance(geomagnetism_bias_mean, geomagnetism_bias_covariance, geomagnetism_observation, geomagnetism_observation_state);
  }

  BluetoothObservationModel<BluetoothProbabilityMapper> bluetooth_observation_model(void) {
    return this->bluetooth_observation_model_;
  }

  WifiObservationModel<WifiProbabilityMapper> wifi_observation_model(void) {
    return this->wifi_observation_model_;
  }

  GeomagnetismObservationModelYaw<GeomagnetismProbabilityMapper> geomagnetism_observation_model(void) {
    return this->geomagnetism_observation_model_;
  }

  GeomagnetismObservationModelYaw<GeomagnetismProbabilityMapper>* geomagnetism_observation_model_ptr(void) {
    return &(this->geomagnetism_observation_model_);
  }

  double bluetooth_log_probability_weight(void) {
    return this->bluetooth_log_probability_weight_;
  }

  double wifi_log_probability_weight(void) {
    return this->wifi_log_probability_weight_;
  }

  double geomagnetism_log_probability_weight(void) {
    return this->geomagnetism_log_probability_weight_;
  }

  void SetGeomagnetismProbabilityMapper(const GeomagnetismProbabilityMapper& geomagnetism_probability_mapper) {
    this->geomagnetism_observation_model_.probability_mapper(geomagnetism_probability_mapper);
  }

  FusionObservationModel(void) {
    this->bluetooth_observation_model_ = BluetoothObservationModel<BluetoothProbabilityMapper>();
    this->wifi_observation_model_ = WifiObservationModel<WifiProbabilityMapper>();
    this->geomagnetism_observation_model_ = GeomagnetismObservationModelYaw<GeomagnetismProbabilityMapper>();
    this->orientation_observation_model_ = OrientationObservationYawModel();
    this->bluetooth_log_probability_weight_ = 1.0;
    this->wifi_log_probability_weight_ = 1.0;
    this->geomagnetism_log_probability_weight_ = 1.0;
    this->use_orientation_sensor_constraint_ = false;
  }

  ~FusionObservationModel() {}

 private:
  BluetoothObservationModel<BluetoothProbabilityMapper> bluetooth_observation_model_;
  WifiObservationModel<WifiProbabilityMapper> wifi_observation_model_;
  GeomagnetismObservationModelYaw<GeomagnetismProbabilityMapper> geomagnetism_observation_model_;
  OrientationObservationYawModel orientation_observation_model_;
  double bluetooth_log_probability_weight_;
  double wifi_log_probability_weight_;
  double geomagnetism_log_probability_weight_;
  bool use_orientation_sensor_constraint_;
};

}  // namespace observation_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OBSERVATION_MODEL_FUSION_OBSERVATION_MODEL_H_
