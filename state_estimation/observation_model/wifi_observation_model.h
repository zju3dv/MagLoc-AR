/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-31 13:54:56
 * @Last Modified by: xuehua
 * @Last Modified time: 2023-01-18 17:18:26
 */
#ifndef STATE_ESTIMATION_OBSERVATION_MODEL_WIFI_OBSERVATION_MODEL_H_
#define STATE_ESTIMATION_OBSERVATION_MODEL_WIFI_OBSERVATION_MODEL_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "observation_model/base.h"
#include "prediction_model/base.h"
#include "util/variable_name_constants.h"
#include "variable/position.h"

namespace state_estimation {

namespace observation_model {

class WifiSignalData {
 public:
  // Beacondata from collect line
  double unix_timestamp_ = 0;
  double unix_timestamp_lastseen_ = 0;
  double system_timestamp_lastseen_ = 0;
  double time_unit_in_second_ = 1.0;
  std::string ssid_, bssid_, channel_;
  // Identity_str: Identity of Signal : ssid_bssid_channel
  std::string identity_str_;
  int RSSI_, num_items_;

  // init for 6 items
  WifiSignalData(double unix_timestamp,
                 std::string ssid, std::string bssid,
                 int RSSI, std::string channel, double unix_timestamp_lastseen, double system_timestamp_lastseen)
      : unix_timestamp_(unix_timestamp), ssid_(ssid), bssid_(bssid), RSSI_(RSSI), channel_(channel), unix_timestamp_lastseen_(unix_timestamp_lastseen), system_timestamp_lastseen_(system_timestamp_lastseen), num_items_(6) {}

  WifiSignalData(void) {}

  ~WifiSignalData() {}
};

bool GetWifiSignalDataFromLine(const std::string& line_data, WifiSignalData& wifi_data);

class WifiObservation : public Observation {
 public:
  void Init(double buffer_duration) {
    this->buffer_duration_ = buffer_duration;
  }

  int GetObservationFromLines(const std::vector<std::string>& wifi_lines, double time_unit_in_second);
  std::vector<std::pair<std::string, double>> GetFeatureVector(void);
  std::vector<std::pair<std::string, double>> GetZeroCenteredFeatureVector(void);

  WifiObservation(void);
  ~WifiObservation();

 private:
  std::unordered_map<std::string, std::vector<int>> feature_rssis_;
  double buffer_duration_;
};

class WifiObservationState : public State {
 public:
  variable::Position position(void) {
    return this->position_;
  }

  void position(variable::Position position) {
    this->position_ = position;
  }

  double observation_dynamic_offset(void) {
    return this->observation_dynamic_offset_;
  }

  void observation_dynamic_offset(double observation_dynamic_offset) {
    this->observation_dynamic_offset_ = observation_dynamic_offset;
  }

  void FromPredictionModelState(const prediction_model::State* prediction_model_state_ptr) {
    std::vector<std::pair<std::string, double>> named_values;
    prediction_model_state_ptr->GetAllNamedValues(&named_values);
    variable::Position temp_position;
    for (int i = 0; i < named_values.size(); i++) {
      if (named_values.at(i).first == util::kNamePositionX) {
        temp_position.x(named_values.at(i).second);
      } else if (named_values.at(i).first == util::kNamePositionY) {
        temp_position.y(named_values.at(i).second);
      } else if (named_values.at(i).first == util::kNameWifiDynamicOffset) {
        this->observation_dynamic_offset_ = named_values.at(i).second;
      }
    }
    this->position_ = temp_position;
  }

  std::string ToKey(void) {
    return this->position_.ToKey();
  }

  void GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) {
    named_values->clear();
    named_values->emplace(named_values->end(), util::kNamePositionX, this->position_.x());
    named_values->emplace(named_values->end(), util::kNamePositionY, this->position_.y());
    named_values->emplace(named_values->end(), util::kNameWifiDynamicOffset, this->observation_dynamic_offset_);
  }

  WifiObservationState(void);
  ~WifiObservationState();

 private:
  variable::Position position_;
  double observation_dynamic_offset_;
};

template <typename ProbabilityMapper>
class WifiObservationModel : public ObservationModel {
 public:
  void Init(ProbabilityMapper probability_mapper, double map_spatial_interval, bool is_zero_centered) {
    this->probability_mapper_ = probability_mapper;
    this->probability_mapper_.SetZeroCentering(is_zero_centered);
    this->zero_centering_ = is_zero_centered;
    this->map_spatial_interval_ = map_spatial_interval;
  }

  double GetProbabilityObservationConditioningState(Observation* observation, State* state) const {
    WifiObservation* my_observation =
        reinterpret_cast<WifiObservation*>(observation);
    WifiObservationState* my_state =
        reinterpret_cast<WifiObservationState*>(state);

    double dynamic_map_offset = my_state->observation_dynamic_offset();
    variable::Position position = my_state->position();
    position.Round(this->map_spatial_interval_);
    return this->probability_mapper_.CalculateProbabilityFeatureVectorConditioningState(
        my_observation->GetFeatureVector(), position.ToKey(), dynamic_map_offset);
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation, State* state) const {
#ifdef DEBUG_FOCUSING
    std::cout << "WifiObservationModel::GetProbabilityObservationConditioningStateLog" << std::endl;
#endif
    WifiObservation* my_observation =
        reinterpret_cast<WifiObservation*>(observation);
    WifiObservationState* my_state =
        reinterpret_cast<WifiObservationState*>(state);

    double dynamic_map_offset = my_state->observation_dynamic_offset();
    variable::Position position = my_state->position();
    position.Round(this->map_spatial_interval_);
    return this->probability_mapper_.CalculateProbabilityFeatureVectorConditioningStateLog(
        my_observation->GetFeatureVector(), position.ToKey(), dynamic_map_offset);
  }

  std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const {
    WifiObservation* my_observation = reinterpret_cast<WifiObservation*>(observation);

    return this->probability_mapper_.CalculateProbabilityStatesConditioningFeatureVector(my_observation->GetFeatureVector());
  }

  double CalculateWifiOffset(WifiObservation observation, WifiObservationState gt_state) {
    variable::Position position = gt_state.position();
    position.Round(this->map_spatial_interval_);
    std::unordered_map<std::string, double> named_means = this->probability_mapper_.LookupMeans(position.ToKey());
    std::vector<std::pair<std::string, double>> feature_vector = observation.GetFeatureVector();
    int counter = 0;
    double offset = 0.0;
    for (int i = 0; i < feature_vector.size(); i++) {
      std::string feature_name = feature_vector.at(i).first;
      double feature_value = feature_vector.at(i).second;
      if (named_means.find(feature_name) != named_means.end() && named_means.at(feature_name) > 1e-9 && feature_value > 1e-9) {
        offset += named_means.at(feature_name) - feature_value;
        counter++;
      }
    }
    if (counter > 0) {
      offset /= counter;
    }
    return offset;
  }

  WifiObservationModel(void) {
    ProbabilityMapper probability_mapper;
    this->probability_mapper_ = probability_mapper;
    this->zero_centering_ = false;
    this->map_spatial_interval_ = 0.0;
  }

  ~WifiObservationModel() {}

 private:
  ProbabilityMapper probability_mapper_;
  bool zero_centering_;
  double map_spatial_interval_;
};

}  // namespace observation_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OBSERVATION_MODEL_WIFI_OBSERVATION_MODEL_H_
