/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:16:46
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-21 15:31:44
 */
#ifndef STATE_ESTIMATION_OBSERVATION_MODEL_BLUETOOTH_OBSERVATION_MODEL_H_
#define STATE_ESTIMATION_OBSERVATION_MODEL_BLUETOOTH_OBSERVATION_MODEL_H_

#include <iostream>
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

static const int kExclusiveMinValidRssiValue = -100;
static const int kExclusiveMaxValidRssiValue = 0;
static const int kMaxFeatures = 1000;

enum class BluetoothIdentityType {
  kUUIDMajorMinor = 0,
  kUUIDMajorMinorMAC,
};

class BeaconSignalData {
 public:
  typedef uint32_t beacon_t;
  // Beacondata from collect line
  double timestamp_ = 0;
  double time_unit_in_second_ = 1.0;
  // Identity_str: Identity of Signal : Name or UUID_Major_Minor
  std::string Name_, UUID_, Major_, Minor_, Identity_str_, FactoryID_, MAC_;
  int RSSI_, num_items_;

  // stable data of Beacon
  // Sorted BeaconID in RSSI vec
  beacon_t ID_vec_;
  // Beacon's dominate AreaStr
  std::string Area_domin_;
  // RSSI data
  int rssi_mean, rssi_max, times_recv;
  // Site String
  std::string Site_Code_domin_;

  // init for 9 items
  BeaconSignalData(double timestamp, std::string Name,
                   std::string UUID, std::string Major,
                   std::string Minor, std::string FactoryID,
                   std::string MAC, int RSSI)
      : timestamp_(timestamp), Name_(Name), UUID_(UUID), Major_(Major), Minor_(Minor), Identity_str_(UUID + "_" + Major + "_" + Minor), FactoryID_(FactoryID), MAC_(MAC), RSSI_(RSSI), num_items_(9) {}

  // init for 5 items
  BeaconSignalData(double timestamp, std::string Name, int RSSI)
      : timestamp_(timestamp), Name_(Name), RSSI_(RSSI), num_items_(5) {}

  BeaconSignalData(void) {}

  inline bool HaveName() const;

  inline bool HaveArea() const;

  // To verify Beacondata is legalLocationBeacon or wild BlueToothdata
  bool IsLegalSignalinSite(std::string SiteCode, std::string UUID_pre = "");

  ~BeaconSignalData() {}
};

bool GetBeaconSignalDataFromLine(const std::string& line_data, BeaconSignalData& beacon_signal_data, double time_unit_in_second = 1.0);

class BluetoothObservation : public Observation {
 public:
  void Init(double buffer_duration, double timestamp, double time_unit_in_second = 1.0) {
    this->buffer_duration_ = buffer_duration;
    this->timestamp(timestamp, time_unit_in_second);
  }

  int GetObservationFromLines(const std::vector<std::string>& bluetooth_lines, double time_unit_in_second, BluetoothIdentityType identity_type = BluetoothIdentityType::kUUIDMajorMinor);
  int GetObservationFromBeaconSignalDatas(const std::vector<BeaconSignalData>& beacon_signal_datas, BluetoothIdentityType identity_type = BluetoothIdentityType::kUUIDMajorMinor);
  std::vector<std::pair<std::string, double>> GetFeatureVector(void);
  std::vector<std::pair<std::string, double>> GetZeroCenteredFeatureVector(void);

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

  BluetoothObservation(void);
  ~BluetoothObservation();

 private:
  std::unordered_map<std::string, std::vector<int>> feature_rssis_;
  double buffer_duration_;
  double timestamp_;
  double time_unit_in_second_;
};

class BluetoothObservationState : public State {
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
      } else if (named_values.at(i).first == util::kNameBluetoothDynamicOffset) {
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
    named_values->emplace(named_values->end(), util::kNameBluetoothDynamicOffset, this->observation_dynamic_offset_);
  }

  BluetoothObservationState(void);
  ~BluetoothObservationState();

 private:
  variable::Position position_;
  double observation_dynamic_offset_;
};

template <typename ProbabilityMapper>
class BluetoothObservationModel : public ObservationModel {
 public:
  void Init(ProbabilityMapper probability_mapper, double map_spatial_interval, bool is_zero_centered) {
    this->probability_mapper_ = probability_mapper;
    this->probability_mapper_.SetZeroCentering(is_zero_centered);
    this->zero_centering_ = is_zero_centered;
    this->map_spatial_interval_ = map_spatial_interval;
  }

  double GetProbabilityObservationConditioningState(Observation* observation, State* state) const {
#ifdef DEBUG_FOCUSING
    std::cout << "BluetoothObservationModel::GetProbabilityObservationConditioningState" << std::endl;
#endif
    BluetoothObservation* my_observation =
        reinterpret_cast<BluetoothObservation*>(observation);
    BluetoothObservationState* my_state =
        reinterpret_cast<BluetoothObservationState*>(state);

    double dynamic_map_offset = my_state->observation_dynamic_offset();
    variable::Position position = my_state->position();
    position.Round(this->map_spatial_interval_);
    return this->probability_mapper_.CalculateProbabilityFeatureVectorConditioningState(
        my_observation->GetFeatureVector(), position.ToKey(), dynamic_map_offset);
  }

  double GetProbabilityObservationConditioningStateLog(Observation* observation, State* state) const {
#ifdef DEBUG_FOCUSING
    std::cout << "BluetoothObservationModel::GetProbabilityObservationConditioningStateLog" << std::endl;
#endif
    BluetoothObservation* my_observation =
        reinterpret_cast<BluetoothObservation*>(observation);
    BluetoothObservationState* my_state =
        reinterpret_cast<BluetoothObservationState*>(state);

    double dynamic_map_offset = my_state->observation_dynamic_offset();
    variable::Position position = my_state->position();
    position.Round(this->map_spatial_interval_);
    return this->probability_mapper_.CalculateProbabilityFeatureVectorConditioningStateLog(
        my_observation->GetFeatureVector(), position.ToKey(), dynamic_map_offset);
  }

  std::unordered_map<std::string, double> GetProbabilityStatesConditioningObservation(Observation* observation) const {
    BluetoothObservation* my_observation =
        reinterpret_cast<BluetoothObservation*>(observation);

    return this->probability_mapper_.CalculateProbabilityStatesConditioningFeatureVector(my_observation->GetFeatureVector());
  }

  double CalculateBluetoothOffset(BluetoothObservation observation, BluetoothObservationState gt_state) {
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

  BluetoothObservationModel(void) {
    ProbabilityMapper probability_mapper;
    this->probability_mapper_ = probability_mapper;
    this->zero_centering_ = false;
    this->map_spatial_interval_ = 0.0;
  }

  ~BluetoothObservationModel() {}

 private:
  ProbabilityMapper probability_mapper_;
  bool zero_centering_;
  double map_spatial_interval_;
};

}  // namespace observation_model

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OBSERVATION_MODEL_BLUETOOTH_OBSERVATION_MODEL_H_
