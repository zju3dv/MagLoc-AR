/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-04-12 16:34:48
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-04-12 16:48:16
 */
#include "observation_model/fusion_observation_model.h"

#include <string>
#include <unordered_map>
#include <utility>

#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace observation_model {

int FusionObservation::GetObservationFromLines(
    const std::vector<std::string>& bluetooth_lines,
    const std::vector<std::string>& wifi_lines,
    const std::vector<std::string>& geomagnetism_lines,
    const std::vector<std::string>& gravity_lines,
    double time_unit_in_second) {
  this->bluetooth_observation_.GetObservationFromLines(bluetooth_lines, time_unit_in_second, BluetoothIdentityType::kUUIDMajorMinor);
  this->wifi_observation_.GetObservationFromLines(wifi_lines, time_unit_in_second);
  this->geomagnetism_observation_.GetObservationFromLines(geomagnetism_lines, time_unit_in_second);
  this->geomagnetism_observation_.GetGravityFromLines(gravity_lines, timestamp_ / time_unit_in_second);
  this->geomagnetism_observation_.GetRsgsFromGravity();
  return bluetooth_lines.size();
}

int FusionObservation::GetObservationFromData(
    const std::vector<BeaconSignalData>& bluetooth_data,
    const std::vector<WifiSignalData>& wifi_data,
    const std::vector<GeomagnetismData>& geomagnetism_data,
    const std::vector<GravityData>& gravity_data) {
  return -1;
}

std::vector<std::pair<std::string, double>>
FusionObservation::GetFeatureVector(void) {
  std::vector<std::pair<std::string, double>> bluetooth_feature_vector = this->bluetooth_observation_.GetFeatureVector();
  std::vector<std::pair<std::string, double>> wifi_feature_vector = this->wifi_observation_.GetFeatureVector();
  std::vector<std::pair<std::string, double>> geomagnetism_feature_vector = this->geomagnetism_observation_.GetFeatureVector();
  std::vector<std::pair<std::string, double>> fusion_feature_vector;
  for (int i = 0; i < bluetooth_feature_vector.size(); i++) {
    fusion_feature_vector.push_back(wifi_feature_vector.at(i));
  }
  for (int i = 0; i < wifi_feature_vector.size(); i++) {
    fusion_feature_vector.push_back(bluetooth_feature_vector.at(i));
  }
  for (int i = 0; i < geomagnetism_feature_vector.size(); i++) {
    fusion_feature_vector.push_back(geomagnetism_feature_vector.at(i));
  }
  return fusion_feature_vector;
}

void FusionObservationState::GetAllNamedValues(std::vector<std::pair<std::string, double>>* named_values) {
  named_values->clear();
  std::vector<std::pair<std::string, double>> sub_named_values;
  this->bluetooth_observation_state_.GetAllNamedValues(&sub_named_values);
  for (int i = 0; i < sub_named_values.size(); i++) {
    named_values->emplace_back(sub_named_values.at(i));
  }
  this->wifi_observation_state_.GetAllNamedValues(&sub_named_values);
  for (int i = 0; i < sub_named_values.size(); i++) {
    named_values->emplace_back(sub_named_values.at(i));
  }
  this->geomagnetism_observation_state_.GetAllNamedValues(&sub_named_values);
  for (int i = 0; i < sub_named_values.size(); i++) {
    named_values->emplace_back(sub_named_values.at(i));
  }
}

}  // namespace observation_model

}  // namespace state_estimation
