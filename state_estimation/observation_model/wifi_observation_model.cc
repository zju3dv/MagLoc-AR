/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-31 14:40:47
 * @Last Modified by: xuehua
 * @Last Modified time: 2023-01-18 17:19:41
 */
#include "observation_model/wifi_observation_model.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <iostream>

#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace observation_model {

static const int kExclusiveMinValidRssiValue = -100;
static const int kExclusiveMaxValidRssiValue = 0;
static const int kMaxFeatures = 1000;

bool GetWifiSignalDataFromLine(
    const std::string& line_data, WifiSignalData& wifi_data) {
  std::vector<std::string> one_signal_line;
  SplitString(line_data, one_signal_line, ",");
  if (one_signal_line.size() == 7) {
    wifi_data.unix_timestamp_ = atof(one_signal_line[0].c_str());
    wifi_data.ssid_ = one_signal_line[1];
    wifi_data.bssid_ = one_signal_line[2];
    wifi_data.RSSI_ = stoi(one_signal_line[3]);
    wifi_data.channel_ = one_signal_line[4];
    wifi_data.identity_str_ = wifi_data.ssid_;
    wifi_data.identity_str_.append("_")
                           .append(wifi_data.bssid_)
                           .append("_")
                           .append(wifi_data.channel_);
    wifi_data.unix_timestamp_lastseen_ = atof(one_signal_line[5].c_str());
    wifi_data.system_timestamp_lastseen_ = atof(one_signal_line[6].c_str());
    wifi_data.num_items_ = 6;
    return true;
  }
  return false;
}

int WifiObservation::GetObservationFromLines(
    const std::vector<std::string>& wifi_lines,
    double time_unit_in_second) {
  WifiSignalData signal_raw_data;
  double last_timestamp = -1.0;
  std::vector<WifiSignalData> wifi_signal_data_vector;
  for (int i = 0; i < wifi_lines.size(); i++) {
    if (!GetWifiSignalDataFromLine(wifi_lines[i], signal_raw_data)) {
      continue;
    }
    if (signal_raw_data.system_timestamp_lastseen_ > last_timestamp) {
      last_timestamp = signal_raw_data.system_timestamp_lastseen_;
    }
    wifi_signal_data_vector.push_back(signal_raw_data);
  }

  std::string beaconKey;
  std::unordered_map<std::string, std::vector<int>> beacon_id_to_rssis;
  for (int i = 0; i < wifi_signal_data_vector.size(); i++) {
    signal_raw_data = wifi_signal_data_vector.at(i);
    if ((last_timestamp - signal_raw_data.system_timestamp_lastseen_) > this->buffer_duration_ / time_unit_in_second) {
      continue;
    }
    if ((signal_raw_data.RSSI_ <= kExclusiveMinValidRssiValue) ||
        (signal_raw_data.RSSI_ >= kExclusiveMaxValidRssiValue)) {
      continue;
    }
    beaconKey = signal_raw_data.ssid_;
    beaconKey = beaconKey.append("_")
                    .append(signal_raw_data.bssid_)
                    .append("_")
                    .append(signal_raw_data.channel_);
    for (int j = 0; j < beaconKey.size(); j++) {
      beaconKey[j] = tolower(beaconKey[j]);
    }
    if (beacon_id_to_rssis.find(beaconKey) == beacon_id_to_rssis.end()) {
      std::vector<int> rssis;
      rssis.push_back(signal_raw_data.RSSI_ - kExclusiveMinValidRssiValue);
      beacon_id_to_rssis.emplace(beaconKey, rssis);
    } else {
      beacon_id_to_rssis[beaconKey]
          .push_back(signal_raw_data.RSSI_ - kExclusiveMinValidRssiValue);
    }
  }

  this->feature_rssis_ = beacon_id_to_rssis;

  return wifi_lines.size();
}

std::vector<std::pair<std::string, double>>
WifiObservation::GetFeatureVector(void) {
  std::vector<std::pair<std::string, double>> mean_rssis;
  double beacon_sum;
  int num_beacon_samples;
  double beacon_mean;
  for (auto it = this->feature_rssis_.begin();
       it != this->feature_rssis_.end(); it++) {
    num_beacon_samples = it->second.size();
    if (num_beacon_samples > 0) {
      beacon_sum = 0.0;
      for (int j = 0; j < num_beacon_samples; j++) {
        beacon_sum += it->second[j];
      }
      beacon_mean = beacon_sum / num_beacon_samples;
      mean_rssis.push_back(
          std::pair<std::string, double>(it->first, beacon_mean));
    } else {
      mean_rssis.push_back(std::pair<std::string, double>(it->first, 0.0));
    }
  }

  std::vector<std::pair<std::string, double>> selected_mean_rssis;
  if (mean_rssis.size() <= kMaxFeatures) {
    selected_mean_rssis = mean_rssis;
  } else {
    std::sort(mean_rssis.begin(), mean_rssis.end(),
              [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) { return a.second > b.second; });
    for (int i = 0; i < kMaxFeatures; i++) {
      selected_mean_rssis.push_back(mean_rssis.at(i));
    }
  }

  return selected_mean_rssis;
}

std::vector<std::pair<std::string, double>>
WifiObservation::GetZeroCenteredFeatureVector(void) {
  std::vector<std::pair<std::string, double>> sub_mean_feature;
  std::vector<std::pair<std::string, double>>
      mean_rssis = this->GetFeatureVector();
  std::vector<double> tmp;
  for (auto i : mean_rssis) {
    tmp.push_back(i.second);
  }
  double mean_rssi = GetDoubleVectorMean(tmp);

  for (int i = 0; i < mean_rssis.size(); i++) {
    if (mean_rssis[i].second != 0) {
      sub_mean_feature.push_back(std::pair<std::string, double>(
          mean_rssis[i].first, mean_rssis[i].second - mean_rssi));
    }
  }

  return sub_mean_feature;
}

WifiObservation::WifiObservation(void) {
  std::unordered_map<std::string, std::vector<int>> feature_rssis;
  this->feature_rssis_ = feature_rssis;
  this->buffer_duration_ = 0.0;
}

WifiObservation::~WifiObservation() {}

WifiObservationState::WifiObservationState(void) {
  variable::Position position;
  this->position_ = position;
  this->observation_dynamic_offset_ = 0.0;
}

WifiObservationState::~WifiObservationState() {}

}  // namespace observation_model

}  // namespace state_estimation
