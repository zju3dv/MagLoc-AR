/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:16:30
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-20 16:25:20
 */
#include "observation_model/bluetooth_observation_model.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <algorithm>

#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace observation_model {

bool GetBeaconSignalDataFromLine(const std::string& line_data, BeaconSignalData& beacon_data, double time_unit_in_second) {
  std::vector<std::string> one_signal_line;
  SplitString(line_data, one_signal_line, ",");
  beacon_data.time_unit_in_second_ = time_unit_in_second;
  // rule linedata
  // for 5 items
  if (one_signal_line.size() == 5) {
    beacon_data.Name_ = one_signal_line[2];
    beacon_data.RSSI_ = stoi(one_signal_line[4]);
    // for IOS data
    if (beacon_data.RSSI_ == 0) beacon_data.RSSI_ = -98;

    beacon_data.timestamp_ = atof(one_signal_line[1].c_str());
    // beacon_data.Identity_str_ = beacon_data.Name_;
    beacon_data.num_items_ = 5;
    return true;
    // for 9 items
  } else if (one_signal_line.size() == 9) {
    beacon_data.Name_ = one_signal_line[2];
    beacon_data.MAC_ = one_signal_line[7];
    beacon_data.UUID_ = ToLower(one_signal_line[3]);
    beacon_data.Major_ = one_signal_line[4];
    beacon_data.Minor_ = one_signal_line[5];
    //// Identity_str
    // beacon_data.Identity_str_ = beacon_data.Name_;
    // beacon_data.Identity_str = one_signal_line[3]
    // + "_" + one_signal_line[4] + "_" + one_signal_line[6] ;
    // FactoryID = first number?
    beacon_data.Identity_str_ = beacon_data.UUID_;
    beacon_data.Identity_str_.append("_")
                             .append(beacon_data.Major_)
                             .append("_")
                             .append(beacon_data.Minor_)
                             .append("_")
                             .append(beacon_data.MAC_);
    beacon_data.FactoryID_ = one_signal_line[6];
    beacon_data.timestamp_ = atof(one_signal_line[1].c_str());
    beacon_data.RSSI_ = stoi(one_signal_line[8]);
    if (beacon_data.RSSI_ == 0) beacon_data.RSSI_ = -100;
    beacon_data.num_items_ = 9;
    return true;
  } else if (one_signal_line.size() == 10) {
    beacon_data.Name_ = one_signal_line[2];
    beacon_data.MAC_ = one_signal_line[7];
    beacon_data.UUID_ = ToLower(one_signal_line[3]);
    beacon_data.Major_ = one_signal_line[4];
    beacon_data.Minor_ = one_signal_line[5];
    //// Identity_str
    // beacon_data.Identity_str_ = beacon_data.Name_;
    // beacon_data.Identity_str = one_signal_line[3]
    // + "_" + one_signal_line[4] + "_" + one_signal_line[6] ;
    // FactoryID = first number?
    beacon_data.Identity_str_ = beacon_data.UUID_;
    beacon_data.Identity_str_.append("_")
                             .append(beacon_data.Major_)
                             .append("_")
                             .append(beacon_data.Minor_)
                             .append("_")
                             .append(beacon_data.MAC_);
    beacon_data.FactoryID_ = one_signal_line[6];
    beacon_data.timestamp_ = atof(one_signal_line[1].c_str());
    beacon_data.RSSI_ = stoi(one_signal_line[8]);
    if (beacon_data.RSSI_ == 0) beacon_data.RSSI_ = -100;
    beacon_data.num_items_ = 10;
    return true;
  }
  return false;
}

int BluetoothObservation::GetObservationFromLines(
    const std::vector<std::string>& bluetooth_lines,
    double time_unit_in_second,
    BluetoothIdentityType identity_type) {
  std::vector<BeaconSignalData> beacon_signal_datas;
  for (int i = 0; i < bluetooth_lines.size(); i++) {
    BeaconSignalData beacon_signal_data;
    if (GetBeaconSignalDataFromLine(bluetooth_lines.at(i), beacon_signal_data, time_unit_in_second)) {
      beacon_signal_datas.push_back(std::move(beacon_signal_data));
    }
  }
  return this->GetObservationFromBeaconSignalDatas(beacon_signal_datas, identity_type);
}

int BluetoothObservation::GetObservationFromBeaconSignalDatas(const std::vector<BeaconSignalData>& beacon_signal_datas, BluetoothIdentityType identity_type) {
  // assume that the beacon_signal_datas are sorted in time order.
  // the last is the most current;
  this->feature_rssis_.clear();
  if (beacon_signal_datas.size() == 0) {
    return 0;
  }

  std::string beaconKey;
  std::unordered_map<std::string, std::vector<int>> beacon_id_to_rssis;
  double current_t = this->timestamp_ * this->time_unit_in_second_;
  if (current_t < 0.0) {
    std::cout << "BluetoothObservation::GetObservationFromBeaconSignalDatas: the timestamp of BluetoothObservation is not assigned." << std::endl;
    return 0;
  }
  int valid_data_counter = 0;

  for (int i = beacon_signal_datas.size() - 1; i >= 0; i--) {
    BeaconSignalData signal_raw_data = beacon_signal_datas.at(i);
    double data_t = signal_raw_data.timestamp_ * signal_raw_data.time_unit_in_second_;
    if (data_t > current_t) {
      // the data is in the future.
      continue;
    }
    if (current_t - data_t > this->buffer_duration_) {
      // the data is out-of-date.
      break;
    }
    if ((signal_raw_data.RSSI_ <= kExclusiveMinValidRssiValue) ||
        (signal_raw_data.RSSI_ >= kExclusiveMaxValidRssiValue)) {
      continue;
    }
    valid_data_counter++;
    beaconKey = signal_raw_data.UUID_;
    switch (identity_type) {
      case BluetoothIdentityType::kUUIDMajorMinor:
        beaconKey = beaconKey.append("_")
                             .append(signal_raw_data.Major_)
                             .append("_")
                             .append(signal_raw_data.Minor_);
        break;
      case BluetoothIdentityType::kUUIDMajorMinorMAC:
        beaconKey = beaconKey.append("_")
                             .append(signal_raw_data.Major_)
                             .append("_")
                             .append(signal_raw_data.Minor_)
                             .append("_")
                             .append(signal_raw_data.MAC_);
        break;
      default:
        std::cout << "BluetoothObservation::GetObservationFromLines: BluetoothIdentityType "
                  << static_cast<int>(identity_type)
                  << " is not allowed. Use the default kUUIDMajorMinor."
                  << std::endl;
        beaconKey = beaconKey.append("_")
                             .append(signal_raw_data.Major_)
                             .append("_")
                             .append(signal_raw_data.Minor_);
    }
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

  return valid_data_counter;
}

std::vector<std::pair<std::string, double>>
BluetoothObservation::GetFeatureVector(void) {
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

  return mean_rssis;
}

std::vector<std::pair<std::string, double>>
BluetoothObservation::GetZeroCenteredFeatureVector(void) {
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

BluetoothObservation::BluetoothObservation(void) {
  this->feature_rssis_ = std::unordered_map<std::string, std::vector<int>>();
  this->buffer_duration_ = 0.0;
  this->timestamp_ = -1.0;
  this->time_unit_in_second_ = 1.0;
}

BluetoothObservation::~BluetoothObservation() {}

BluetoothObservationState::BluetoothObservationState(void) {
  variable::Position position;
  this->position_ = position;
  this->observation_dynamic_offset_ = 0.0;
}

BluetoothObservationState::~BluetoothObservationState() {}

}  // namespace observation_model

}  // namespace state_estimation
