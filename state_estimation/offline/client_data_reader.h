/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:17:53
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-24 14:37:44
 */
#ifndef STATE_ESTIMATION_OFFLINE_CLIENT_DATA_READER_H_
#define STATE_ESTIMATION_OFFLINE_CLIENT_DATA_READER_H_

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

#include "offline/sequential_reader_base.h"
#include "util/client_request.h"
#include "util/misc.h"
#include "variable/position.h"

namespace state_estimation {

namespace offline {

class ClientDataReader : public SequentialReader {
 public:
  int Init(std::string client_folder_path);

  std::vector<std::string> client_request_folder_names(void) {
    return this->client_request_folder_names_;
  }

  int GetSize(void) {
    return this->client_request_folder_names_.size();
  }

  std::unordered_map<std::string, std::string> timestamp_ms_str_to_folder_name(void) {
    return this->timestamp_ms_str_to_folder_name_;
  }

  std::unordered_map<int, std::string> index_to_folder_name(void) {
    return this->index_to_folder_name_;
  }

  void SortRequestFolderPaths(bool reverse = false);
  util::ClientRequest GetRequestByRequestFoldername(std::string request_folder_name);
  util::ClientRequest GetNextRequest(void);
  util::ClientRequest GetRequestByTimestampMsStr(std::string timestamp_ms_str);
  util::ClientRequest GetRequestByIndex(int request_index);

 private:
  std::string client_folder_path_;
  std::vector<std::string> client_request_folder_names_;
  std::unordered_map<std::string, std::string> timestamp_ms_str_to_folder_name_;
  std::unordered_map<int, std::string> index_to_folder_name_;
};

}  // namespace offline

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_OFFLINE_CLIENT_DATA_READER_H_
