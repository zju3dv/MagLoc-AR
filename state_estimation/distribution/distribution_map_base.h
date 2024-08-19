/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-01 11:11:34
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-01 11:21:30
 */
#ifndef STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_BASE_H_
#define STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_BASE_H_

#include <set>
#include <string>

namespace state_estimation {

namespace distribution {

class DistributionMapBase {
 public:
  // insert map elements from map_files;
  virtual int Insert(std::string distribution_map_filepath) = 0;
  // update map elements from map_files;
  virtual int Update(std::string distribution_map_filepath) = 0;
  // get all keys of the map (e.g. locations for traditional maps);
  virtual std::set<std::string> GetAllKeys(void) = 0;
  // clear the map storage;
  virtual void Clear(void) = 0;
  // get the size of the map;
  virtual int GetSize(void) = 0;
};

}  // namespace distribution

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_DISTRIBUTION_DISTRIBUTION_MAP_BASE_H_
