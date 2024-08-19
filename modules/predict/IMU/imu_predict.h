#ifndef SRC_IMU_IMU_PREDICT_H_
#define SRC_IMU_IMU_PREDICT_H_

#include <fstream>
#include <list>
#include <string>
#include <vector>

#include "common/DataHelp.h"

namespace SenseSLAM {

class IMU_Predict {
 public:
  IMU_Predict(const int& predict_frame_num, const std::string& result_path,
              const STSLAMCommon::CameraCalibration& cam_calibration_l);
  void addVIOState(const STSLAMCommon::VIOState& state);
  void addIMUdata(const std::vector<STSLAMCommon::IMUData>& imus);

  static std::vector<STSLAMCommon::VIOState> PropagateIMU(
      const std::vector<STSLAMCommon::IMUData>& v_imus_data,
      const Eigen::Vector3f& t_ci, const Eigen::Matrix3f& r_ci,
      const STSLAMCommon::VIOState& start_state,
      STSLAMCommon::VIOState* p_end_state);

 private:
  void run(const STSLAMCommon::VIOState& end);

 private:
  int m_predict_frame_num;  // 1, 2, 3......
  Eigen::Vector3f t_ci;
  Eigen::Matrix3f r_ci;
  double last_vioState_time = -1;

  std::ofstream ofs_all_predict;
  std::ofstream ofs_imu_predict_image_freq;
  std::ofstream ofs_image_state;

  // for IMU prediction module
  std::list<STSLAMCommon::VIOState> vioState_history;
  std::list<STSLAMCommon::IMUData> imus_buffer;
};

}  // namespace SenseSLAM
#endif  // SRC_IMU_IMU_PREDICT_H_
