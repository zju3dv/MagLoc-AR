#include "imu_predict.h"  // NOLINT

#include <iomanip>
#include <string>
#include <vector>

#include "common/Utility.h"

namespace SenseSLAM {

IMU_Predict::IMU_Predict(
    const int& predict_frame_num, const std::string& result_path,
    const STSLAMCommon::CameraCalibration& cam_calibration_l)
    : m_predict_frame_num(predict_frame_num), last_vioState_time(-1.) {
  r_ci = cam_calibration_l.r_ic.transpose();
  t_ci = -(r_ci * cam_calibration_l.t_ic);
  ofs_all_predict.open(result_path + "/imu_predict_all.csv");
  ofs_imu_predict_image_freq.open(result_path + "/imu_predict_image_freq.csv");
  ofs_image_state.open(result_path + "/image_state.csv");
}

void IMU_Predict::addVIOState(const STSLAMCommon::VIOState& state) {
  vioState_history.emplace_back(state);
  run(state);
}

void IMU_Predict::addIMUdata(const std::vector<STSLAMCommon::IMUData>& imus) {
  for (const auto& tmp : imus) imus_buffer.emplace_back(tmp);
}

std::vector<STSLAMCommon::VIOState> IMU_Predict::PropagateIMU(
    const std::vector<STSLAMCommon::IMUData>& v_imus_data,
    const Eigen::Vector3f& t_ci, const Eigen::Matrix3f& r_ci,
    const STSLAMCommon::VIOState& start_state,
    STSLAMCommon::VIOState* p_end_state) {
  const Eigen::Vector3f gravity{0.0, 0.0, 9.81};
  std::vector<STSLAMCommon::VIOState> all_state;

  Eigen::Vector3f ba0 = start_state.ba;
  Eigen::Vector3f bw0 = start_state.bw;
  Eigen::Vector3f acc_0, gyro_0;
  double time_0;
  acc_0 = start_state.imu_data.acc;
  gyro_0 = start_state.imu_data.gyr;
  time_0 = start_state.imu_data.t;

  Eigen::Vector3f end_t_wi = start_state.t_wc + start_state.r_wc * t_ci;
  Eigen::Matrix3f end_r_wi = start_state.r_wc * r_ci;
  p_end_state->v = start_state.v;
  p_end_state->ba = start_state.ba;
  p_end_state->bw = start_state.bw;

  for (auto& it : v_imus_data) {
    double dt = (it.t - time_0);
    // midpoint integration
    Eigen::Vector3f un_acc_0 = end_r_wi * (acc_0 - ba0) - gravity;
    Eigen::Vector3f un_gyr = 0.5 * (gyro_0 + it.gyr) - bw0;
    end_r_wi = end_r_wi * STSLAMCommon::deltaQ(un_gyr * dt).toRotationMatrix();
    STSLAMCommon::Reorthogonalize(end_r_wi);

    Eigen::Vector3f un_acc_1 = end_r_wi * (it.acc - ba0) - gravity;
    Eigen::Vector3f un_acc = 0.5 * (un_acc_0 + un_acc_1);
    end_t_wi += dt * p_end_state->v + 0.5 * dt * dt * un_acc;
    p_end_state->v += dt * un_acc;
    acc_0 = it.acc;
    gyro_0 = it.gyr;
    time_0 = it.t;

    p_end_state->r_wc = end_r_wi * r_ci.transpose();
    p_end_state->t_wc = end_t_wi - end_r_wi * r_ci.transpose() * t_ci;
    p_end_state->timestamp = it.t;
    p_end_state->imu_data = it;
    all_state.push_back(*p_end_state);
  }
  return all_state;
}

void IMU_Predict::run(const STSLAMCommon::VIOState& end) {
  if (vioState_history.size() > m_predict_frame_num) {
    STSLAMCommon::VIOState start = vioState_history.front();
    vioState_history.pop_front();
    STSLAMCommon::VIOState second = vioState_history.front();

    STSLAMCommon::VIOState end_state;
    std::vector<STSLAMCommon::IMUData> v_imus_data;
    for (auto it = imus_buffer.begin(); it != imus_buffer.end();) {
      if (it->t > start.timestamp) v_imus_data.push_back(*it);
      if (it->t < second.timestamp)
        it = imus_buffer.erase(it);
      else
        it++;
    }

    std::vector<STSLAMCommon::VIOState> output =
        PropagateIMU(v_imus_data, t_ci, r_ci, start, &end_state);

    ofs_image_state << std::fixed << std::setprecision(6) << end.timestamp
                    << "," << end.t_wc(0) << "," << end.t_wc(1) << ","
                    << end.t_wc(2) << "," << end.r_wc.w() << "," << end.r_wc.x()
                    << "," << end.r_wc.y() << "," << end.r_wc.z() << ","
                    << end.v(0) << "," << end.v(1) << "," << end.v(2) << ","
                    << end.bw(0) << "," << end.bw(1) << "," << end.bw(2) << ","
                    << end.ba(0) << "," << end.ba(1) << "," << end.ba(2)
                    << std::endl;

    ofs_imu_predict_image_freq
        << std::fixed << std::setprecision(6) << end_state.timestamp << ","
        << end_state.t_wc(0) << "," << end_state.t_wc(1) << ","
        << end_state.t_wc(2) << "," << end_state.r_wc.w() << ","
        << end_state.r_wc.x() << "," << end_state.r_wc.y() << ","
        << end_state.r_wc.z() << "," << end_state.v(0) << "," << end_state.v(1)
        << "," << end_state.v(2) << "," << end_state.bw(0) << ","
        << end_state.bw(1) << "," << end_state.bw(2) << "," << end_state.ba(0)
        << "," << end_state.ba(1) << "," << end_state.ba(2) << std::endl;

    // traj_all_predict << output;
    for (const auto& item : output) {
      if (item.timestamp > last_vioState_time) {
        ofs_all_predict << std::fixed << std::setprecision(6) << item.timestamp
                        << "," << item.t_wc(0) << "," << item.t_wc(1) << ","
                        << item.t_wc(2) << "," << item.r_wc.w() << ","
                        << item.r_wc.x() << "," << item.r_wc.y() << ","
                        << item.r_wc.z() << "," << item.v(0) << "," << item.v(1)
                        << "," << item.v(2) << "," << item.bw(0) << ","
                        << item.bw(1) << "," << item.bw(2) << "," << item.ba(0)
                        << "," << item.ba(1) << "," << item.ba(2) << std::endl;
      }
    }
  }
  last_vioState_time = end.timestamp;
}

}  // namespace SenseSLAM
