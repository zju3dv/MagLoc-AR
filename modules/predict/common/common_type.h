#ifndef SRC_COMMON_TYPE_H_
#define SRC_COMMON_TYPE_H_

#include <Eigen/Dense>
#include <chrono>
#include <cfloat>

#include "DataHelp.h"

namespace ST_Predict {

struct PredictInterfaceInputParameter{
  bool save_online_data; //save control flag
  int stage; //render or warping
  double predict_real_time;
  std::string vio_state_file_path;
  std::string imu_file_path;
  std::string predict_stage_file_path;
  PredictInterfaceInputParameter() :
      save_online_data(false),stage(-1),predict_real_time(-1),
      vio_state_file_path("./Predict_vio_state.csv"),
      imu_file_path("./Predict_imu.csv"),
      predict_stage_file_path("./PredictPoseStage.csv") {};
};

struct PredictParameter {
  double imu_freq; // 0.002s --> 1
  double render_freq; // 0.013 --> 1
};

//struct OnlineDataSaveFilePathParameter {
//  std::string vio_state_file_path;
//  std::string imu_file_path;
//  std::string predict_stage_file_path;
//  std::string current_time_str=GetTimeStampString();
//  OnlineDataSaveFilePathParameter() :
//      vio_state_file_path("/Predict_vio_state-"+current_time_str+".csv"), imu_file_path("/Predict_imu-"+current_time_str+".csv"),predict_stage_file_path("/PredictPoseStage-"+current_time_str+".csv") {};
//};



struct VIOState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int frame_id;
  bool b_state_update; //default false
  double timestamp;
  Eigen::Vector3f t_wi;
  Eigen::Quaternionf r_wi;
  Eigen::Vector3f v;
  Eigen::Vector3f bw;
  Eigen::Vector3f ba;
  STSLAMCommon::IMUData imu_data;
  VIOState() : frame_id(-1), timestamp(-1),b_state_update(false) {}
};

struct PredictInput {
  double imu_freq; // 0.002s --> 1
  double render_freq; // 0.013 --> 1

  double cur_query_t;
  VIOState vio_state;
  std::vector<STSLAMCommon::IMUData> imu_datas;
};


struct PredictState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp;
  Eigen::Vector3d t_wb;
  Eigen::Matrix3d r_wb;
  Eigen::Vector3d vel;          // v_wb
  Eigen::Vector3d angular_vel;  // w_b
  Eigen::Vector3d acc;          // accerlation
  Eigen::Vector3d angular_acc;  //
  Eigen::Vector3d acc_jerk;     // m/ms^{3}
  Eigen::Vector3d angular_jerk;

  int update_position_cnt = 0;
  int update_rotation_cnt = 0;
  int result_is_valid = 0;


  PredictState() {
    t_wb.setZero();
    r_wb.setIdentity();
    vel.setZero();
    angular_vel.setZero();
    acc.setZero();
    angular_acc.setZero();
    acc_jerk.setZero();
    angular_jerk.setZero();
    timestamp = FLT_MIN;
  }
};

}  // namespace ST_Predict
#endif  // SRC_COMMON_TYPE_H_
