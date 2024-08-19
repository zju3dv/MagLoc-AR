#ifndef SRC_PREDICT_PREDICTFILTER_H_
#define SRC_PREDICT_PREDICTFILTER_H_

#include <glog/logging.h>

#include <Eigen/Dense>
#include <deque>
#include <mutex>
#include <vector>

#include "common/DataHelp.h"
#include "IMU/IMUPreIntegration.h"
#include "common/common_type.h"
#include "sophus/so3.h"

namespace ST_Predict {

double CurrentTimestampS();

std::string GetTimeStampString();

namespace LieAlgebra {

using namespace Eigen; // NOLINT

Matrix3d hat(const Vector3d &phi);

Matrix3d exp(const Vector3d &phi);

Vector3d log(const Matrix3d &R);

Matrix3d left_jacobian(const Vector3d &phi);

Matrix3d left_jacobian(const Matrix3d &R);

Matrix3d right_jacobian(const Eigen::Vector3d &phi);

Matrix3d right_jacobian(const Matrix3d &R);

Matrix3d inv_left_jacobian(const Vector3d &phi);

Matrix3d inv_left_jacobian(const Matrix3d &R);

Matrix3d inv_right_jacobian(const Vector3d &phi);

Matrix3d inv_right_jacobian(const Matrix3d &R);
} // namespace LieAlgebra

class
PredictFilter {
public:
  PredictFilter( PredictParameter pp); // explicit
  void Predict(const double &predict_t,
               const std::vector<STSLAMCommon::IMUData> &imu_datas, const int stage);

  std::vector<STSLAMCommon::IMUData> LowerPassFilter(const std::vector<STSLAMCommon::IMUData> &imu_datas);

//  std::vector<double> ButterFilter(const std::vector<double> &data,const double fps, double low_pass_f,double high_pass_f);
//
//  std::vector<STSLAMCommon::IMUData> ButterFilterIMUData(const std::vector<STSLAMCommon::IMUData> &imu_datas);
//

  void UpdateVIOState(const VIOState &vio_state);

  void PredictInterface(const double &predict_t,
                        const std::vector<STSLAMCommon::IMUData> &imu_datas,const VIOState &vio_state,
                        const PredictInterfaceInputParameter &predictInterfaceInputParameter );

  PredictState GetSimulationPose(const double &predict_t);


  PredictState GetFirstFilterState();
  PredictState GetSmoothFilterState();

  PredictState GetSmoothPredictState(const double &predict_t);
  PredictState Get_IMUIntegration_PRV();
  double GetPredictTimeIntervel();
  void SaveOnlineData(PredictInterfaceInputParameter predictInterfaceInputParameter);

private:
  PredictState GetPredictState();

  void PredictRT(double predict_t);

  void UpdateRT(double predict_t);

  void UpdatePosition(double predict_t);

  void UpdateRotation(double predict_t);

  void IMUPreIntegration(const STSLAMCommon::IMUData &imu_data);

  void UpdateCoarseState();

  void UpdateSmoothWithoutChangeState(bool update_t, bool update_r,
                                                     const PredictState &cur_predict_state,
                                                     const Eigen::Matrix<double, 12, 12> &cur_state_t_cov,
                                                     const Eigen::Matrix<double, 12, 12> &cur_state_r_cov,
                                                     Eigen::Matrix<double, 12, 12> &cur_smooth_t_cov,
                                                     Eigen::Matrix<double, 12, 12> &cur_smooth_r_cov,
                                                     PredictState &cur_smooth_state);

  void UpdateSmoothState(bool update_t, bool update_r);

  void PredictModel(const double predict_time, const PredictState &last_state,
                    PredictState &cur_state,           // NOLINT
                    Eigen::MatrixXd &jacobian_t_state, // NOLINT
                    Eigen::MatrixXd &jacobian_t_noise, // NOLINT
                    Eigen::MatrixXd &jacobian_r_state, // NOLINT
                    Eigen::MatrixXd &jacobian_r_noise, // NOLINT
                    const double &scale_t, const bool &calculate_jacobian);

  void PredictModelOptional(const double predict_time,
                                           const PredictState &last_state,
                                           PredictState &cur_state, // NOLINT
                                           Eigen::MatrixXd &jacobian_t_state,
                                           Eigen::MatrixXd &jacobian_t_noise, // NOLINT
                                           Eigen::MatrixXd &jacobian_r_state, // NOLINT
                                           Eigen::MatrixXd &jacobian_r_noise, // NOLINT
                                           const double &scale_t,
                                           const bool &calculate_jacobian,
                                           const bool predict_t_flag,
                                           const bool predict_r_flag);

  int predict_num_ =0;
  double last_predict_t_=0;

  double predict_threshold=0.025;

  Eigen::Vector3d gravity_{0.0, 0.0, 9.81};
  double scale_t_;        // 0.002s --> 1
  double smooth_scale_t_; // 0.011 --> 1

  PredictState first_filter_predict_state_;
  PredictState smooth_filter_predict_state_;

  double predict_time_intervel_;

//  Eigen::Vector3d t_ic_{0.004100979075940995,0.022231619796053936,-0.0048932009126057725};  // left camera to imu
//  Eigen::Matrix3d r_ic_;

  bool initial_;
  bool first_predict_t_ = true;

  double latest_imu_t_ = -1; // 送进预测器的最新的imu的时间
  bool vio_state_change_ = false;
  int latest_vio_id_ = -1; // 记录UpdateVIOState更新的最新id

  int vio_update_num=0;

  double latest_t_;
  Eigen::Vector3d acc_0_, gyr_0_;
  Eigen::Vector3d linearized_ba_, linearized_bg_;
  Eigen::Vector3d last_linearized_ba_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_linearized_bg_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_angular_vel_ = Eigen::Vector3d ::Zero();
  Eigen::Vector3d latest_p_, latest_v_, latest_angular_vel_;
  Eigen::Matrix3d latest_q_;

  std::deque<PredictState> states_buf_; // result

  PredictState last_predict_state_, cur_predict_state_;
  Eigen::Matrix<double, 12, 12> last_state_t_cov_, cur_state_t_cov_;
  Eigen::Matrix<double, 12, 12> last_state_r_cov_, cur_state_r_cov_;
  Eigen::Matrix<double, 12, 12> model_t_noise_cov_, model_r_noise_cov_;

  // output kalman filter state, for smooth
  PredictState last_smooth_state_, cur_smooth_state_;
  Eigen::Matrix<double, 12, 12> last_smooth_t_cov_, cur_smooth_t_cov_;
  Eigen::Matrix<double, 12, 12> last_smooth_r_cov_, cur_smooth_r_cov_;
  Eigen::Matrix<double, 12, 12> smooth_model_t_noise_cov_,
      smooth_model_r_noise_cov_;

  Eigen::Matrix<double, 6, 6> R_t_meas_, R_r_meas_;
  Eigen::Matrix3d R_meas_p_, R_meas_v_, R_meas_q_;
  Eigen::Matrix3d latest_vio_q_;

  std::mutex mtx_;

  SenseSLAM::SenseVIO::IMUPreIntegration imu_pre_integration_;
};
} // namespace ST_Predict
#endif // SRC_PREDICT_PREDICTFILTER_H_
