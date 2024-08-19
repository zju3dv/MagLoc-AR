#include "PredictFilter.h"
#include <gflags/gflags.h>
#include <iomanip>
#include <fstream>
#include "butter.h"

//#include "Helper/Timer.h"
#include "common/Utility.h"

//DEFINE_double(IMU_Freq, 500.0, "IMU frequency");
//DEFINE_double(Render_Freq, 90.0, "Render frequency");

// DEFINE_double(vio_meas_p, 1e-10,
DEFINE_double(vio_meas_p, 1e-5,  // for sensenavi_mag
              "first kalman filter, vio state p noise covariance");
DEFINE_double(vio_meas_v, 7e-6,
              "first kalman filter, vio state v noise covariance");
//DEFINE_double(vio_meas_q, 1e-10,
//              "first kalman filter, vio state q noise covariance");
//DEFINE_double(vio_meas_w, 7e-6, "first kalman filter, gyro noise covariance");


//DEFINE_double(vio_meas_p, 1e-3,
//              "first kalman filter, vio state p noise covariance");
//DEFINE_double(vio_meas_v, 1,
//              "first kalman filter, vio state v noise covariance");
//DEFINE_double(vio_meas_p, 1e-10,
//              "first kalman filter, vio state p noise covariance");
//DEFINE_double(vio_meas_v, 7e-6,
//              "first kalman filter, vio state v noise covariance");

DEFINE_double(vio_meas_q, 2e-11,
              "first kalman filter, vio state q noise covariance");
DEFINE_double(vio_meas_w, 1e-13, "first kalman filter, gyro noise covariance");

DEFINE_double(model_acc_jerk, 1e-11,
              "first kalman filter, model acc jerk noise covariance");
DEFINE_double(model_p, 1e-10, "first kalman filter, model p noise covariance");
DEFINE_double(model_v, 1e-11, "first kalman filter, model v noise covariance");
DEFINE_double(model_acc, 1e-12,
              "first kalman filter, model acc noise covariance");

DEFINE_double(smooth_model_acc_jerk, 5e-11,
              "second kalman filter, model acc jerk noise covariance");
DEFINE_double(smooth_model_p, 1e-12,
              "second kalman filter, model p noise covariance");
DEFINE_double(smooth_model_v, 1e-12,
              "second kalman filter, model v noise covariance");
DEFINE_double(smooth_model_acc, 5e-15,
              "second kalman filter, model acc noise covariance");


DEFINE_double(model_gyr_jerk, 1e-11,
              "first kalman filter, model gyro jerk noise covariance");
DEFINE_double(model_q, 1e-10,
              "first kalman filter, model rotation noise covariance");
DEFINE_double(model_w, 1e-11,
              "first kalman filter, model gyro angular noise covariance");
DEFINE_double(model_gyr, 1e-11,
              "first kalman filter, model gyro acc noise covariance");

DEFINE_double(smooth_model_gyr_jerk, 1e-8, "");
DEFINE_double(smooth_model_q, 1e-8, "");
DEFINE_double(smooth_model_w, 1e-6, "");
DEFINE_double(smooth_model_gyr, 1e-10, "");

namespace ST_Predict {

double CurrentTimestampS(){
  const long int kUsecsPerSec = 1000 * 1000;
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / ((double)kUsecsPerSec);
}

std::string GetTimeStampString() {
  std::time_t t = std::time(0);
  std::tm *now = std::localtime(&t);
  char str[150];
  snprintf(str, sizeof(str), "%4d-%02d-%02d-%02d-%02d-%02d",
           now->tm_year + 1900, now->tm_mon + 1, now->tm_mday, now->tm_hour,
           now->tm_min, now->tm_sec);
  return std::string(str);
}

namespace LieAlgebra {

using namespace Eigen; // NOLINT

constexpr float eps = 1E-6;

Matrix3d hat(const Vector3d &phi) { return Sophus::SO3::hat(phi); }

Matrix3d exp(const Vector3d &phi) { return Sophus::SO3::exp(phi).matrix(); }

Vector3d log(const Matrix3d &R) { return Sophus::SO3(R).log(); }

Matrix3d left_jacobian(const Vector3d &phi) {
  // J = sin(phi) / phi + (1 - sin(phi) / phi) * a * a^t + (1 - cos(phi)) /
  // phi * a^
  double norm = phi.norm();
  auto a = phi.normalized();

  if (norm > eps) {
    // locally on 0, |x| >= |sin(x)|
    double f = sin(norm) / norm;
    return f * Matrix3d::Identity() + (1 - f) * a * a.transpose() +
           (1 - cos(norm)) / norm * hat(a);
  } else {
    return Matrix3d::Identity() + 0.5 * hat(phi);
  }
}

Matrix3d left_jacobian(const Matrix3d &R) { return left_jacobian(log(R)); }

Matrix3d right_jacobian(const Eigen::Vector3d &phi) {
  return left_jacobian((-phi).eval());
}

Matrix3d right_jacobian(const Matrix3d &R) { return right_jacobian(log(R)); }

Matrix3d inv_left_jacobian(const Vector3d &phi) {
  // invJ = phi/2 * cot(phi/2) + (1 - phi/2 * cot(phi/2)) * a * a^t - phi/2 *
  // a^
  double norm = phi.norm(), half_norm = 0.5 * norm;
  auto a = phi.normalized();

  if (half_norm > eps) {
    // locally on 0, |tan(x)| >= |x|
    double f = half_norm / std::tan(half_norm);
    return f * Matrix3d::Identity() + (1 - f) * a * a.transpose() -
           0.5 * hat(phi);
  } else {
    return Matrix3d::Identity() - 0.5 * hat(phi);
  }
}

Matrix3d inv_left_jacobian(const Matrix3d &R) {
  return inv_left_jacobian(log(R));
}

Matrix3d inv_right_jacobian(const Vector3d &phi) {
  return inv_left_jacobian((-phi).eval());
}

Matrix3d inv_right_jacobian(const Matrix3d &R) {
  return inv_right_jacobian(log(R));
}
} // namespace LieAlgebra

PredictFilter::PredictFilter( PredictParameter pp  ) {
  scale_t_ = pp.imu_freq;
  smooth_scale_t_ = pp.render_freq;

  initial_ = false;

  // second kalman filter parameters
  smooth_model_t_noise_cov_.setZero();
  smooth_model_t_noise_cov_.block<3, 3>(0, 0) =
      FLAGS_smooth_model_acc_jerk * Eigen::Matrix3d::Identity();
  smooth_model_t_noise_cov_.block<3, 3>(3, 3) =
      FLAGS_smooth_model_p * Eigen::Matrix3d::Identity();
  smooth_model_t_noise_cov_.block<3, 3>(6, 6) =
      FLAGS_smooth_model_v * Eigen::Matrix3d::Identity();
  smooth_model_t_noise_cov_.block<3, 3>(9, 9) =
      FLAGS_smooth_model_acc * Eigen::Matrix3d::Identity();

  smooth_model_r_noise_cov_.setZero();
  smooth_model_r_noise_cov_.block<3, 3>(0, 0) =
      FLAGS_smooth_model_gyr_jerk * Eigen::Matrix3d::Identity();
  smooth_model_r_noise_cov_.block<3, 3>(3, 3) =
      FLAGS_smooth_model_q * Eigen::Matrix3d::Identity();
  smooth_model_r_noise_cov_.block<3, 3>(6, 6) =
      FLAGS_smooth_model_w * Eigen::Matrix3d::Identity();
  smooth_model_r_noise_cov_.block<3, 3>(9, 9) =
      FLAGS_smooth_model_gyr * Eigen::Matrix3d::Identity();

  // first kalman filter parameters
  model_t_noise_cov_.setZero();
  model_t_noise_cov_.block<3, 3>(0, 0) =
      FLAGS_model_acc_jerk * Eigen::Matrix3d::Identity();
  model_t_noise_cov_.block<3, 3>(3, 3) =
      FLAGS_model_p * Eigen::Matrix3d::Identity();
  model_t_noise_cov_.block<3, 3>(6, 6) =
      FLAGS_model_v * Eigen::Matrix3d::Identity();
  model_t_noise_cov_.block<3, 3>(9, 9) =
      FLAGS_model_acc * Eigen::Matrix3d::Identity();

  model_r_noise_cov_.setZero();
  model_r_noise_cov_.block<3, 3>(0, 0) =
      FLAGS_model_gyr_jerk * Eigen::Matrix3d::Identity();
  model_r_noise_cov_.block<3, 3>(3, 3) =
      FLAGS_model_q * Eigen::Matrix3d::Identity();
  model_r_noise_cov_.block<3, 3>(6, 6) =
      FLAGS_model_w * Eigen::Matrix3d::Identity();
  model_r_noise_cov_.block<3, 3>(9, 9) =
      FLAGS_model_gyr * Eigen::Matrix3d::Identity();
}

std::vector<STSLAMCommon::IMUData>  PredictFilter::LowerPassFilter(const std::vector<STSLAMCommon::IMUData> &imu_datas){
  if(imu_datas.size()<2)
    return imu_datas;
  float weight=0.4;

  std::vector<STSLAMCommon::IMUData> imu_datas_lowpass;
  Eigen::Vector3f acc_raw;
  Eigen::Vector3f gyr_raw;
  Eigen::Vector3f acc_filter;
  Eigen::Vector3f gyr_filter;
  Eigen::Vector3f acc;
  Eigen::Vector3f gyr;
  double t;
  imu_datas_lowpass.push_back(imu_datas[0]);
  STSLAMCommon::IMUData imudata;
  for (int i = 1; i < imu_datas.size(); ++i) {
    STSLAMCommon::IMUData one_raw_imu_data = imu_datas[i];
    acc_raw=one_raw_imu_data.acc;
    gyr_raw=one_raw_imu_data.gyr;
    t=one_raw_imu_data.t;
    STSLAMCommon::IMUData one_filter_imu_data = imu_datas_lowpass[i-1];
    acc_filter=one_filter_imu_data.acc;
    gyr_filter=one_filter_imu_data.gyr;
    for(int j=0;j<acc_raw.size();++j){
      acc[j]= weight*acc_filter[j] + (1-weight)*acc_raw[j];
      gyr[j]= weight*gyr_filter[j] + (1-weight)*gyr_raw[j];
    }
    imudata.acc=acc;
    imudata.gyr=gyr;
    imudata.t=t;
    imu_datas_lowpass.push_back(imudata);
  }
//  std::cout<<"lower " << imu_datas_lowpass[2].acc[1] <<std::endl;
//  std::cout<<"raw: " <<imu_datas[2].acc[1] <<std::endl;

  return imu_datas_lowpass;
}

//std::vector<double> PredictFilter::ButterFilter(const std::vector<double> &data,const double fps, double low_pass_f,double high_pass_f){
//  int N = data.size();
//  std::vector<double> y(N);
//  //double fps = 500;
//  //double FrequencyBands[2] = { 0.0001/fps*2, 13/fps*2 };
//  high_pass_f=2*high_pass_f/fps;
//  low_pass_f=2*low_pass_f/fps;
//  int FiltOrd = 1;
//  std::vector<double> a;
//  std::vector<double> b;
//  a = butter::ComputeDenCoeffs(FiltOrd, low_pass_f, high_pass_f);
//  b = butter::ComputeNumCoeffs(FiltOrd, low_pass_f, high_pass_f, a);
//  y = butter::filter(data,b,a);
//  return y;
//}
//
//std::vector<STSLAMCommon::IMUData> PredictFilter::ButterFilterIMUData(const std::vector<STSLAMCommon::IMUData> &imu_datas){
//  if(imu_datas.size()<2)
//    return imu_datas;
//  std::vector<STSLAMCommon::IMUData> imu_datas_lowpass;
//  Eigen::Vector3f acc_raw;
//  Eigen::Vector3f gyr_raw;
//  Eigen::Vector3f acc;
//  Eigen::Vector3f gyr;
//  double t;
//
//  std::vector<double> time_list;
//  std::vector<double> acc_raw_list_0;
//  std::vector<double> acc_raw_list_1;
//  std::vector<double> acc_raw_list_2;
//  std::vector<double> gyro_raw_list_0;
//  std::vector<double> gyro_raw_list_1;
//  std::vector<double> gyro_raw_list_2;
//  STSLAMCommon::IMUData one_raw_imu_data;
//  for (int i = 0; i < imu_datas.size(); ++i) {
//    one_raw_imu_data= imu_datas[i];
//    acc_raw=one_raw_imu_data.acc;
//    gyr_raw=one_raw_imu_data.gyr;
//    t=one_raw_imu_data.t;
//
//    time_list.push_back(t);
//    acc_raw_list_0.push_back(acc_raw[0]);
//    acc_raw_list_1.push_back(acc_raw[1]);
//    acc_raw_list_2.push_back(acc_raw[2]);
//    gyro_raw_list_0.push_back(gyr_raw[0]);
//    gyro_raw_list_1.push_back(gyr_raw[1]);
//    gyro_raw_list_2.push_back(gyr_raw[2]);
//  }
//
//  std::vector<double> acc_filter_list_0;
//  std::vector<double> acc_filter_list_1;
//  std::vector<double> acc_filter_list_2;
//  std::vector<double> gyro_filter_list_0;
//  std::vector<double> gyro_filter_list_1;
//  std::vector<double> gyro_filter_list_2;
//
//  acc_filter_list_0 = ButterFilter(acc_raw_list_0,500,0,80);
//  acc_filter_list_1 = ButterFilter(acc_raw_list_1,500,0,80);
//  acc_filter_list_2 = ButterFilter(acc_raw_list_2,500,0,80);
//  gyro_filter_list_0 = ButterFilter(gyro_raw_list_0,500,0,80);
//  gyro_filter_list_1 = ButterFilter(gyro_raw_list_1,500,0,80);
//  gyro_filter_list_2 = ButterFilter(gyro_raw_list_2,500,0,80);
//
//  STSLAMCommon::IMUData imudata;
//  for (int i = 0; i < imu_datas.size(); ++i) {
//    imudata.acc << acc_filter_list_0[i],acc_filter_list_1[i],acc_filter_list_2[i];
//    imudata.gyr << gyro_filter_list_0[i],gyro_filter_list_1[i],gyro_filter_list_2[i];
//    imudata.t = time_list[i];
//    imu_datas_lowpass.push_back( imudata );
//  }
//
//  return  imu_datas_lowpass;
//
//}


PredictState PredictFilter::GetPredictState() { return cur_predict_state_; }

void PredictFilter::PredictInterface(const double &predict_t,
                                     const std::vector<STSLAMCommon::IMUData> &imu_datas,
                                     const VIOState &vio_state,
                                     const PredictInterfaceInputParameter &predictInterfaceInputParameter
){

  if( predictInterfaceInputParameter.save_online_data){

    int stage= predictInterfaceInputParameter.stage;
    double predict_real_time=predictInterfaceInputParameter.predict_real_time;
    std::string vio_state_file_path=predictInterfaceInputParameter.vio_state_file_path;
    std::string imu_file_path=predictInterfaceInputParameter.imu_file_path;
    std::string predict_stage_file_path=predictInterfaceInputParameter.predict_stage_file_path;

    static std::ofstream f_predict_stage(predict_stage_file_path.c_str());
    f_predict_stage << std::fixed << std::setprecision(6)
                    <<ST_Predict::CurrentTimestampS()<<" "
                    <<predict_real_time << " "
                    <<predict_t << " "
                    << stage <<" "
                    << std::endl;

    static std::ofstream f1(vio_state_file_path.c_str());

    f1 << std::fixed << std::setprecision(6)
       << predict_real_time << ","
       << vio_state.timestamp <<","
       << vio_state.t_wi[0]<<","<< vio_state.t_wi[1]<<","<< vio_state.t_wi[2]<<","
       << vio_state.r_wi.w()<<","<<vio_state.r_wi.x()<<","<< vio_state.r_wi.y()<<","<< vio_state.r_wi.z()<<","
       << vio_state.v[0]<<","<<vio_state.v[1]<<","<< vio_state.v[2]<<","
       << vio_state.bw[0]<<","<<vio_state.bw[1]<<","<< vio_state.bw[2]<<","
       << vio_state.ba[0]<<","<<vio_state.ba[1]<<","<< vio_state.ba[2]<<","
       << vio_state.imu_data.t<<","
       <<vio_state.imu_data.gyr[0]<<","<<vio_state.imu_data.gyr[1]<<","<<vio_state.imu_data.gyr[2]<<","
       <<vio_state.imu_data.acc[0]<<","<<vio_state.imu_data.acc[1]<<","<<vio_state.imu_data.acc[2]<<","
       <<vio_state.frame_id<<","
       <<vio_state.b_state_update
       << std::endl;

    static std::ofstream f2(imu_file_path.c_str());
    for(int i =0;i<imu_datas.size();i++){
      f2 << std::fixed << std::setprecision(6)
         << predict_real_time << ","
         << stage << ","
         << imu_datas[i].t <<","
         << imu_datas[i].gyr[0]<<","<< imu_datas[i].gyr[1]<<","<< imu_datas[i].gyr[2]<<","
         << imu_datas[i].acc[0]<<","<< imu_datas[i].acc[1]<<","<< imu_datas[i].acc[2]
         << std::endl;
    }
  }

  //VIOState process_vio_state =GetInterpolatedImuData(imu_datas,vio_state);
  UpdateVIOState(vio_state);
  Predict(predict_t,imu_datas,predictInterfaceInputParameter.stage);
}

PredictState PredictFilter::GetSmoothPredictState(const double &predict_t) {
  PredictState res;
  for (auto iter = states_buf_.end(); iter > states_buf_.begin(); iter--) {
    if (std::fabs(iter->timestamp - predict_t) < 0.002)
      res = *iter;
  }

  if (FLT_MIN == res.timestamp) {
    LOG(ERROR) << "res.timestamp: " << res.timestamp;

    double min_diff = DBL_MAX;
    for (auto iter = states_buf_.begin(); iter < states_buf_.end(); iter++) {
      if (std::fabs(iter->timestamp - predict_t) < min_diff) {
        min_diff = std::fabs(iter->timestamp - predict_t);
        res = *iter;
      }
    }
    LOG(ERROR) << "unknown predict time: " << predict_t
               << ", use most near state" <<res.timestamp;
  }

//  res.t_wb = res.t_wb + res.r_wb * t_ic_;
//  res.r_wb =res.r_wb * r_ic_;

  res.vel *= smooth_scale_t_;
  res.angular_vel *= smooth_scale_t_;
  res.acc *= smooth_scale_t_ * smooth_scale_t_;
  res.angular_acc *= smooth_scale_t_ * smooth_scale_t_;
  res.acc_jerk *= smooth_scale_t_ * smooth_scale_t_ * smooth_scale_t_;
  res.angular_jerk *= smooth_scale_t_ * smooth_scale_t_ * smooth_scale_t_;

  return res;
}

double PredictFilter::GetPredictTimeIntervel(){

  return predict_time_intervel_;
}

PredictState PredictFilter::GetFirstFilterState()  {
  PredictState res;
  res=first_filter_predict_state_;
//  res.t_wb = latest_p_;
//  res.r_wb = latest_q_;
//  res.t_wb = res.t_wb + res.r_wb * t_ic_;
//  res.r_wb =res.r_wb * r_ic_;
  //res.vel = latest_v_;
  //res.timestamp = latest_t_;
  res.vel *= smooth_scale_t_;
  res.angular_vel *= smooth_scale_t_;
  res.acc *= smooth_scale_t_ * smooth_scale_t_;
  res.angular_acc *= smooth_scale_t_ * smooth_scale_t_;
  res.acc_jerk *= smooth_scale_t_ * smooth_scale_t_ * smooth_scale_t_;
  res.angular_jerk *= smooth_scale_t_ * smooth_scale_t_ * smooth_scale_t_;
  return res;
}

PredictState PredictFilter::GetSmoothFilterState()  {
  PredictState res;
  res=smooth_filter_predict_state_;
//  res.t_wb = latest_p_;
//  res.r_wb = latest_q_;
//  res.t_wb = res.t_wb + res.r_wb * t_ic_;
//  res.r_wb =res.r_wb * r_ic_;
  //res.vel = latest_v_;
  //res.timestamp = latest_t_;
  res.vel *= smooth_scale_t_;
  res.angular_vel *= smooth_scale_t_;
  res.acc *= smooth_scale_t_ * smooth_scale_t_;
  res.angular_acc *= smooth_scale_t_ * smooth_scale_t_;
  res.acc_jerk *= smooth_scale_t_ * smooth_scale_t_ * smooth_scale_t_;
  res.angular_jerk *= smooth_scale_t_ * smooth_scale_t_ * smooth_scale_t_;
  return res;
}

PredictState PredictFilter::Get_IMUIntegration_PRV() {
  PredictState res;
  res.t_wb = latest_p_;
  res.r_wb = latest_q_;
  res.vel = latest_v_;
//  res.t_wb = res.t_wb + res.r_wb * t_ic_;
//  res.r_wb =res.r_wb * r_ic_;
  //res.vel = latest_v_;
  res.timestamp = latest_imu_t_;
  res.angular_vel=latest_angular_vel_;
  //res.vel *= smooth_scale_t_;
//  res.angular_vel *= smooth_scale_t_;
//  res.acc *= smooth_scale_t_ * smooth_scale_t_;
//  res.angular_acc *= smooth_scale_t_ * smooth_scale_t_;
//  res.acc_jerk *= smooth_scale_t_ * smooth_scale_t_ * smooth_scale_t_;
//  res.angular_jerk *= smooth_scale_t_ * smooth_scale_t_ * smooth_scale_t_;
  return res;
}

// time:     --------- T-1 -------- T ---------- T+1 ------------
// wrap   ----T-1/T----|----T/T+1---|-----
// render ---- T ------|---- T+1 ---|------

// first kalman filter is update by every imu measurement arrive
// position is update by second kalman filter every 30ms, rotation is update by
// second kalman filter every 18ms this is due to render and wrap thread
// property, every predict pose is equal to 30ms position + 18 ms rotation
//　在正常的情况下,　每个预测时间应该调用三次
// 第一次调用的时候更新position.
// 第二次，第三次调用的时候更新rotation
void PredictFilter::Predict(
    const double &predict_t,
    const std::vector<STSLAMCommon::IMUData> &imu_datas_raw, const int stage) {
  double cur_predict_t;
  std::lock_guard<std::mutex> lck(mtx_);
  if (!initial_) {
    LOG(WARNING) << "predict filter not initial. " << predict_t;
    return;
  }

  std::vector<STSLAMCommon::IMUData> imu_datas = LowerPassFilter( imu_datas_raw );
  //std::vector<STSLAMCommon::IMUData> imu_datas =ButterFilterIMUData(imu_datas_raw);

  //std::vector<STSLAMCommon::IMUData> imu_datas =  imu_datas_raw;

  // predict_time_intervel_=predict_t-imu_datas[imu_datas.size()-1].t;
  if (imu_datas.size() <= 0) {
    LOG(WARNING) << "PredictFilter::Predict: imu_datas.size = 0";
  }
  predict_time_intervel_ = imu_datas.size() > 0 ? predict_t - imu_datas[imu_datas.size() - 1].t : predict_t - cur_smooth_state_.timestamp;

//  int predict_num_ =0;
//  double last_predict_t_=0;
  cur_predict_t=predict_t;
//  if(predict_time_intervel_>predict_threshold){
//    cur_predict_t = imu_datas[imu_datas.size()-1].t+predict_threshold;
//  }

  if(first_predict_t_)
    last_predict_t_=cur_predict_t;

//  //　预测的时间应该保持递增
//  if (!first_predict_t_ &&  predict_num_==0) {
////    LOG(WARNING) << "Predict seq out of order: " << predict_t
////                 << ", cur smooth: " << cur_smooth_state_.timestamp;
//    //return;
//    last_predict_t_=predict_t;
//    cur_predict=last_predict_t_;
//  } else{
//    cur_predict=last_predict_t_;
//  }
//
//  LOG(WARNING) << "Predict seq out of order: " << predict_t << "predict_num_: " << predict_num_
//               << ", cur_predict: " << cur_predict;
//
//  predict_num_=predict_num_+1;
//
//  if(predict_num_==3)
//    predict_num_=0;
//
//  if (!first_predict_t_ && cur_smooth_state_.timestamp - cur_predict > 0.001){
//    return;
//  }


//　预测的时间应该保持递增
  if (!first_predict_t_ && cur_smooth_state_.timestamp - predict_t > 0.002) {
    LOG(WARNING) << "Predict seq out of order: " << predict_t
                 << ", cur smooth: " << cur_smooth_state_.timestamp;
    return;
  }

  bool update_r = false;
  bool update_t = false;
  // 新到的一个预测时间．根据stage更新rt
  if ( cur_predict_t - cur_smooth_state_.timestamp > 0.002 ) {
    if(stage==1){
      update_t = true;
    } else{
      update_r = true;
    }
  } else {  // 第二次，第三次更新rotation
    if(stage==1){
      update_t = true;
    } else{
      update_r = true;
    }
  }

  if (cur_predict_t - cur_smooth_state_.timestamp > 0.002) {
    Eigen::MatrixXd A_t, G_t;
    Eigen::MatrixXd A_r, G_r;
    last_smooth_t_cov_ = cur_smooth_t_cov_;
    last_smooth_r_cov_ = cur_smooth_r_cov_;
    last_smooth_state_ = cur_smooth_state_;
    PredictModel(cur_predict_t, last_smooth_state_, cur_smooth_state_, A_t, G_t, A_r,
                 G_r, smooth_scale_t_, true);
    cur_smooth_t_cov_ = A_t * last_smooth_t_cov_ * A_t.transpose() +
        G_t * smooth_model_t_noise_cov_ * G_t.transpose();
    cur_smooth_r_cov_ = A_r * last_smooth_r_cov_ * A_r.transpose() +
        G_r * smooth_model_r_noise_cov_ * G_r.transpose();

    smooth_filter_predict_state_=cur_smooth_state_;
  }

  if (!imu_datas.empty()) CHECK_LT(imu_datas.back().t, cur_predict_t);

  int index = 0;
  if (!vio_state_change_) {
    while (index < imu_datas.size() && imu_datas[index].t <= latest_imu_t_)
      index++;
  } else {
    vio_state_change_ = false;
    // 最近的VIO state改变了．从最新的vio state开始重新积分
    for (; index < imu_datas.size() && imu_datas[index].t <= latest_imu_t_;
           index++)
      IMUPreIntegration(imu_datas[index]);
  }

  VLOG(5) << "predict t: " << predict_t
          << ", cur smooth state t: " << cur_smooth_state_.timestamp
          << ", update_t: " << update_t << ", update_r: " << update_r
          << ", imu size: " << imu_datas.size() - index;

  for (; index < imu_datas.size(); index++) {
    IMUPreIntegration(imu_datas[index]);
    latest_imu_t_ = imu_datas[index].t;

    R_t_meas_.setZero();
    R_t_meas_.block<3, 3>(0, 0) = R_meas_p_;
    R_t_meas_.block<3, 3>(3, 3) = R_meas_v_ / pow(scale_t_, 2);

    R_r_meas_.setZero();
    R_r_meas_.block<3, 3>(0, 0) = R_meas_q_;
    R_r_meas_.block<3, 3>(3, 3) =
        FLAGS_vio_meas_w * Eigen::Matrix3d::Identity();

    Eigen::MatrixXd A_t, G_t;
    Eigen::MatrixXd A_r, G_r;
    PredictModel(imu_datas[index].t, last_predict_state_, cur_predict_state_,
                 A_t, G_t, A_r, G_r, scale_t_, true);

    cur_state_t_cov_ = A_t * last_state_t_cov_ * A_t.transpose() +
        G_t * model_t_noise_cov_ * G_t.transpose();
    cur_state_r_cov_ = A_r * last_state_r_cov_ * A_r.transpose() +
        G_r * model_r_noise_cov_ * G_r.transpose();

    // first kalman update
    UpdateCoarseState();
    last_predict_state_ = cur_predict_state_;
    last_state_r_cov_ = cur_state_r_cov_;
    last_state_t_cov_ = cur_state_t_cov_;
  }

//  if( latest_imu_t_ > cur_smooth_state_.timestamp ){
//    UpdateSmoothState(true, true);
//  }


  if (first_predict_t_) {
    LOG(WARNING) << "first predict t: " << predict_t;
    first_predict_t_ = false;
    UpdatePosition(cur_predict_t);
    update_t = false;
  }

  if (update_t && cur_smooth_state_.update_rotation_cnt < 2) {
    if (latest_imu_t_ < cur_smooth_state_.timestamp) update_r = true;
    VLOG(5) << "cur smooth state t: " << cur_smooth_state_.timestamp
            << ", update_rotation: " << cur_smooth_state_.update_rotation_cnt
            << "," << latest_imu_t_ << "," << predict_t;
  }

  if (update_r) {
    VLOG(5) << "update_state: r " << predict_t << ","
            << cur_smooth_state_.timestamp
            << ", imu state t: " << last_predict_state_.timestamp;
    UpdateRotation(cur_smooth_state_.timestamp);
  }

  if (update_t) {
    VLOG(5) << "update_state: t " << predict_t
            << ", imu state t: " << last_predict_state_.timestamp;
    UpdatePosition(cur_predict_t);
  }

}

//only predict, do not update state
void PredictFilter::PredictRT(double predict_t) {

  Eigen::MatrixXd A_t, G_t;
  Eigen::MatrixXd A_r, G_r;

  //predict but not change state
  PredictState cur_predict_state;
  Eigen::Matrix<double, 12, 12> cur_state_r_cov, cur_state_t_cov;
  PredictModel(predict_t, last_predict_state_, cur_predict_state,
               A_t, G_t, A_r, G_r, scale_t_, true);

  cur_state_t_cov = A_t * last_state_t_cov_ * A_t.transpose() +
      G_t * model_t_noise_cov_ * G_t.transpose();
  cur_state_r_cov = A_r * last_state_r_cov_ * A_r.transpose() +
      G_r * model_r_noise_cov_ * G_r.transpose();

  //cur_smooth_state_.update_rotation_cnt = 0;
  PredictState cur_smooth_state_tmp =cur_smooth_state_;
  Eigen::Matrix<double, 12, 12> cur_state_r_cov_tmp, cur_state_t_cov_tmp;
  cur_state_r_cov_tmp=cur_state_r_cov_;
  cur_state_t_cov_tmp=cur_state_t_cov_;
  UpdateSmoothWithoutChangeState(true, true,
                                 cur_predict_state,
                                 cur_state_t_cov,
                                 cur_state_r_cov,
                                 cur_state_t_cov_tmp,
                                 cur_state_r_cov_tmp,
                                 cur_smooth_state_tmp);

  cur_smooth_state_tmp.result_is_valid=0;
  states_buf_.push_back(cur_smooth_state_tmp);

}

void PredictFilter::UpdateRT(double predict_t){


  last_smooth_t_cov_ = cur_smooth_t_cov_;
  last_smooth_r_cov_ = cur_smooth_r_cov_;
  last_smooth_state_ = cur_smooth_state_;

  Eigen::MatrixXd A_t, G_t;
  Eigen::MatrixXd A_r, G_r;
  Eigen::Vector3d twc, angle;
  //VLOG(3)<< "UpdatePosition: " <<predict_t-last_smooth_state_.timestamp;
  PredictModel(latest_t_, last_smooth_state_, cur_smooth_state_, A_t, G_t, A_r,
               G_r, smooth_scale_t_, true);
  cur_smooth_t_cov_ = A_t * last_smooth_t_cov_ * A_t.transpose() +
      G_t * smooth_model_t_noise_cov_ * G_t.transpose();
  cur_smooth_r_cov_ = A_r * last_smooth_r_cov_ * A_r.transpose() +
      G_r * smooth_model_r_noise_cov_ * G_r.transpose();

  UpdateSmoothState(true, true);

  last_smooth_t_cov_ = cur_smooth_t_cov_;
  last_smooth_r_cov_ = cur_smooth_r_cov_;
  last_smooth_state_ = cur_smooth_state_;




//  if (states_buf_.size() > 1000) {
//    states_buf_.erase(states_buf_.begin(), states_buf_.end() - 100);
//    //CHECK_LT(states_buf_.front().timestamp, cur_smooth_state_.timestamp);
//  }
  auto iter = states_buf_.begin();
  for (; iter < states_buf_.end(); iter++) {
    if (std::fabs(iter->timestamp - cur_smooth_state_.timestamp) < 0.002) {
      //CHECK_EQ(iter->result_is_valid, 0);
      if(iter->result_is_valid== 0)
      states_buf_.erase(iter);
      break;
    }
  }
  cur_smooth_state_.result_is_valid=1;
  states_buf_.push_back(cur_smooth_state_);

  //predict but not change state
  last_predict_state_ = cur_predict_state_;
  last_state_r_cov_ = cur_state_r_cov_;
  last_state_t_cov_ = cur_state_t_cov_;

  PredictState cur_predict_state  ;
  Eigen::Matrix<double, 12, 12> cur_state_r_cov, cur_state_t_cov;
  PredictModel(predict_t, last_predict_state_, cur_predict_state,
                 A_t, G_t, A_r, G_r, scale_t_, true);

  cur_state_t_cov = A_t * last_state_t_cov_ * A_t.transpose() +
        G_t * model_t_noise_cov_ * G_t.transpose();
  cur_state_r_cov = A_r * last_state_r_cov_ * A_r.transpose() +
        G_r * model_r_noise_cov_ * G_r.transpose();

  //cur_smooth_state_.update_rotation_cnt = 0;
  PredictState cur_smooth_state_tmp =cur_smooth_state_;
  Eigen::Matrix<double, 12, 12> cur_state_r_cov_tmp, cur_state_t_cov_tmp;
  cur_state_r_cov_tmp=cur_state_r_cov_;
  cur_state_t_cov_tmp=cur_state_t_cov_;
  UpdateSmoothWithoutChangeState(true, true,
      cur_predict_state,
      cur_state_t_cov,
      cur_state_r_cov,
      cur_state_t_cov_tmp,
      cur_state_r_cov_tmp,
  cur_smooth_state_tmp);

  cur_smooth_state_tmp.result_is_valid=0;
  states_buf_.push_back(cur_smooth_state_tmp);

  // use imu measurement seem more smooth than use predict pose
//  cur_predict_state_ = last_predict_state_; // Actually is cur_predict_state_ . see line 284
//  cur_predict_state_.timestamp = predict_t; // ?????
//  cur_state_t_cov_ = last_state_t_cov_;

//  if(predict_t - latest_imu_t_ > 0.002 || predict_t-cur_predict_state_.timestamp >0.002 ){
//    PredictModel(predict_t, last_predict_state_, cur_predict_state_,
//                 A_t, G_t, A_r, G_r, scale_t_, true);
//
//    cur_state_t_cov_ = A_t * last_state_t_cov_ * A_t.transpose() +
//        G_t * model_t_noise_cov_ * G_t.transpose();
//    cur_state_r_cov_ = A_r * last_state_r_cov_ * A_r.transpose() +
//        G_r * model_r_noise_cov_ * G_r.transpose();
//
//    last_predict_state_ = cur_predict_state_;
//    last_state_r_cov_ = cur_state_r_cov_;
//    last_state_t_cov_ = cur_state_t_cov_;
//  }

//  cur_smooth_state_.update_rotation_cnt = 0;
//  cur_smooth_state_.update_position_cnt = 1;
}

void PredictFilter::UpdatePosition(double predict_t) {
//  last_smooth_t_cov_ = cur_smooth_t_cov_;
//  last_smooth_r_cov_ = cur_smooth_r_cov_;
//  last_smooth_state_ = cur_smooth_state_;

  Eigen::MatrixXd A_t, G_t;
  Eigen::MatrixXd A_r, G_r;
  Eigen::Vector3d twc, angle;
  PredictState tmp_state;

//  PredictModel(predict_t, last_smooth_state_, cur_smooth_state_, A_t, G_t, A_r,
//               G_r, smooth_scale_t_, true);
//  cur_smooth_t_cov_ = A_t * last_smooth_t_cov_ * A_t.transpose() +
//      G_t * smooth_model_t_noise_cov_ * G_t.transpose();
//  cur_smooth_r_cov_ = A_r * last_smooth_r_cov_ * A_r.transpose() +
//      G_r * smooth_model_r_noise_cov_ * G_r.transpose();

//  // use imu measurement seem more smooth than use predict pose
//  cur_predict_state_ = last_predict_state_;
//  cur_predict_state_.timestamp = predict_t;
//  cur_state_t_cov_ = last_state_t_cov_;

//  last_predict_state_ = cur_predict_state_;
//  last_state_t_cov_ = cur_state_t_cov_;
//  last_state_r_cov_ = cur_state_r_cov_;
// PredictModelOptional(predict_t, last_predict_state_, cur_predict_state_,
//               A_t, G_t, A_r, G_r, scale_t_, true,true, true);
  PredictModel(predict_t, last_predict_state_, cur_predict_state_,
                       A_t, G_t, A_r, G_r, scale_t_, true);
  cur_state_t_cov_ = A_t * last_state_t_cov_ * A_t.transpose() +
      G_t * model_t_noise_cov_ * G_t.transpose();
  cur_state_r_cov_ = A_r * last_state_r_cov_ * A_r.transpose() +
      G_r * model_r_noise_cov_ * G_r.transpose();

  if (predict_t < last_predict_state_.timestamp - 0.000001) {
    LOG(WARNING) << "PredictFilter::UpdatePosition: predict_t < last_predict_state_.timestamp";
  }
//  last_predict_state_ = cur_predict_state_;
//  last_state_t_cov_ = cur_state_t_cov_;
//  last_state_r_cov_ = cur_state_r_cov_;
  first_filter_predict_state_=cur_predict_state_;

  cur_smooth_state_.update_rotation_cnt = 1;
  cur_smooth_state_.update_position_cnt = 1;

  UpdateSmoothState(true, true);
}

void PredictFilter::UpdateRotation(double predict_t) {
  Eigen::MatrixXd A_t, G_t;
  Eigen::MatrixXd A_r, G_r;
  Eigen::Vector3d twc, angle;
  PredictState tmp_state;

  PredictModel(predict_t, last_predict_state_, cur_predict_state_, A_t, G_t,
               A_r, G_r, scale_t_, true);

  //first_filter_predict_state_=cur_predict_state_;

  //PredictModelOptional(predict_t, last_predict_state_, cur_predict_state_,
  //                     A_t, G_t, A_r, G_r, scale_t_, true,false, true);
  cur_state_r_cov_ = A_r * last_state_r_cov_ * A_r.transpose() +
      G_r * model_r_noise_cov_ * G_r.transpose();

  UpdateSmoothState(true, true);
  cur_smooth_state_.update_rotation_cnt++;
}

void PredictFilter::IMUPreIntegration(const STSLAMCommon::IMUData &imu_data) {

  Eigen::Vector3d un_acc_0 = latest_q_ * (acc_0_ - linearized_ba_) - gravity_;
  Eigen::Vector3d un_gyr =
      0.5 * (gyr_0_ + imu_data.gyr.cast<double>()) - linearized_bg_;
  double dt = imu_data.t - latest_t_;
  latest_q_ = latest_q_ * STSLAMCommon::deltaQ(un_gyr * dt).normalized();
  Eigen::Vector3d un_acc_1 =
      latest_q_ * (imu_data.acc.cast<double>() - linearized_ba_) - gravity_;
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  latest_p_ = latest_p_ + latest_v_ * dt + 0.5 * un_acc * dt * dt;  // q_wb

  latest_v_ = latest_v_ + un_acc * dt;                              // v_wb
  latest_angular_vel_ = imu_data.gyr.cast<double>() - linearized_bg_;
  imu_pre_integration_.PreIntegrate(dt, imu_data.acc.cast<double>(),
                                    imu_data.gyr.cast<double>());

  Eigen::Matrix3d vio_p_cov = FLAGS_vio_meas_p * Eigen::Matrix3d::Identity();
  Eigen::Matrix3d vio_v_cov = FLAGS_vio_meas_v * Eigen::Matrix3d::Identity();
  Eigen::Matrix3d vio_q_cov = FLAGS_vio_meas_q * Eigen::Matrix3d::Identity();

  double sum_t = imu_pre_integration_.sum_dt;

  // IMU preIntergration get \Delta p, \Delta v, \Delta q, we need is p, v, q
  // v_{k+1}^{w} = v_{k}^{w} + R_{bk}^{w} * \Delta V - g^{w} * \Delta t
  // \Sigma V_{k+1} = \Sigma V_{k} + {R_{bk}^{w}}^T * \Sigma \Delta V *
  // {R_{bk}^{w}}
  // + {LeftJacobian(R_{bk}^{w} * \Delta V)}^T * \Sigma {R_{bk}^{w}} *
  // {LeftJacobian(R_{bk}^{w} * \Delta V)}^T
  Eigen::Matrix3d j;
  j = LieAlgebra::inv_left_jacobian(
      Eigen::Vector3d(latest_vio_q_ * imu_pre_integration_.delta_p));
  R_meas_p_ =
      vio_p_cov + vio_v_cov * sum_t * sum_t + j.transpose() * vio_q_cov * j +
      latest_vio_q_.transpose() *
          imu_pre_integration_.covariance.block<3, 3>(0, 0) * latest_vio_q_;

  j = LieAlgebra::inv_left_jacobian(
      Eigen::Vector3d(latest_vio_q_ * imu_pre_integration_.delta_v));
  R_meas_v_ = vio_v_cov +
              latest_vio_q_.transpose() *
                  imu_pre_integration_.covariance.block<3, 3>(6, 6) *
                  latest_vio_q_ +
              j.transpose() * vio_q_cov * j;

  j = LieAlgebra::inv_left_jacobian(Eigen::Matrix3d(
      latest_vio_q_ * imu_pre_integration_.delta_q.toRotationMatrix()));
  Eigen::Matrix3d j2 = j * latest_vio_q_;
  R_meas_q_ =
      j.transpose() * vio_q_cov * j +
      j2.transpose() * imu_pre_integration_.covariance.block<3, 3>(3, 3) * j2;

  latest_t_ = imu_data.t;
  acc_0_ = imu_data.acc.cast<double>();
  gyr_0_ = imu_data.gyr.cast<double>();
}

void PredictFilter::UpdateCoarseState() {
  PredictState predict_state = cur_predict_state_;

  // position
  Eigen::Matrix<double, 6, 12> H_pose = Eigen::Matrix<double, 6, 12>::Zero();
  H_pose.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  H_pose.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  H_pose = -H_pose;

  //    Eigen::MatrixXd S_pose = H_pose * cur_state_t_cov_ * H_pose.transpose()
  //    + R_t_meas_;
  Eigen::Matrix<double, 6, 6> S_pose =
      cur_state_t_cov_.block<6, 6>(0, 0) + R_t_meas_;

  //    Eigen::MatrixXd K_pose_transpose = S_pose.ldlt().solve(H_pose *
  //    cur_state_t_cov_); Eigen::MatrixXd K_pose =
  //    K_pose_transpose.transpose();
  Eigen::Matrix<double, 12, 6> K_pose =
      (S_pose.ldlt().solve(-cur_state_t_cov_.block<6, 12>(0, 0))).transpose();

  // unit is different, all time related variable need take care
  Eigen::Vector3d delta_p = cur_predict_state_.t_wb - latest_p_;
  Eigen::Vector3d delta_v = cur_predict_state_.vel - latest_v_ / scale_t_;

  Eigen::Matrix<double, 6, 1> residual_pose;
  residual_pose << delta_p, delta_v;

  Eigen::Matrix<double, 12, 1> innovation_pose = K_pose * residual_pose;

  cur_predict_state_.t_wb += innovation_pose.head(3);
  cur_predict_state_.vel += innovation_pose.segment(3, 3);
  cur_predict_state_.acc += innovation_pose.segment(6, 3);
  cur_predict_state_.acc_jerk += innovation_pose.tail(3);

  //    cur_state_t_cov_ = (Eigen::Matrix<double, 12, 12>::Identity() - K_pose *
  //    H_pose) * cur_state_t_cov_; Eigen::Matrix<double, 12, 12> I =
  //    Eigen::Matrix<double, 12, 12>::Identity(); Eigen::MatrixXd _mat = I -
  //    K_pose * H_pose;
  Eigen::Matrix<double, 12, 12> _mat =
      Eigen::Matrix<double, 12, 12>::Identity();
  _mat.block<12, 6>(0, 0) += K_pose;

  Eigen::Matrix<double, 12, 12> cur_state_t_cov_before_update = cur_state_t_cov_;
  cur_state_t_cov_ = _mat * cur_state_t_cov_ * _mat.transpose() +
                     K_pose * R_t_meas_ * K_pose.transpose();
  double cur_state_t_cov_gain = cur_state_t_cov_before_update.determinant() - cur_state_t_cov_.determinant();
  if (cur_state_t_cov_gain < -0.000001) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);
    std::cout << "PredictFilter::UpdateCoarseState: cur_state_t_cov_gain is less than zero: " << cur_state_t_cov_gain << std::endl;
  }
  //    cur_state_t_cov_ = cur_state_t_cov_ - K_pose * S_pose *
  //    K_pose.transpose(); VLOG(3) << "update cov(t):\n" << cur_state_t_cov_;

  // rotation
  Eigen::Matrix<double, 6, 12> H_rotation =
      Eigen::Matrix<double, 6, 12>::Zero();
  Eigen::Matrix3d delta_r = cur_predict_state_.r_wb * latest_q_.transpose();
  H_rotation.block<3, 3>(0, 0) = LieAlgebra::inv_left_jacobian(delta_r);
  H_rotation.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  H_rotation = -H_rotation;

  Eigen::MatrixXd S_rotation = H_rotation * cur_state_r_cov_ * H_rotation.transpose() + R_r_meas_;
  // Eigen::Matrix<double, 6, 12> _mat1 = -cur_state_r_cov_.block<6, 12>(0, 0);
  // _mat1.block<3, 12>(0, 0) =
  //     -H_rotation.block<3, 3>(0, 0) * _mat1.block<3, 12>(0, 0);
  // Eigen::Matrix<double, 6, 6> _mat2 = _mat1.block<6, 6>(0, 0);
  // _mat2.block<6, 3>(0, 0) =
  //     _mat1.block<6, 3>(0, 0) * H_rotation.block<3, 3>(0, 0);
  // _mat2.block<6, 3>(0, 3) *= -1;

  // Eigen::Matrix<double, 6, 6> S_rotation = _mat2 + R_r_meas_;

  //    Eigen::MatrixXd K_rotation = cur_state_r_cov_ * H_rotation.transpose() *
  //    S_rotation.inverse();
  //    Eigen::MatrixXd K_rotation_transpose =
  //    S_rotation.ldlt().solve(H_rotation * cur_state_r_cov_); Eigen::MatrixXd
  //    K_rotation = K_rotation_transpose.transpose();
  // Eigen::Matrix<double, 12, 6> K_rotation =
  //     (S_rotation.ldlt().solve(_mat1)).transpose();
  // Eigen::Matrix<double, 12, 6> K_rotation = (S_rotation.inverse() * _mat1).transpose();
  Eigen::Matrix<double, 12, 6> K_rotation = cur_state_r_cov_ * H_rotation.transpose() * S_rotation.inverse();

  Eigen::Vector3d delta_q = LieAlgebra::log(delta_r);
  Eigen::Vector3d delta_angular =
      cur_predict_state_.angular_vel - latest_angular_vel_ / scale_t_;

  Eigen::Matrix<double, 6, 1> residual_rotation;
  residual_rotation << delta_q, delta_angular;

  Eigen::Matrix<double, 12, 1> innovation_rotation =
      K_rotation * residual_rotation;

  //std::cout<<"UpdateCoarseState: " << innovation_rotation.head(3) << std::endl;
  cur_predict_state_.r_wb =
      LieAlgebra::exp(innovation_rotation.head(3)) * cur_predict_state_.r_wb;
  //std::cout<<"success " << std::endl;

  cur_predict_state_.angular_vel += innovation_rotation.segment(3, 3);
  cur_predict_state_.angular_acc += innovation_rotation.segment(6, 3);
  cur_predict_state_.angular_jerk += innovation_rotation.tail(3);

  //    cur_state_r_cov_ = (Eigen::Matrix<double, 12, 12>::Identity() -
  //    K_rotation * H_rotation) * cur_state_r_cov_; _mat = I - K_rotation *
  //    H_rotation; cur_state_r_cov_ = _mat * cur_state_r_cov_ *
  //    _mat.transpose() + K_rotation * R_r_meas_ * K_rotation.transpose();

  Eigen::MatrixXd _mat4 = Eigen::Matrix<double, 12, 12>::Identity();
  Eigen::Matrix<double, 12, 6> _mat5 = K_rotation;
  _mat5.block<12, 3>(0, 0) =
      K_rotation.block<12, 3>(0, 0) * H_rotation.block<3, 3>(0, 0);
  _mat5.block<12, 3>(0, 3) *= -1;
  _mat4.block<12, 6>(0, 0) -= _mat5;

  Eigen::Matrix<double, 12, 12> cur_state_r_cov_before_update = cur_state_r_cov_;
  cur_state_r_cov_ = _mat4 * cur_state_r_cov_ * _mat4.transpose() +
                     K_rotation * R_r_meas_ * K_rotation.transpose();
  double cur_state_r_cov_gain = cur_state_r_cov_before_update.determinant() - cur_state_r_cov_.determinant();
  if (cur_state_r_cov_gain < -0.000001) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);
    std::cout << "PredictFilter::UpdateCoarseState: cur_state_r_cov_gain is less than zero: " << cur_state_r_cov_gain << std::endl;
  }
}


void PredictFilter::UpdateSmoothWithoutChangeState(bool update_t, bool update_r,
                                                   const PredictState &cur_predict_state,
                                                   const Eigen::Matrix<double, 12, 12> &cur_state_t_cov,
                                                   const Eigen::Matrix<double, 12, 12> &cur_state_r_cov,
                                                   Eigen::Matrix<double, 12, 12> &cur_smooth_t_cov,
                                                   Eigen::Matrix<double, 12, 12> &cur_smooth_r_cov,
                                                   PredictState &cur_smooth_state){
  Eigen::Matrix<double, 12, 12> R_t = cur_state_t_cov;
  Eigen::Matrix<double, 12, 12> R_r = cur_state_r_cov;

//  R_t.block<3,3>(0,0) +=  Eigen::Matrix3d::Identity();
//  R_t.block<3,3>(3,3) +=  Eigen::Matrix3d::Identity();


  // convert unit
  double scale = scale_t_ / smooth_scale_t_;
  double scale2 = scale * scale;
  double scale3 = scale2 * scale;

  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      double s = 1;
      for (int k = 0; k < r + c; k++) s *= scale;
      R_t.block<3, 3>(3 * r, 3 * c) *= s;
      R_r.block<3, 3>(3 * r, 3 * c) *= s;
    }
  }

  PredictState meas_state = cur_predict_state;
  meas_state.vel *= scale;
  meas_state.angular_vel *= scale;
  meas_state.acc *= scale2;
  meas_state.angular_acc *= scale2;
  meas_state.acc_jerk *= scale3;
  meas_state.angular_jerk *= scale3;

  cur_smooth_state.timestamp = cur_predict_state.timestamp;

  Eigen::Vector3d delta_p = cur_smooth_state.t_wb - meas_state.t_wb;
  Eigen::Matrix3d delta_r =
      cur_smooth_state.r_wb * meas_state.r_wb.transpose();
  Eigen::Vector3d delta_q = LieAlgebra::log(delta_r);
  Eigen::Vector3d delta_v = cur_smooth_state.vel - meas_state.vel;
  Eigen::Vector3d delta_angular =
      cur_smooth_state.angular_vel - meas_state.angular_vel;
  Eigen::Vector3d delta_acc = cur_smooth_state.acc - meas_state.acc;
  Eigen::Vector3d delta_angular_acc =
      cur_smooth_state.angular_acc - meas_state.angular_acc;
  Eigen::Vector3d delta_acc_jerk =
      cur_smooth_state.acc_jerk - meas_state.acc_jerk;
  Eigen::Vector3d delta_angular_jerk =
      cur_smooth_state.angular_jerk - meas_state.angular_jerk;

  Eigen::Vector3d meas_twc, meas_angle;

  if (update_t) {
    //        Eigen::Matrix<double, 12, 12> H_t = Eigen::Matrix<double, 12,
    //        12>::Identity();
    Eigen::Matrix<double, 6, 12> H_t = Eigen::Matrix<double, 6, 12>::Zero();
    H_t.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    H_t.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    H_t = -H_t;

    //        Eigen::Matrix<double, 6, 6> S_t = H_t * cur_smooth_t_cov_ *
    //        H_t.transpose() + R_t.block(0, 0, 6, 6);
    Eigen::Matrix<double, 6, 6> S_t =
        cur_smooth_t_cov.block<6, 6>(0, 0) + R_t.block(0, 0, 6, 6);
    //        Eigen::MatrixXd K_t = cur_smooth_t_cov_ * H_t.transpose() *
    //        S_t.inverse();

    //        Eigen::Matrix<double, 12, 6> K_t = (S_t.ldlt().solve(H_t *
    //        cur_smooth_t_cov_)).transpose();
    Eigen::Matrix<double, 12, 6> K_t =
        (S_t.ldlt().solve(-cur_smooth_t_cov.block<6, 12>(0, 0))).transpose();

    Eigen::Matrix<double, 6, 1> residual_t;
    //        residual_t << delta_p, delta_v, delta_acc, delta_acc_jerk;
    residual_t << delta_p, delta_v;

    Eigen::Matrix<double, 12, 1> innovation_t = K_t * residual_t;

    cur_smooth_state.t_wb += innovation_t.head(3);
    cur_smooth_state.vel += innovation_t.segment(3, 3);
    cur_smooth_state.acc += innovation_t.segment(6, 3);
    cur_smooth_state.acc_jerk += innovation_t.tail(3);

    //        Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12,
    //        12>::Identity(); Eigen::Matrix<double, 12, 12> _mat = I - K_t *
    //        H_t;
    Eigen::Matrix<double, 12, 12> _mat =
        Eigen::Matrix<double, 12, 12>::Identity();
    _mat.block<12, 6>(0, 0) += K_t.block<12, 6>(0, 0);

    cur_smooth_t_cov= _mat * cur_smooth_t_cov* _mat.transpose() +
        K_t * R_t.block(0, 0, 6, 6) * K_t.transpose();
  }

  //     rotation
  if (update_r) {
    Eigen::Matrix<double, 12, 12> H_r =
        Eigen::Matrix<double, 12, 12>::Identity();
    H_r.block<3, 3>(0, 0) = LieAlgebra::inv_left_jacobian(delta_r);
    H_r = -H_r;

    Eigen::Matrix<double, 12, 12> _mat = cur_smooth_r_cov;
    _mat.block<3, 12>(0, 0) = H_r.block<3, 3>(0, 0) * _mat.block<3, 12>(0, 0);
    _mat.block<9, 12>(3, 0) *= -1;
    Eigen::Matrix<double, 12, 12> _mat2 = _mat;
    _mat2.block<12, 3>(0, 0) =
        _mat2.block<12, 3>(0, 0) * H_r.block<3, 3>(0, 0).transpose();
    _mat2.block<12, 9>(0, 3) *= -1;

    //        Eigen::Matrix<double, 12, 12> S_r = H_r * cur_smooth_r_cov_ *
    //        H_r.transpose() + R_r; Eigen::Matrix<double, 12, 12> K_r =
    //        (S_r.ldlt().solve(H_r * cur_smooth_r_cov_)).transpose();

    Eigen::Matrix<double, 12, 12> S_r = _mat2 + R_r;
    Eigen::Matrix<double, 12, 12> K_r = (S_r.ldlt().solve(_mat)).transpose();

    Eigen::Matrix<double, 12, 1> residual_r;
    residual_r << delta_q, delta_angular, delta_angular_acc, delta_angular_jerk;
    //        residual_r << delta_q, delta_angular;

    //        double pre_cost = residual_r.transpose() *
    //        cur_smooth_r_cov_.inverse() * residual_r; LOG(ERROR) << "det2: "
    //        << S_r.determinant() << " " << R_r.determinant();
    Eigen::Matrix<double, 12, 1> innovation_r = K_r * residual_r;

    //std::cout<<"UpdateSmoothState: " << innovation_r.head(3) << std::endl;
    cur_smooth_state.r_wb =
        LieAlgebra::exp(innovation_r.head(3)) * cur_smooth_state.r_wb;
    //std::cout<<"success " << std::endl;

    cur_smooth_state.angular_vel += innovation_r.segment(3, 3);
    cur_smooth_state.angular_acc += innovation_r.segment(6, 3);
    cur_smooth_state.angular_jerk += innovation_r.tail(3);

    //        Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12,
    //        12>::Identity(); _mat = I - K_r * H_r;
    _mat = K_r;
    _mat.block<12, 3>(0, 0) = K_r.block<12, 3>(0, 0) * H_r.block<3, 3>(0, 0);
    _mat.block<12, 3>(0, 0) *= -1;
    _mat += Eigen::Matrix<double, 12, 12>::Identity();

    cur_smooth_r_cov = _mat * cur_smooth_r_cov * _mat.transpose() +
        K_r * R_r * K_r.transpose();
  }

}

void PredictFilter::UpdateSmoothState(bool update_t, bool update_r) {
  Eigen::Matrix<double, 12, 12> R_t = cur_state_t_cov_;
  Eigen::Matrix<double, 12, 12> R_r = cur_state_r_cov_;

  // convert unit
  double scale = scale_t_ / smooth_scale_t_;
  double scale2 = scale * scale;
  double scale3 = scale2 * scale;

  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      double s = 1;
      for (int k = 0; k < r + c; k++) s *= scale;
      R_t.block<3, 3>(3 * r, 3 * c) *= s;
      R_r.block<3, 3>(3 * r, 3 * c) *= s;
    }
  }

  PredictState meas_state = cur_predict_state_;
  meas_state.vel *= scale;
  meas_state.angular_vel *= scale;
  meas_state.acc *= scale2;
  meas_state.angular_acc *= scale2;
  meas_state.acc_jerk *= scale3;
  meas_state.angular_jerk *= scale3;

  cur_smooth_state_.timestamp = cur_predict_state_.timestamp;

  Eigen::Vector3d delta_p = cur_smooth_state_.t_wb - meas_state.t_wb;
  Eigen::Matrix3d delta_r =
      cur_smooth_state_.r_wb * meas_state.r_wb.transpose();
  Eigen::Vector3d delta_q = LieAlgebra::log(delta_r);
  Eigen::Vector3d delta_v = cur_smooth_state_.vel - meas_state.vel;
  Eigen::Vector3d delta_angular =
      cur_smooth_state_.angular_vel - meas_state.angular_vel;
  Eigen::Vector3d delta_acc = cur_smooth_state_.acc - meas_state.acc;
  Eigen::Vector3d delta_angular_acc =
      cur_smooth_state_.angular_acc - meas_state.angular_acc;
  Eigen::Vector3d delta_acc_jerk =
      cur_smooth_state_.acc_jerk - meas_state.acc_jerk;
  Eigen::Vector3d delta_angular_jerk =
      cur_smooth_state_.angular_jerk - meas_state.angular_jerk;

  Eigen::Vector3d meas_twc, meas_angle;

  if (update_t) {
    PredictState _state = cur_smooth_state_;
    PredictState tmp_state = cur_smooth_state_;

    //        Eigen::Matrix<double, 12, 12> H_t = Eigen::Matrix<double, 12,
    //        12>::Identity();
    Eigen::Matrix<double, 6, 12> H_t = Eigen::Matrix<double, 6, 12>::Zero();
    H_t.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    H_t.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    H_t = -H_t;

    //        Eigen::Matrix<double, 6, 6> S_t = H_t * cur_smooth_t_cov_ *
    //        H_t.transpose() + R_t.block(0, 0, 6, 6);
    Eigen::Matrix<double, 6, 6> S_t =
        cur_smooth_t_cov_.block<6, 6>(0, 0) + R_t.block(0, 0, 6, 6);
    //        Eigen::MatrixXd K_t = cur_smooth_t_cov_ * H_t.transpose() *
    //        S_t.inverse();

    //        Eigen::Matrix<double, 12, 6> K_t = (S_t.ldlt().solve(H_t *
    //        cur_smooth_t_cov_)).transpose();
    Eigen::Matrix<double, 12, 6> K_t =
        (S_t.ldlt().solve(-cur_smooth_t_cov_.block<6, 12>(0, 0))).transpose();

    Eigen::Matrix<double, 6, 1> residual_t;
    //        residual_t << delta_p, delta_v, delta_acc, delta_acc_jerk;
    residual_t << delta_p, delta_v;

    Eigen::Matrix<double, 12, 1> innovation_t = K_t * residual_t;

    cur_smooth_state_.t_wb += innovation_t.head(3);
    cur_smooth_state_.vel += innovation_t.segment(3, 3);
    cur_smooth_state_.acc += innovation_t.segment(6, 3);
    cur_smooth_state_.acc_jerk += innovation_t.tail(3);

    //        Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12,
    //        12>::Identity(); Eigen::Matrix<double, 12, 12> _mat = I - K_t *
    //        H_t;
    Eigen::Matrix<double, 12, 12> _mat =
        Eigen::Matrix<double, 12, 12>::Identity();
    _mat.block<12, 6>(0, 0) += K_t.block<12, 6>(0, 0);

    Eigen::Matrix<double, 12, 12> cur_smooth_t_cov_before_update = cur_smooth_t_cov_;
    cur_smooth_t_cov_ = _mat * cur_smooth_t_cov_ * _mat.transpose() +
        K_t * R_t.block(0, 0, 6, 6) * K_t.transpose();
    double cur_smooth_t_cov_gain = cur_smooth_t_cov_before_update.determinant() - cur_smooth_t_cov_.determinant();
    if (cur_smooth_t_cov_gain < -0.000001) {
      std::cout.setf(std::ios::fixed, std::ios::floatfield);
      std::cout.precision(6);
      std::cout << "PredictFilter::UpdateSmoothState: cur_smooth_t_cov_gain is less than zero: " << cur_smooth_t_cov_gain << std::endl;
    }
    //        cur_smooth_t_cov_ = cur_smooth_t_cov_ - K_t * S_t *
    //        K_t.transpose();

  }

  //     rotation
  if (update_r) {
    PredictState tmp_state = cur_smooth_state_;

    Eigen::Matrix<double, 12, 12> H_r =
        Eigen::Matrix<double, 12, 12>::Identity();
    H_r.block<3, 3>(0, 0) = LieAlgebra::inv_left_jacobian(delta_r);
    H_r = -H_r;

    Eigen::Matrix<double, 12, 12> _mat = cur_smooth_r_cov_;
    _mat.block<3, 12>(0, 0) = H_r.block<3, 3>(0, 0) * _mat.block<3, 12>(0, 0);
    _mat.block<9, 12>(3, 0) *= -1;
    Eigen::Matrix<double, 12, 12> _mat2 = _mat;
    _mat2.block<12, 3>(0, 0) =
        _mat2.block<12, 3>(0, 0) * H_r.block<3, 3>(0, 0).transpose();
    _mat2.block<12, 9>(0, 3) *= -1;

    //        Eigen::Matrix<double, 12, 12> S_r = H_r * cur_smooth_r_cov_ *
    //        H_r.transpose() + R_r; Eigen::Matrix<double, 12, 12> K_r =
    //        (S_r.ldlt().solve(H_r * cur_smooth_r_cov_)).transpose();

    Eigen::Matrix<double, 12, 12> S_r = _mat2 + R_r;
    Eigen::Matrix<double, 12, 12> K_r = (S_r.ldlt().solve(_mat)).transpose();

    Eigen::Matrix<double, 12, 1> residual_r;
    residual_r << delta_q, delta_angular, delta_angular_acc, delta_angular_jerk;
    //        residual_r << delta_q, delta_angular;

    //        double pre_cost = residual_r.transpose() *
    //        cur_smooth_r_cov_.inverse() * residual_r; LOG(ERROR) << "det2: "
    //        << S_r.determinant() << " " << R_r.determinant();
    Eigen::Matrix<double, 12, 1> innovation_r = K_r * residual_r;

    cur_smooth_state_.r_wb =
        LieAlgebra::exp(innovation_r.head(3)) * cur_smooth_state_.r_wb;
    cur_smooth_state_.angular_vel += innovation_r.segment(3, 3);
    cur_smooth_state_.angular_acc += innovation_r.segment(6, 3);
    cur_smooth_state_.angular_jerk += innovation_r.tail(3);

    //        Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12,
    //        12>::Identity(); _mat = I - K_r * H_r;
    _mat = K_r;
    _mat.block<12, 3>(0, 0) = K_r.block<12, 3>(0, 0) * H_r.block<3, 3>(0, 0);
    _mat.block<12, 3>(0, 0) *= -1;
    _mat += Eigen::Matrix<double, 12, 12>::Identity();

    Eigen::Matrix<double, 12, 12> cur_smooth_r_cov_before_update = cur_smooth_r_cov_;
    cur_smooth_r_cov_ = _mat * cur_smooth_r_cov_ * _mat.transpose() +
        K_r * R_r * K_r.transpose();
    double cur_smooth_r_cov_gain = cur_smooth_r_cov_before_update.determinant() - cur_smooth_r_cov_.determinant();
    if (cur_smooth_r_cov_gain < -0.000001) {
      std::cout.setf(std::ios::fixed, std::ios::floatfield);
      std::cout.precision(6);
      std::cout << "PredictFilter::UpdateSmoothState: cur_smooth_r_cov_gain is less than zero: " << cur_smooth_r_cov_gain << std::endl;
    }

    //        cur_smooth_r_cov_ = cur_smooth_r_cov_ - K_r * S_r *
    //        K_r.transpose();


  }

  if (states_buf_.size() > 15) {
    states_buf_.erase(states_buf_.begin(), states_buf_.end() - 10);
    CHECK_LT(states_buf_.front().timestamp, cur_smooth_state_.timestamp);
  }
  auto iter = states_buf_.begin();
  for (; iter < states_buf_.end(); iter++) {
    if (std::fabs(iter->timestamp - cur_smooth_state_.timestamp) < 0.002) {
      //CHECK_EQ(iter->update_position_cnt, 1);
      states_buf_.erase(iter);
      break;
    }
  }
  states_buf_.push_back(cur_smooth_state_);
}

void PredictFilter::UpdateVIOState(const VIOState &vio_state) {
  if (vio_state.frame_id > latest_vio_id_  || vio_state.b_state_update) {
    vio_update_num=vio_update_num+1;
    //LOG(INFO) <<

    VLOG(5) << "Update predict filter with state: " << vio_state.timestamp;
    acc_0_ = vio_state.imu_data.acc.cast<double>();
    gyr_0_ = vio_state.imu_data.gyr.cast<double>();
    linearized_ba_ =
        0.3 * last_linearized_ba_ + 0.7 * vio_state.ba.cast<double>();

    linearized_bg_ =
        0.3 * last_linearized_bg_ + 0.7 * vio_state.bw.cast<double>();
    latest_p_ = vio_state.t_wi.cast<double>();
    latest_q_ = vio_state.r_wi.toRotationMatrix().cast<double>();

//    double bg_x=linearized_bg_[0];
//    double bg_y=linearized_bg_[1];
//    double bg_z=linearized_bg_[2];
//
//    r_ic_ << 0.9740121972758088, 0.22268877838014103, -0.04135151196162753,
//        0.2242995054539643, -0.9737185467521668, 0.03952117867620647,
//        -0.03146381113373308, -0.047769233764108424, -0.9983627241110978;   // left camera to imu
//
//    latest_p_ = vio_state.t_wi.cast<double>() -
//        vio_state.r_wi.cast<double>() * r_ic_.transpose() * t_ic_;
//    latest_q_ =
//        vio_state.r_wi.toRotationMatrix().cast<double>() * r_ic_.transpose();

    latest_v_ = vio_state.v.cast<double>();
    latest_t_ = vio_state.timestamp;
    //LOG(INFO) <<"vio_update_num: " << vio_update_num<<" latest_p_: " <<latest_p_[2] ;
    imu_pre_integration_.SetInit(acc_0_, gyr_0_, linearized_ba_, linearized_bg_,
                                 latest_t_);

    latest_vio_q_ = latest_q_;

    latest_vio_id_ = vio_state.frame_id;
    vio_state_change_ = true;

    if (!initial_) {
      LOG(WARNING) << "PredictFilter initial";

      linearized_ba_ = vio_state.ba.cast<double>();
      linearized_bg_ = vio_state.bw.cast<double>();
      initial_ = true;
      last_predict_state_.timestamp = latest_t_;
      last_predict_state_.t_wb = latest_p_;
      last_predict_state_.r_wb = latest_q_;
      last_predict_state_.vel = latest_v_ / scale_t_;
      last_predict_state_.angular_vel = (gyr_0_ - linearized_bg_) / scale_t_;
      last_predict_state_.acc.setZero();
      last_predict_state_.angular_acc.setZero();
      last_predict_state_.acc_jerk.setZero();
      last_predict_state_.angular_jerk.setZero();

      last_state_t_cov_.setZero();
      last_state_t_cov_.block<3, 3>(0, 0) = 1e-2 * Eigen::Matrix3d::Identity();
      last_state_t_cov_.block<3, 3>(3, 3) =
          1e-2 * Eigen::Matrix3d::Identity() / pow(scale_t_, 2);
      last_state_t_cov_.block<3, 3>(6, 6) =
          1e2 * Eigen::Matrix3d::Identity() / pow(scale_t_, 4);
      last_state_t_cov_.block<3, 3>(9, 9) =
          1e2 * Eigen::Matrix3d::Identity() / pow(scale_t_, 6);

      last_state_r_cov_.setZero();
      last_state_r_cov_.block<3, 3>(0, 0) = 1e-2 * Eigen::Matrix3d::Identity();
      last_state_r_cov_.block<3, 3>(3, 3) =
          1e-2 * Eigen::Matrix3d::Identity() / pow(scale_t_, 2);
      last_state_r_cov_.block<3, 3>(6, 6) =
          1e2 * Eigen::Matrix3d::Identity() / pow(scale_t_, 4);
      last_state_r_cov_.block<3, 3>(9, 9) =
          1e2 * Eigen::Matrix3d::Identity() / pow(scale_t_, 6);

      cur_smooth_state_.timestamp = latest_t_;
      cur_smooth_state_.t_wb = latest_p_;
      cur_smooth_state_.r_wb = latest_q_;
      cur_smooth_state_.vel = latest_v_ / smooth_scale_t_;
      cur_smooth_state_.angular_vel =
          (gyr_0_ - linearized_bg_) / smooth_scale_t_;
      cur_smooth_state_.acc.setZero();
      cur_smooth_state_.angular_acc.setZero();
      cur_smooth_state_.acc_jerk.setZero();
      cur_smooth_state_.angular_jerk.setZero();

      cur_smooth_t_cov_.setZero();
      cur_smooth_t_cov_.block<3, 3>(0, 0) = 1e-2 * Eigen::Matrix3d::Identity();
      cur_smooth_t_cov_.block<3, 3>(3, 3) =
          1e-2 * Eigen::Matrix3d::Identity() / pow(smooth_scale_t_, 2);
      cur_smooth_t_cov_.block<3, 3>(6, 6) =
          1e2 * Eigen::Matrix3d::Identity() / pow(smooth_scale_t_, 4);
      cur_smooth_t_cov_.block<3, 3>(9, 9) =
          1e2 * Eigen::Matrix3d::Identity() / pow(smooth_scale_t_, 6);

      cur_smooth_r_cov_.setZero();
      cur_smooth_r_cov_.block<3, 3>(0, 0) = 1e-2 * Eigen::Matrix3d::Identity();
      cur_smooth_r_cov_.block<3, 3>(3, 3) =
          1e-2 * Eigen::Matrix3d::Identity() / pow(smooth_scale_t_, 2);
      cur_smooth_r_cov_.block<3, 3>(6, 6) =
          1e2 * Eigen::Matrix3d::Identity() / pow(smooth_scale_t_, 4);
      cur_smooth_r_cov_.block<3, 3>(9, 9) =
          1e2 * Eigen::Matrix3d::Identity() / pow(smooth_scale_t_, 6);
    }
    last_linearized_ba_ = vio_state.ba.cast<double>();
    last_linearized_bg_ = vio_state.bw.cast<double>();
  }
}




// 变加速模型
// Variable acceleration model seem no perfect, all noise is cause by acc jerk
// noise, it can cause matrix irrerersible add some noise to p,v,a to compensate
// model unperfect matrix determinant too small, test add some to noise p, v, a
// is better. b_{k+1} = b_{k} + \delta w (acc jerk white noise) a_{k+1} = a_{k}
// + b_{k} * \Delta t v_{k+1} = v_{k} + 0.5 * (a_{k+1} + a_{k}) * \Delta t
// p_{k+1} = p_{k} + v_{k} * \Delta t + 0.5 * (a_{k+1} + a_{k}) * \Delta t *
// \Delta t
void PredictFilter::PredictModel(const double predict_time,
                                 const PredictState &last_state,
                                 PredictState &cur_state, // NOLINT
                                 Eigen::MatrixXd &jacobian_t_state,
                                 Eigen::MatrixXd &jacobian_t_noise, // NOLINT
                                 Eigen::MatrixXd &jacobian_r_state, // NOLINT
                                 Eigen::MatrixXd &jacobian_r_noise, // NOLINT
                                 const double &scale_t,
                                 const bool &calculate_jacobian) {
  double dt = predict_time - last_state.timestamp;
  if(dt>predict_threshold){
    dt=predict_threshold;
  }

//  LOG(INFO) << "predict model: " << last_state.timestamp << " --> "
//          << predict_time << ", dt: " << dt;

//  if(dt<=0.001){
//    return;
//  }

  dt *= scale_t; // normalize dt, make dt = 1, avoid numerical instability
  double dt2 = dt * dt;
  double dt3 = dt2 * dt;

  cur_state.acc_jerk = last_state.acc_jerk;
  cur_state.angular_jerk = last_state.angular_jerk;

  cur_state.acc = last_state.acc + last_state.acc_jerk * dt;
  cur_state.angular_acc = last_state.angular_acc + last_state.angular_jerk * dt;

  const Eigen::Vector3d mid_acc = 0.5 * (cur_state.acc + last_state.acc);
  const Eigen::Vector3d mid_angular_acc =
      0.5 * (cur_state.angular_acc + last_state.angular_acc);

  cur_state.vel = last_state.vel + mid_acc * dt;
  cur_state.angular_vel = last_state.angular_vel + mid_angular_acc * dt;

  cur_state.t_wb = last_state.t_wb + last_state.vel * dt + 0.5 * mid_acc * dt2;

//  std::cout<<"PredictModel: " << last_state.angular_vel * dt +
//      0.5 * mid_angular_acc * dt2 << std::endl;
//  std::cout<<"PredictModel: " << last_state.angular_vel << std::endl;
//  std::cout<<"PredictModel: " << mid_angular_acc << std::endl;
//  std::cout<<"PredictModel: " << dt << std::endl;

  cur_state.r_wb =
      last_state.r_wb * LieAlgebra::exp(last_state.angular_vel * dt +
                                        0.5 * mid_angular_acc * dt2);
  //std::cout<<"success " << std::endl;
  cur_state.timestamp = predict_time;

  if (!calculate_jacobian) return;

  // position
  jacobian_t_state = Eigen::Matrix<double, 12, 12>::Zero();
  jacobian_t_state.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  jacobian_t_state.block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();
  jacobian_t_state.block<3, 3>(0, 6) = 0.5 * dt2 * Eigen::Matrix3d::Identity();
  jacobian_t_state.block<3, 3>(0, 9) = 0.25 * dt3 * Eigen::Matrix3d::Identity();

  jacobian_t_state.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  jacobian_t_state.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
  jacobian_t_state.block<3, 3>(3, 9) = 0.5 * dt2 * Eigen::Matrix3d::Identity();

  jacobian_t_state.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
  jacobian_t_state.block<3, 3>(6, 9) = dt * Eigen::Matrix3d::Identity();

  jacobian_t_state.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

  jacobian_t_noise = Eigen::MatrixXd::Zero(12, 12);
  jacobian_t_noise.block<3, 3>(0, 0) = 0.25 * dt3 * Eigen::Matrix3d::Identity();
  jacobian_t_noise.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  jacobian_t_noise.block<3, 3>(3, 0) = 0.5 * dt2 * Eigen::Matrix3d::Identity();
  jacobian_t_noise.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
  jacobian_t_noise.block<3, 3>(6, 0) = dt * Eigen::Matrix3d::Identity();
  jacobian_t_noise.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity();
  jacobian_t_noise.block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();

  // rotation
  jacobian_r_state = Eigen::Matrix<double, 12, 12>::Zero();
  jacobian_r_state.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  Eigen::Vector3d _angle =
      last_state.angular_vel * dt + 0.5 * mid_angular_acc * dt2;
  jacobian_r_state.block<3, 3>(0, 3) =
      cur_state.r_wb * LieAlgebra::right_jacobian(_angle) * dt;
  jacobian_r_state.block<3, 3>(0, 6) =
      cur_state.r_wb * LieAlgebra::right_jacobian(_angle) * dt2 * 0.5;
  jacobian_r_state.block<3, 3>(0, 9) =
      cur_state.r_wb * LieAlgebra::right_jacobian(_angle) * dt3 * 0.25;

  jacobian_r_state.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  jacobian_r_state.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
  jacobian_r_state.block<3, 3>(3, 9) = 0.5 * dt2 * Eigen::Matrix3d::Identity();

  jacobian_r_state.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
  jacobian_r_state.block<3, 3>(6, 9) = dt * Eigen::Matrix3d::Identity();

  jacobian_r_state.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

  jacobian_r_noise = Eigen::MatrixXd::Zero(12, 12);
  jacobian_r_noise.block<3, 3>(0, 0) =
      cur_state.r_wb * LieAlgebra::right_jacobian(_angle) * 0.25 * dt3;
  jacobian_r_noise.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  jacobian_r_noise.block<3, 3>(3, 0) = 0.5 * dt2 * Eigen::Matrix3d::Identity();
  jacobian_r_noise.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
  jacobian_r_noise.block<3, 3>(6, 0) = dt * Eigen::Matrix3d::Identity();
  jacobian_r_noise.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity();
  jacobian_r_noise.block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();
}



void PredictFilter::PredictModelOptional(const double predict_time,
                                 const PredictState &last_state,
                                 PredictState &cur_state, // NOLINT
                                 Eigen::MatrixXd &jacobian_t_state,
                                 Eigen::MatrixXd &jacobian_t_noise, // NOLINT
                                 Eigen::MatrixXd &jacobian_r_state, // NOLINT
                                 Eigen::MatrixXd &jacobian_r_noise, // NOLINT
                                 const double &scale_t,
                                 const bool &calculate_jacobian,
    const bool predict_t_flag,
    const bool predict_r_flag) {


  double dt = predict_time - last_state.timestamp;
//  if(dt<=0.001){
//    return;
//  }

  VLOG(5) << "predict model: " << last_state.timestamp << " --> "
          << predict_time << ", dt: " << dt;
  //        CHECK_GT(dt, 0) << dt;

  dt *= scale_t; // normalize dt, make dt = 1, avoid numerical instability
  double dt2 = dt * dt;
  double dt3 = dt2 * dt;

  if(predict_t_flag){
    cur_state.acc_jerk = last_state.acc_jerk;
    cur_state.acc = last_state.acc + last_state.acc_jerk * dt;
    const Eigen::Vector3d mid_acc = 0.5 * (cur_state.acc + last_state.acc);
    cur_state.vel = last_state.vel + mid_acc * dt;
    cur_state.t_wb = last_state.t_wb + last_state.vel * dt + 0.5 * mid_acc * dt2;
    if(calculate_jacobian) {
      // position
      jacobian_t_state = Eigen::Matrix<double, 12, 12>::Zero();
      jacobian_t_state.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      jacobian_t_state.block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();
      jacobian_t_state.block<3, 3>(0, 6) =
          0.5 * dt2 * Eigen::Matrix3d::Identity();
      jacobian_t_state.block<3, 3>(0, 9) =
          0.25 * dt3 * Eigen::Matrix3d::Identity();

      jacobian_t_state.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
      jacobian_t_state.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
      jacobian_t_state.block<3, 3>(3, 9) =
          0.5 * dt2 * Eigen::Matrix3d::Identity();

      jacobian_t_state.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
      jacobian_t_state.block<3, 3>(6, 9) = dt * Eigen::Matrix3d::Identity();

      jacobian_t_state.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

      jacobian_t_noise = Eigen::MatrixXd::Zero(12, 12);
      jacobian_t_noise.block<3, 3>(0, 0) =
          0.25 * dt3 * Eigen::Matrix3d::Identity();
      jacobian_t_noise.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
      jacobian_t_noise.block<3, 3>(3, 0) =
          0.5 * dt2 * Eigen::Matrix3d::Identity();
      jacobian_t_noise.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
      jacobian_t_noise.block<3, 3>(6, 0) = dt * Eigen::Matrix3d::Identity();
      jacobian_t_noise.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity();
      jacobian_t_noise.block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();
    }

  }

  if(predict_r_flag){
    cur_state.angular_jerk = last_state.angular_jerk;

    cur_state.angular_acc = last_state.angular_acc + last_state.angular_jerk * dt;
    const Eigen::Vector3d mid_angular_acc =
        0.5 * (cur_state.angular_acc + last_state.angular_acc);
    cur_state.angular_vel = last_state.angular_vel + mid_angular_acc * dt;
    cur_state.r_wb =
        last_state.r_wb * LieAlgebra::exp(last_state.angular_vel * dt +
            0.5 * mid_angular_acc * dt2);

    if(calculate_jacobian) {
      // rotation
      jacobian_r_state = Eigen::Matrix<double, 12, 12>::Zero();
      jacobian_r_state.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      Eigen::Vector3d _angle =
          last_state.angular_vel * dt + 0.5 * mid_angular_acc * dt2;
      jacobian_r_state.block<3, 3>(0, 3) =
          cur_state.r_wb * LieAlgebra::right_jacobian(_angle) * dt;
      jacobian_r_state.block<3, 3>(0, 6) =
          cur_state.r_wb * LieAlgebra::right_jacobian(_angle) * dt2 * 0.5;
      jacobian_r_state.block<3, 3>(0, 9) =
          cur_state.r_wb * LieAlgebra::right_jacobian(_angle) * dt3 * 0.25;

      jacobian_r_state.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
      jacobian_r_state.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
      jacobian_r_state.block<3, 3>(3, 9) =
          0.5 * dt2 * Eigen::Matrix3d::Identity();

      jacobian_r_state.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
      jacobian_r_state.block<3, 3>(6, 9) = dt * Eigen::Matrix3d::Identity();

      jacobian_r_state.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

      jacobian_r_noise = Eigen::MatrixXd::Zero(12, 12);
      jacobian_r_noise.block<3, 3>(0, 0) =
          cur_state.r_wb * LieAlgebra::right_jacobian(_angle) * 0.25 * dt3;
      jacobian_r_noise.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
      jacobian_r_noise.block<3, 3>(3, 0) =
          0.5 * dt2 * Eigen::Matrix3d::Identity();
      jacobian_r_noise.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
      jacobian_r_noise.block<3, 3>(6, 0) = dt * Eigen::Matrix3d::Identity();
      jacobian_r_noise.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity();
      jacobian_r_noise.block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();

    }

  }
  cur_state.timestamp = predict_time;

  if (!calculate_jacobian) return;

}
} // namespace ST_Predict
