//
// Created by SENSETIME\zhaolinsheng on 2021/8/23.
//
#ifndef SENSE_PDR_PDR_H
#define SENSE_PDR_PDR_H

#include "util.h"
#include <string>
#include <deque>

//using namespace std;
namespace pdr {

enum ResultType {
  ResultType_ERROR = -1,
  RESULT_TYPE_INIT =
  0,// sensor data is too small to be used for positioning. Data for at least 3 seconds
  RESULT_TYPE_NORMAL = 1,
};

//using namespace std;
enum MotionStatus {
  MOTION_STATUS_STAND = 0,
  MOTION_STATUS_MOVE = 1,
  MOTION_STATUS_UNKOWN = 3,
  MOTION_STATUS_INTI = 4,
  MOTION_STATUS_STEP_DETECTED = 5,
};


struct PdrResult {
  double time_stamp = 0.0;
  double delta_x =
      0.0;// It should be in the world coordinate system (East-North-Up coordina)
  double delta_y =
      0.0;// It should be in the world coordinate system (East-North-Up coordina)
  //int32_t motion_status=0;
  double confidence = 1.0;
  //int32_t result_type = -1;
  ResultType result_type = RESULT_TYPE_INIT;
  MotionStatus motion_status = MOTION_STATUS_UNKOWN;
  std::string result_type_info = "default";
  double velocity2d = 0.0f;
};

class PdrComputer {
 private:
  void ParamInit();
  void
  ProcessIncrementalAccData(std::deque<Eigen::Vector4d> acc_request_raw_data_vector);
  int32_t ComputerStep();
  void FindPeaks(std::deque<double> &src, float distance, float threshold,
                 std::vector<int32_t> &peaks_index, float threshold_offset);
  double ComputeStrideLength();
  double ComputeVariance(std::deque<double> &accs, int32_t start, int32_t end);
  MotionStatus MotionStatusCheck();
  void ComputeAccelerometerFrequency();

  //double ComputeHeadings(Eigen::Vector4d rv_raw_data_current);
  PdrResult
  ComputeRelPositions(double timestamp, double stride_lengths, double yaw);

  double timestamp_unit;


 public:

  //timestamp nanosecond
  void OnNewAcceByString(const std::string acc_data_line);
  void OnNewAcce(double t_ns, double x, double y, double z);
  //timestamp nanosecond
  void OnNewRvByString(const std::string rv_data_line);
  void OnNewRv(double t_ns, double x, double y, double z, double w);
  //timestamp nanosecond
  PdrResult GetPdrResult(const double t);
  //timestamp nanosecond
  PdrResult GetPdrVelocity(const double t);
  double ComputeHeadings(Eigen::Vector4d rv_raw_data_current);


  PdrComputer();
  ~PdrComputer();


  double last_peak_timestamp;
  double new_peak_timestamp;

  std::deque<double> acc_norm_data_filter_window;
  std::deque<double> acc_norm_data_window;
  std::deque<double> acc_raw_data_timestamp_window;
  std::deque<double> acc_norm_raw_data;
  //std::deque<Vector4d> acc_raw_data_vector_window;
  std::vector<int32_t> peaks_indexs_window;

  double sensor_time_interval_new_and_old;
  std::deque<double> acc_norm_low_pass_filter;
  double acc_filter_weight;
  double last_update_acc_frequency_timestamp;
  int32_t last_peak_index, new_peak_index;
  int32_t accelerometer_data_length;
  int32_t sensor_collect_frequency;

  int32_t average_peak_width;
  double step_length, step_frequency, step_variance;
  float peak_distance;
  double peak_threshold;
  double last_window_step_timestamp_hold;

  double accelerometer_frequency;
  double gravity_offset;
  std::deque<double> step_happen_timestamp_deque;
  std::vector<double> step_happen_acc_mean_deque;
  std::vector<double> step_happen_acc_std_deque;
  MotionStatus motion_status;
  double test_tmp;
  double velocity2d;

  Eigen::Vector4d rv_raw_data_current;
  double rv_raw_data_current_timestamp{};

//  static PdrComputer &instance() {
//      static PdrComputer s_global_object;
//      s_global_object.ParamInit();
//      return s_global_object;
//  }
};


/***reserved  interface*/
//timestamp nanosecond
void onNewAcceByString(const std::string acc_data_line);

//timestamp nanosecond
void onNewRvByString(const std::string rv_data_line);

//timestamp nanosecond
PdrResult getPdrResult(const double t);

void
onNewAcceByMutilString(const std::vector<std::string> mutil_acc_data_lines);
void onNewAccelerometer(const double t,
                        const double x,
                        const double y,
                        const double z);
void onNewRotationVector(const double t,
                         const double x,
                         const double y,
                         const double z,
                         const double w);
//0:
void setParameter(const int32_t para_name_index, std::string data);

}//namespace pdr

#endif //SENSE_PDR_PDR_H
