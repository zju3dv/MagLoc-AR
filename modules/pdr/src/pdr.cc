//
// Created by SENSETIME\zhaolinsheng on 2021/8/23.
//

#include "pdr.h"
namespace pdr {

static const double kAccExpFilterTimeConstant = 0.017570492957164466;

//PdrComputer &pc = PdrComputer::instance();
PdrComputer::PdrComputer(void) {
  ParamInit();
}

PdrComputer::~PdrComputer() {}

void PdrComputer::ParamInit() {
  last_peak_index = 0;
  new_peak_index = 0;
  //accelerometer_data_length = 250;
  //acc_static_count = 0;
  average_peak_width = 30;
  //get_first_acc = false;

  step_length = 0.7;
  step_frequency = 0.0;
  step_variance = 0.0;
  last_update_acc_frequency_timestamp = 0.0;
  accelerometer_frequency = 30.0;
  gravity_offset = 9.8f;
  peak_threshold = 4.0;

  acc_filter_weight = std::exp(-(1.0 / accelerometer_frequency) / kAccExpFilterTimeConstant);

  sensor_time_interval_new_and_old = 0.2;//秒/s
  sensor_collect_frequency = 400;//Hz
  timestamp_unit = 1e-9;
  accelerometer_data_length = 2 * sensor_collect_frequency;
  //last_window_step_timestamp_hold=1e20;
  if (sensor_collect_frequency > 50)
    peak_distance = sensor_collect_frequency * 0.4;
  else
    peak_distance = 20.0f;
  motion_status = MOTION_STATUS_INTI;
}


void
PdrComputer::ProcessIncrementalAccData(std::deque<Eigen::Vector4d> acc_request_raw_data_vector) {
  if (acc_request_raw_data_vector.size() <= 0) {
    return;
  }

  if (this->acc_raw_data_timestamp_window.size() > 0
      && abs(acc_request_raw_data_vector.front()[0]
                 - this->acc_raw_data_timestamp_window.back())
          < sensor_time_interval_new_and_old) {
    //reset window()

  }

  double acc_norm_filter_tmp;
  double acc_timestamp_tmp;
  double acc_x_tmp;
  double acc_y_tmp;
  double acc_z_tmp;

  if (this->acc_raw_data_timestamp_window.size() == 0) {
    acc_timestamp_tmp =
        acc_request_raw_data_vector[0][0] * timestamp_unit;// 单位　秒
    //cout<<acc_timestamp_tmp<<endl;
    acc_x_tmp = acc_request_raw_data_vector[0][1];
    acc_y_tmp = acc_request_raw_data_vector[0][2];
    acc_z_tmp = acc_request_raw_data_vector[0][3];
    acc_norm_filter_tmp = sqrt(acc_x_tmp * acc_x_tmp + acc_y_tmp * acc_y_tmp
                                   + acc_z_tmp * acc_z_tmp);
    this->acc_norm_data_filter_window.push_back(acc_norm_filter_tmp);
    this->acc_raw_data_timestamp_window.push_back(acc_timestamp_tmp);
  }

  assert(this->acc_raw_data_timestamp_window.size() > 0);
  assert(this->acc_norm_data_filter_window.size() > 0);
  assert(this->acc_norm_data_filter_window.size()
             == this->acc_raw_data_timestamp_window.size());

  for (size_t i = 0; i < acc_request_raw_data_vector.size(); i++) {
    acc_timestamp_tmp = acc_request_raw_data_vector[i][0] * timestamp_unit;
    acc_x_tmp = acc_request_raw_data_vector[i][1];
    acc_y_tmp = acc_request_raw_data_vector[i][2];
    acc_z_tmp = acc_request_raw_data_vector[i][3];
    acc_norm_filter_tmp = sqrt(acc_x_tmp * acc_x_tmp + acc_y_tmp * acc_y_tmp
                                   + acc_z_tmp * acc_z_tmp);
    this->acc_norm_data_window.push_back(acc_norm_filter_tmp);
    acc_norm_filter_tmp = acc_norm_filter_tmp * (1 - acc_filter_weight)
        + acc_filter_weight * acc_norm_data_filter_window.back();
    this->acc_norm_data_filter_window.push_back(acc_norm_filter_tmp);
    this->acc_raw_data_timestamp_window.push_back(acc_timestamp_tmp);

    if (acc_norm_data_filter_window.size()
        > accelerometer_data_length)
      acc_norm_data_filter_window.pop_front();
    if (acc_raw_data_timestamp_window.size()
        > accelerometer_data_length)
      acc_raw_data_timestamp_window.pop_front();
    if (acc_norm_data_window.size()
        > accelerometer_data_length)
      acc_norm_data_window.pop_front();
  }

  ComputeAccelerometerFrequency();
}

int32_t PdrComputer::ComputerStep() {

  //ComputeAccelerometerFrequency();
  std::vector<int32_t> peaks_indexs;
//    cout<< "acc_norm_data_filter_window size "<<acc_norm_data_filter_window.size()  <<endl;
//    if(this->acc_norm_data_filter_window.size()<this->accelerometer_data_length){
//        FindPeaks(acc_norm_data_filter_window, peak_distance, 0.3f, peaks_indexs, accelerometer_threshold_offset);
//        return 0;
//    }
  this->peak_threshold = std::max(this->peak_threshold, compute_deque_std(
      acc_norm_data_filter_window) * 0.4);
  //cout<< "peak_distance "<<peak_distance  <<endl;
  FindPeaks(acc_norm_data_filter_window,
            peak_distance,
            this->peak_threshold,
            peaks_indexs,
            gravity_offset);
  //cout<< "peaks num: "<<peaks_indexs.size()  <<endl;
  int32_t step_num = 0;

#ifdef DEBUG
  //std::cout << "DEBUG::" << std::endl;
#endif

//    cout << "compute_deque_mean " << compute_deque_mean(
//        acc_norm_data_filter_window) << endl;
//    cout << "compute_deque_std " << compute_deque_std(
//        acc_norm_data_filter_window) << endl;
  this->step_happen_acc_mean_deque.push_back(compute_deque_mean(
      acc_norm_data_filter_window));
  this->step_happen_acc_std_deque.push_back(compute_deque_std(
      acc_norm_data_filter_window));

  std::vector<int32_t> steps_dt_vector;
  if (peaks_indexs.size() > 0) {

    double step_dt_tmp;
    double step_is_new_dt_tmp;
    for (size_t i = 0; i < peaks_indexs.size(); i++) {

      step_is_new_dt_tmp = acc_raw_data_timestamp_window[peaks_indexs[i]]
          - this->last_window_step_timestamp_hold;
//          cout<< "acc_raw_data_timestamp_window[ peaks_indexs[i] ] : "<<acc_raw_data_timestamp_window[ peaks_indexs[i] ]-1560847175171 <<endl;
//          cout<< "this->last_window_step_timestamp_hold : "<<this->last_window_step_timestamp_hold-1560847175171 <<endl;
//          cout<< "step_is_new_dt_tmp : "<<step_is_new_dt_tmp <<endl;
      if (i == 0) {
        step_dt_tmp = step_is_new_dt_tmp;
      } else {
        step_dt_tmp = acc_raw_data_timestamp_window[peaks_indexs[i]]
            - acc_raw_data_timestamp_window[peaks_indexs[i - 1]];
      }

      if (step_is_new_dt_tmp > sensor_time_interval_new_and_old) {
        step_num = step_num + 1;
        steps_dt_vector.push_back(step_dt_tmp);
        this->step_happen_timestamp_deque.push_back(
            this->last_window_step_timestamp_hold);
      }
    }
  }
  //cout<< "step_num: "<<step_num <<'\n' <<endl;
//    for(int i=0;i<steps_dt_vector.size();i++){
//        cout<< "steps_dt_vector : "<<steps_dt_vector[i] <<endl;
//    }
  // cout<< "step_num: "<<step_num <<'\n' <<endl;

  if (acc_raw_data_timestamp_window.size() > 0 && peaks_indexs.size() > 0)
    this->last_window_step_timestamp_hold =
        this->acc_raw_data_timestamp_window[peaks_indexs.back()];
  this->peaks_indexs_window = peaks_indexs;
  return step_num;
//    for(int i=0;i<peaks_indexs.size();i++){
//        cout<< acc_raw_data_timestamp_window[peaks_indexs[i]]/1000 - acc_raw_data_timestamp_window[0]/1000  <<endl;
//    }
//    double last_peak_timestamp,new_peak_timestamp;
//    last_peak_timestamp = acc_raw_data_timestamp[peaks_indexs[peaks_indexs.size() - 2]];
//    new_peak_timestamp = acc_raw_data_timestamp[peaks_indexs.back()];
//    double step_dt = new_peak_timestamp - last_peak_timestamp;
}

void PdrComputer::FindPeaks(std::deque<double> &src,
                            float distance,
                            float threshold,
                            std::vector<int32_t> &peaks_index,
                            float threshold_offset) {
  if (src.size() < 2 * distance) return;

  float min_peak_height = threshold + threshold_offset;
  std::vector<int32_t> sign;
  int32_t max_index = 0;

  for (int32_t i = 1; i < src.size(); i++) {
    double diff = src[i] - src[i - 1];
    if (diff > 0) {
      sign.push_back(1);
    } else if (diff < 0) {
      sign.push_back(-1);
    } else {
      sign.push_back(0);
    }
  }

  for (int32_t j = 1; j < src.size() - 1; j++) {
    double diff = sign[j] - sign[j - 1];
    if (diff < 0 && src[j] > min_peak_height) {
      peaks_index.push_back(j);
      max_index++;
    }
  }

  std::vector<int32_t> flag_max_index;
  std::vector<int32_t> idelete;
  std::vector<int32_t> temp_max_index;

  int32_t bigger = 0;
  double tempvalue = 0;
  int32_t i, j, k;
  // 波峰
  for (int32_t i = 0; i < max_index; i++) {
    flag_max_index.push_back(0);
    idelete.push_back(0);
  }
  for (i = 0; i < max_index; i++) {
    tempvalue = -1;
    for (j = 0; j < max_index; j++) {
      if (!flag_max_index[j]) {
        if (src[peaks_index[j]] > tempvalue) {
          bigger = j;
          tempvalue = src[peaks_index[j]];
        }
      }
    }
    flag_max_index[bigger] = 1;
    if (!idelete[bigger]) {
      for (k = 0; k < max_index; k++) {
        idelete[k] |= (peaks_index[k] - distance <= peaks_index[bigger]
            & peaks_index[bigger] <= peaks_index[k] + distance);
      }
      idelete[bigger] = 0;
    }
  }
  for (i = 0, j = 0; i < max_index; i++) {
    if (!idelete[i]) {
      temp_max_index.push_back(peaks_index[i]);
      j++;
    }
  }
  peaks_index.clear();
  for (int32_t i = 0; i < temp_max_index.size(); i++)
    peaks_index.push_back(temp_max_index[i]);

}

double PdrComputer::ComputeStrideLength() {

  if (this->peaks_indexs_window.size() < 2) return 1;

  double last_peak_index, new_peak_index;
  double step_dt;
  double step_length = 0;

  last_peak_index =
      this->peaks_indexs_window[this->peaks_indexs_window.size() - 2];

  new_peak_index = this->peaks_indexs_window.back();

  this->last_peak_timestamp =
      this->acc_raw_data_timestamp_window[this->peaks_indexs_window[
          this->peaks_indexs_window.size() - 2]];
  this->new_peak_timestamp =
      this->acc_raw_data_timestamp_window[this->peaks_indexs_window.back()];
  step_dt = (this->new_peak_timestamp - this->last_peak_timestamp);

  step_frequency = 1 / step_dt;
  step_variance =
      ComputeVariance(this->acc_norm_data_window,
                      last_peak_index,
                      new_peak_index);

//  std::cout<<"last_peak_index: "<<last_peak_index<<std::endl;
//  std::cout<<"new_peak_index: "<<new_peak_index<<std::endl;
//  std::cout<<"step_frequency "<<step_frequency<<std::endl;
//  std::cout<<"step_variance "<<step_variance<<std::endl;
//  std::cout<<"step_dt "<<step_dt<<std::endl;
  step_length = 0.2844 + 0.2231 * step_frequency + 0.0426 * step_variance;
  //std::cout<<"step_length "<<step_length<<"\n"<<std::endl;

  this->velocity2d = step_length / step_dt;
//  if (this->velocity2d > 0) {
//    std::cout << "step_length " << step_length << std::endl;
//    std::cout << "step_dt " << step_dt << std::endl;
//    std::cout << "velocity2d " << velocity2d << std::endl;
//    std::cout << "\n " << velocity2d << std::endl;
//  }
  return step_length;
}

double PdrComputer::ComputeVariance(std::deque<double> &accs,
                                    int32_t start,
                                    int32_t end) {
  if (accs.size() < 10) {
    return 0.0;
  } else if (accs.size() < (end - start + 1)) {
    start = 0;
  }

  double sum = 0.0, var = 0.0;
  for (int32_t i = start; i <= end; i++)
    sum += accs[i];

  sum /= (end - start + 1);

  for (int32_t i = start; i <= end; i++) {
    var += (accs[i] - sum) * (accs[i] - sum);
  }

  return var / (end - start + 1);
}

//0 STAND ; 1 MOVE ;3 UNKOWN
MotionStatus PdrComputer::MotionStatusCheck() {

//    if(this->acc_raw_data_timestamp_window.size()>this->accelerometer_data_length-2){
//       //cout<< "ttt" <<endl;
//    }

  if (acc_raw_data_timestamp_window.back() - this->new_peak_timestamp < 0.5) {
    this->motion_status = MOTION_STATUS_MOVE;
    return this->motion_status;
  }

  if (acc_raw_data_timestamp_window.back() - this->new_peak_timestamp > 2) {
    this->motion_status = MOTION_STATUS_STAND;
    return this->motion_status;
  }

//    if (accelerometer_frequency < 20) {
//        this->motion_status = MotionStatus_UNKOWN;
//        return this->motion_status;
//    }

  //设置的
  auto N = acc_norm_data_filter_window.size();
  //取最近１秒的数据
  double average_acc_diff = 0.0;
  for (int i = this->new_peak_index; i < N; i++) {
    average_acc_diff +=
        std::abs(acc_norm_data_filter_window.at(i) - this->gravity_offset);
  }
  average_acc_diff /= (N - this->new_peak_index);

  if (average_acc_diff < 0.3
      || (acc_raw_data_timestamp_window.back() - new_peak_timestamp)
          > (new_peak_timestamp - last_peak_timestamp) + 0.5) {
    this->motion_status = MOTION_STATUS_STAND;
    return this->motion_status;
  }

  this->motion_status = MOTION_STATUS_MOVE;
//    if(this->accelerometer_frequency<30){
//        auto N = acc_norm_data_filter_window.size();
//
//    } else{
//
//    }
  return this->motion_status;
}

void PdrComputer::ComputeAccelerometerFrequency() {

  static int N = 25;
  if (acc_raw_data_timestamp_window.back()
      - last_update_acc_frequency_timestamp < 2.0)
    return;
  if (acc_raw_data_timestamp_window.size() < N)
    return;

  double dt = acc_raw_data_timestamp_window.back()
      - acc_raw_data_timestamp_window[acc_raw_data_timestamp_window.size()
          - N];
  accelerometer_frequency = N / dt;
  N = accelerometer_frequency * 2 + 6;
  if (N > 130) N = 130;

  acc_filter_weight = std::exp(-(1.0 / accelerometer_frequency) / kAccExpFilterTimeConstant);

  last_update_acc_frequency_timestamp = acc_raw_data_timestamp_window.back();
  average_peak_width = accelerometer_frequency * 0.6;
  if (average_peak_width < 6) {
    average_peak_width = 6;
  }

  this->sensor_collect_frequency = accelerometer_frequency;
  //std::cout<<"sensor_collect_frequency: "<<sensor_collect_frequency <<std::endl;
  this->accelerometer_data_length = 2 * this->sensor_collect_frequency;
  //last_window_step_timestamp_hold=1e20;
  if (this->sensor_collect_frequency > 50)
    this->peak_distance = this->sensor_collect_frequency * 0.4;
  else
    this->peak_distance = 20.0f;
//    // printf("[LVO] acc freq: %.2f, step: %d\n", accelerometer_frequency, total_step_count);
//    if (is_web) {
//        if (accelerometer_frequency > 25.0) {
//            peak_distance = 12.0f;
//        } else if (accelerometer_frequency < 15.0) {
//            peak_distance = 5.0f;
//        } else {
//            peak_distance = 9.0f;
//        }
//    } else {
//        peak_distance = accelerometer_frequency * 0.6 - 10;
//    }
  //peak_distance = accelerometer_frequency * 0.6 - 10;

}

//vector4d(x,y,z,w)
//Eigen::Quaterniond q(w,x,y,z)
//Eigen::Quaterniond q( Vector4d(x,y,z,w) )
//Eigen::Quaterniond q( Matrix3d(R) )
//ref: https://blog.csdn.net/xiaoma_bk/article/details/79082629
//ref: https://zhuanlan.zhihu.com/p/259999988
double PdrComputer::ComputeHeadings(Eigen::Vector4d rv_raw_data_current) {
  double yaw_current;
  Eigen::Quaterniond q(rv_raw_data_current);
  double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
  double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
  //Quaterniond q(rv_raw_data_current.w(),rv_raw_data_current.x(),rv_raw_data_current.y(),rv_raw_data_current.z());

//    Eigen::Vector3d eulerAngle =
//      q.toRotationMatrix().eulerAngles(2, 1, 0);//    yaw pitch roll
//  yaw_current = eulerAngle.x();
//  yaw_current = quart_to_rpy(rv_raw_data_current[3],
//                             rv_raw_data_current[0],
//                             rv_raw_data_current[1],
//                             rv_raw_data_current[2]);

//    cout<<eulerAngle.x()*180/PI<<endl;
//    cout<<eulerAngle.y()*180/PI<<endl;
//    cout<<eulerAngle.z()*180/PI<<endl;
//    yaw_current=yaw_previous*heading_low_pass_weigth+(1-heading_low_pass_weigth)*yaw_current;
//    this->yaw_previous=yaw_current;
  //yaw_current = get_orientation_from_matrix(q.matrix());
  //std::cout<<"yaw_current "<< yaw_current*57.3 <<std::endl;
  //std::cout<<"yaw_current0, "<< yaw_current *180/3.14 <<std::endl;
  // std::cout<<"yaw_current1, "<< fmod(-yaw_current,2 * M_PI) *180/3.14 <<std::endl;

  float yaw = (float) atan2(siny_cosp, cosy_cosp);
  //std::cout<<"yaw, "<< yaw *180/3.14 <<std::endl;
  //return fmod(-yaw_current,2 * M_PI) ;
  return fmod(yaw, 2 * M_PI);
}

PdrResult PdrComputer::ComputeRelPositions(double timestamp,
                                           double stride_lengths,
                                           double yaw) {
  double heading = yaw;
  double delta_x;
  double delta_y;
  delta_x = -stride_lengths * sin(heading);
  delta_y = stride_lengths * cos(heading);
//    cout<<"delta_x "<<delta_x<<endl;
//    cout<<"delta_y "<<delta_y<<endl;

  PdrResult relPositions;
  relPositions.time_stamp = timestamp;
  relPositions.delta_x = delta_x;
  relPositions.delta_y = delta_y;

  return relPositions;
}

void PdrComputer::OnNewAcceByString(const std::string acc_data_line) {
  double t, x, y, z;
  std::vector<std::string> time_x_y_z_vector;
  split_string(acc_data_line, time_x_y_z_vector, ",");
  t = std::stod(time_x_y_z_vector[0]);
  x = std::stod(time_x_y_z_vector[1]);
  y = std::stod(time_x_y_z_vector[2]);
  z = std::stod(time_x_y_z_vector[3]);
  Eigen::Vector4d acc_vector;
  acc_vector << t, x, y, z;
  std::deque<Eigen::Vector4d> acc_request_raw_data_vector;
  acc_request_raw_data_vector.push_back(acc_vector);
  //pdr computer input data
  this->ProcessIncrementalAccData(acc_request_raw_data_vector);
}

void PdrComputer::OnNewAcce(double t_ns, double x, double y, double z) {
  Eigen::Vector4d acc_vector;
  acc_vector << t_ns, x, y, z;
  std::deque<Eigen::Vector4d> acc_request_raw_data_vector;
  acc_request_raw_data_vector.emplace_back(acc_vector);
  this->ProcessIncrementalAccData(acc_request_raw_data_vector);
}

void PdrComputer::OnNewRvByString(const std::string rv_data_line) {
  double tv_t, rv_w, rv_x, rv_y, rv_z;
  Eigen::Vector4d rv_tmp;
  std::vector<Eigen::Vector4d> rv_list;
  std::vector<std::string> time_x_y_z_w_vector;
  split_string(rv_data_line, time_x_y_z_w_vector, ",");
  tv_t = std::stod(time_x_y_z_w_vector[0]);
  rv_w = std::stod(time_x_y_z_w_vector[4]);
  rv_x = std::stod(time_x_y_z_w_vector[1]);
  rv_y = std::stod(time_x_y_z_w_vector[2]);
  rv_z = std::stod(time_x_y_z_w_vector[3]);
  rv_tmp << rv_x, rv_y, rv_z, rv_w;

  ///pdr computer record rv data/ rv timestamp
  this->rv_raw_data_current_timestamp = tv_t;
  this->rv_raw_data_current = rv_tmp;
}

void PdrComputer::OnNewRv(double t_ns, double x, double y, double z, double w) {
  this->rv_raw_data_current(0) = x;
  this->rv_raw_data_current(1) = y;
  this->rv_raw_data_current(2) = z;
  this->rv_raw_data_current(3) = w;
  this->rv_raw_data_current_timestamp = t_ns;
}

PdrResult PdrComputer::GetPdrVelocity(const double current_timestamp) {
  PdrResult pdrResult;
  pdrResult.time_stamp = current_timestamp;
  //TODO check current_timestamp

  if (this->acc_raw_data_timestamp_window.size()
      < this->accelerometer_data_length) {
    pdrResult.result_type = RESULT_TYPE_INIT;
    return pdrResult;
  }

  int32_t step_num = this->ComputerStep();

  this->MotionStatusCheck();
  double stride_length;
  if (step_num >= 1) {
    stride_length = this->ComputeStrideLength();
    pdrResult.motion_status = MOTION_STATUS_STEP_DETECTED;
    pdrResult.result_type = RESULT_TYPE_NORMAL;
    pdrResult.velocity2d = this->velocity2d;
  } else {
    pdrResult.result_type = RESULT_TYPE_NORMAL;
    pdrResult.motion_status = this->motion_status;
    pdrResult.velocity2d = this->velocity2d;
  }
  if (pdrResult.motion_status == MOTION_STATUS_STAND
      || pdrResult.motion_status == MOTION_STATUS_UNKOWN)
    pdrResult.velocity2d = 0;

  return pdrResult;
}


PdrResult PdrComputer::GetPdrResult(const double current_timestamp) {
  PdrResult pdrResult;
  pdrResult.time_stamp = current_timestamp;
  //TODO check current_timestamp

  if (this->acc_raw_data_timestamp_window.size()
      < this->accelerometer_data_length) {
    pdrResult.result_type = RESULT_TYPE_INIT;
    return pdrResult;
  }

  int32_t step_num = this->ComputerStep();
  //if(step_num>0)
  //std::cout<<"step_num "<<step_num <<std::endl;
  double yaw;
  double stride_length;

  this->MotionStatusCheck();

  if (step_num >= 1) {
    yaw = this->ComputeHeadings(this->rv_raw_data_current);

    stride_length = this->ComputeStrideLength();
    pdrResult =
        this->ComputeRelPositions(current_timestamp, stride_length, yaw);
    //pdrResult.motion_status=MOVE;
    pdrResult.motion_status = MOTION_STATUS_STEP_DETECTED;
    pdrResult.result_type = RESULT_TYPE_NORMAL;
    pdrResult.velocity2d = this->velocity2d;
  } else {
    pdrResult.result_type = RESULT_TYPE_NORMAL;
    pdrResult.motion_status = this->motion_status;
    pdrResult.velocity2d = this->velocity2d;
  }
  if (pdrResult.motion_status == MOTION_STATUS_STAND
      || pdrResult.motion_status == MOTION_STATUS_UNKOWN)
    pdrResult.velocity2d = 0;

  return pdrResult;
}

}  // namespace pdr
