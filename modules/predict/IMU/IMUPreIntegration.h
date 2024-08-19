#ifndef SRC_IMU_IMUPREINTEGRATION_H_
#define SRC_IMU_IMUPREINTEGRATION_H_

#include <Eigen/Dense>
#include <vector>

#include "common/DataHelp.h"

namespace SenseSLAM {

namespace SenseVIO {
class IMUPreIntegration {
 public:
  IMUPreIntegration() {
    jacobian.setIdentity();
    covariance.setZero();
    sum_dt = 0.0;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    valid_flag = true;
    just_propagate_bias = false;
  }

  IMUPreIntegration(const Eigen::Vector3d &_acc_0,
                    const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba,
                    const Eigen::Vector3d &_linearized_bw)
      : acc_0{_acc_0},
        gyr_0{_gyr_0},
        acc_i{_acc_0},
        gyr_i{_gyr_0},
        linearized_ba{_linearized_ba},
        linearized_bw{_linearized_bw},
        jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
        covariance{Eigen::Matrix<double, 15, 15>::Zero()},
        sum_dt{0.0},
        delta_p{Eigen::Vector3d::Zero()},
        delta_q{Eigen::Quaterniond::Identity()},
        delta_v{Eigen::Vector3d::Zero()} {
    valid_flag = true;
    just_propagate_bias = false;
  }

  void SetInit(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
               const Eigen::Vector3d &_linearized_ba,
               const Eigen::Vector3d &_linearized_bw, const double &start_);
  void SetInvalidFlag() {
    Clear();
    valid_flag = false;
  }

  void SetJustPropagateBias() { just_propagate_bias = true; }
  void Clear();

  void PreIntegrate(const std::vector<STSLAMCommon::IMUData> &IMUDatas);
  void PreIntegrate(const IMUPreIntegration &preIntegration);

  void PreIntegrate(double dt, const Eigen::Vector3d &acc,
                    const Eigen::Vector3d &gyr);

  void RePreIntegrate(const Eigen::Vector3d &_ba, const Eigen::Vector3d &_bw);

  void Propagate(double dt, const Eigen::Vector3d &_acc_j,
                 const Eigen::Vector3d &_gyr_j);

  void Propagete_Bias(double start_, double dt, int num);

  void midPointIntegration(
      double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
      const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
      const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
      const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
      const Eigen::Vector3d &linearized_bw,
      Eigen::Vector3d &result_delta_p,        // NOLINT
      Eigen::Quaterniond &result_delta_q,     // NOLINT
      Eigen::Vector3d &result_delta_v,        // NOLINT
      Eigen::Vector3d &result_linearized_ba,  // NOLINT
      Eigen::Vector3d &result_linearized_bw,  // NOLINT
      bool update_jacobian);                  // NOLINT

  Eigen::Matrix<double, 15, 1> ComputeRes(
      const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
      const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
      const Eigen::Vector3d &Bgi, const Eigen::Vector3d &Pj,
      const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
      const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj);

  void SetGravity(const Eigen::Vector3d &g) { gravity = g; }

 public:
  Eigen::Vector3d acc_0, gyr_0;
  Eigen::Vector3d acc_i, gyr_i, acc_j, gyr_j;
  Eigen::Vector3d linearized_ba, linearized_bw;

  Eigen::Matrix<double, 15, 15> jacobian, covariance;

  double sum_dt;
  double start;
  Eigen::Vector3d delta_p;
  Eigen::Quaterniond delta_q;
  Eigen::Vector3d delta_v;

  std::vector<double> dt_buf;
  std::vector<Eigen::Vector3d> acc_buf;
  std::vector<Eigen::Vector3d> gyr_buf;

  bool just_propagate_bias;
  bool valid_flag;

  Eigen::Vector3d gravity{0, 0, 9.81};
};
}  // namespace SenseVIO
}  // namespace SenseSLAM
#endif  // SRC_IMU_IMUPREINTEGRATION_H_
