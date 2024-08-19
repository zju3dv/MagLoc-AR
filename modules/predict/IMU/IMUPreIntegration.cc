#include "IMUPreIntegration.h"
#include <glog/logging.h>
#include "common/Utility.h"

namespace SenseSLAM {
namespace SenseVIO {
enum StateOrder { O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12 };
double ACC_SATURATION = 100.0;
double GYR_SATURATION = 20;
double ACC_N = 0.25;
double ACC_W = 1e-3;
double GYR_N = 0.05;
double GYR_W = 3.9967e-5;

void IMUPreIntegration::SetInit(const Eigen::Vector3d &_acc_0,
                                const Eigen::Vector3d &_gyr_0,
                                const Eigen::Vector3d &_linearized_ba,
                                const Eigen::Vector3d &_linearized_bw,
                                const double &start_) {
  acc_i = acc_0 = _acc_0;
  gyr_i = gyr_0 = _gyr_0;
  linearized_ba = _linearized_ba;
  linearized_bw = _linearized_bw;
  start = start_;
  Clear();
}

void IMUPreIntegration::Clear() {
  jacobian.setIdentity();
  covariance.setZero();
  sum_dt = 0.0;
  delta_p.setZero();
  delta_q.setIdentity();
  delta_v.setZero();

  dt_buf.clear();
  acc_buf.clear();
  gyr_buf.clear();
  valid_flag = true;
  just_propagate_bias = false;
}

void IMUPreIntegration::PreIntegrate(
    const std::vector<STSLAMCommon::IMUData> &IMUDatas) {
  double time0 = start;
  double dt = 0.0;
  if (just_propagate_bias) {
    LOG(FATAL) << "should not enter this";
    //        Propagete_Bias(start, IMU::IMU_DT, (IMUDatas.back().t - start) /
    //        IMU::IMU_DT);
    return;
  } else {
    for (auto &it : IMUDatas) {
      dt = it.t - time0;
      time0 = it.t;
      dt_buf.push_back(dt);
      acc_buf.push_back(it.acc.cast<double>());
      gyr_buf.push_back(it.gyr.cast<double>());
      Propagate(dt, it.acc.cast<double>(), it.gyr.cast<double>());
    }
  }
}

void IMUPreIntegration::PreIntegrate(const IMUPreIntegration &preIntegration) {
  if (just_propagate_bias) {
    //        Propagete_Bias(start, IMU::IMU_DT, (sum_dt +
    //        preIntegration.sum_dt) / IMU::IMU_DT);
    LOG(FATAL) << "should not enter this";
    return;
  } else {
    for (int i = 0; i < preIntegration.dt_buf.size(); ++i) {
      dt_buf.push_back(preIntegration.dt_buf[i]);
      acc_buf.push_back(preIntegration.acc_buf[i]);
      gyr_buf.push_back(preIntegration.gyr_buf[i]);
      Propagate(dt_buf.back(), acc_buf.back(), gyr_buf.back());
    }
  }
}

void IMUPreIntegration::PreIntegrate(double dt, const Eigen::Vector3d &acc,
                                     const Eigen::Vector3d &gyr) {
  if (just_propagate_bias) {
    //        Propagete_Bias(start, IMU::IMU_DT, 1);
    LOG(FATAL) << "should not enter this";
    return;
  } else {
    dt_buf.push_back(dt);
    acc_buf.push_back(acc);
    gyr_buf.push_back(gyr);
    Propagate(dt, acc, gyr);
  }
}

void IMUPreIntegration::RePreIntegrate(const Eigen::Vector3d &_ba,
                                       const Eigen::Vector3d &_bw) {
  if (just_propagate_bias) {
    linearized_ba = _ba;
    linearized_bw = _bw;
    return;
  }
  sum_dt = 0.0;
  acc_i = acc_0;
  gyr_i = gyr_0;
  delta_p.setZero();
  delta_q.setIdentity();
  delta_v.setZero();
  linearized_ba = _ba;
  linearized_bw = _bw;
  jacobian.setIdentity();
  covariance.setZero();
  for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
    Propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
}

void IMUPreIntegration::Propagate(double dt, const Eigen::Vector3d &_acc_j,
                                  const Eigen::Vector3d &_gyr_j) {
  acc_j = _acc_j;
  gyr_j = _gyr_j;
  Eigen::Vector3d result_delta_p;
  Eigen::Quaterniond result_delta_q;
  Eigen::Vector3d result_delta_v;
  Eigen::Vector3d result_linearized_ba;
  Eigen::Vector3d result_linearized_bw;

  midPointIntegration(dt, acc_i, gyr_i, acc_j, gyr_j, delta_p, delta_q, delta_v,
                      linearized_ba, linearized_bw, result_delta_p,
                      result_delta_q, result_delta_v, result_linearized_ba,
                      result_linearized_bw, true);

  // checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
  //                    linearized_ba, linearized_bw);

  delta_p = result_delta_p;
  delta_q = result_delta_q;
  delta_v = result_delta_v;
  linearized_ba = result_linearized_ba;
  linearized_bw = result_linearized_bw;
  delta_q.normalize();
  sum_dt += dt;
  acc_i = acc_j;
  gyr_i = gyr_j;
}

void IMUPreIntegration::Propagete_Bias(double start_, double dt, int num) {
  LOG(FATAL) << "should not enter this";
  jacobian.setIdentity();
  covariance.setZero();
  start = start_;
  sum_dt = num * dt;
  delta_p.setZero();
  delta_q.setIdentity();
  delta_v.setZero();
  dt_buf.clear();
  acc_buf.clear();
  gyr_buf.clear();

  covariance.block<3, 3>(9, 9) =
      num * dt * dt * ACC_W * ACC_W * Eigen::Matrix3d::Identity();
  covariance.block<3, 3>(12, 12) =
      num * dt * dt * GYR_W * GYR_W * Eigen::Matrix3d::Identity();
}

void IMUPreIntegration::midPointIntegration(
    double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
    const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
    const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
    const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
    const Eigen::Vector3d &linearized_bw, Eigen::Vector3d &result_delta_p,
    Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
    Eigen::Vector3d &result_linearized_ba,
    Eigen::Vector3d &result_linearized_bw, bool update_jacobian) {
  const double _dt2 = _dt * _dt, _dt3 = _dt * _dt2;

  Eigen::Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
  Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bw;
  result_delta_q = delta_q * (STSLAMCommon::deltaQ(un_gyr * _dt)).normalized();
  Eigen::Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  result_delta_p = delta_p + delta_v * _dt + 0.5 * _dt2 * un_acc;
  result_delta_v = delta_v + un_acc * _dt;
  result_linearized_ba = linearized_ba;
  result_linearized_bw = linearized_bw;

  if (update_jacobian) {
    Eigen::Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bw;
    Eigen::Vector3d a_0_x = _acc_0 - linearized_ba;
    Eigen::Vector3d a_1_x = _acc_1 - linearized_ba;
    Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

    R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;

    R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1),
        a_0_x(0), 0;

    R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1),
        a_1_x(0), 0;

    const Eigen::Matrix3d delta_q_R = delta_q.toRotationMatrix(),
                          result_delta_q_R = result_delta_q.toRotationMatrix();

    Eigen::Matrix<double, 9, 15> F = Eigen::Matrix<double, 9, 15>::Zero();
    F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(0, 3) =
        (-0.25 * _dt2) *
        (delta_q_R * R_a_0_x + result_delta_q_R * R_a_1_x *
                                   (Eigen::Matrix3d::Identity() - _dt * R_w_x));
    F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3, 3) * _dt;
    F.block<3, 3>(0, 9) = -0.25 * _dt2 * (delta_q_R + result_delta_q_R);
    F.block<3, 3>(0, 12) = 0.25 * _dt3 * (result_delta_q_R * R_a_1_x);

    F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * _dt;
    F.block<3, 3>(3, 12) = -_dt * Eigen::MatrixXd::Identity(3, 3);

    F.block<3, 3>(6, 3) =
        -0.5 * _dt *
        (delta_q_R * R_a_0_x + result_delta_q_R * R_a_1_x *
                                   (Eigen::Matrix3d::Identity() - R_w_x * _dt));
    F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 9) = -0.5 * _dt * (delta_q_R + result_delta_q_R);
    F.block<3, 3>(6, 12) = 0.5 * _dt2 * (result_delta_q_R * R_a_1_x);

    Eigen::Matrix<double, 9, 12> V = Eigen::Matrix<double, 9, 12>::Zero();
    V.block<3, 3>(0, 0) = 0.25 * _dt2 * delta_q_R;
    V.block<3, 3>(0, 3) = -0.25 * 0.5 * _dt3 * (result_delta_q_R * R_a_1_x);
    V.block<3, 3>(0, 6) = 0.25 * _dt2 * result_delta_q_R;
    V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);

    V.block<3, 3>(3, 3) = 0.5 * _dt * Eigen::MatrixXd::Identity(3, 3);
    V.block<3, 3>(3, 9) = 0.5 * _dt * Eigen::MatrixXd::Identity(3, 3);

    V.block<3, 3>(6, 0) = 0.5 * _dt * delta_q_R;
    V.block<3, 3>(6, 3) = -0.25 * _dt2 * (result_delta_q_R * R_a_1_x);
    V.block<3, 3>(6, 6) = 0.5 * _dt * result_delta_q_R;
    V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);

    // NOTE: Here I use the fact that noise is diagonal to do optpimziation.
    //  V * noise * V.transpose()
    //  = (V * sqrt(noise)) * (V * sqrt(noise)).transpose()
    double acc_n = ACC_N, gyr_n = GYR_N, acc_w = ACC_W, gyr_w = GYR_W;

    if (fabs(_acc_0(0)) > ACC_SATURATION || fabs(_acc_0(1)) > ACC_SATURATION ||
        fabs(_acc_0(2)) > ACC_SATURATION || fabs(_acc_1(0)) > ACC_SATURATION ||
        fabs(_acc_1(1)) > ACC_SATURATION || fabs(_acc_1(2)) > ACC_SATURATION) {
      acc_n *= 1.0e2;
    }
    if (fabs(_gyr_0(0)) > GYR_SATURATION || fabs(_gyr_0(1)) > GYR_SATURATION ||
        fabs(_gyr_0(2)) > GYR_SATURATION || fabs(_gyr_1(0)) > GYR_SATURATION ||
        fabs(_gyr_1(1)) > GYR_SATURATION || fabs(_gyr_1(2)) > GYR_SATURATION) {
      gyr_n *= 1.0e2;
    }

    Eigen::Matrix<double, 18, 1> noise_;
    noise_ << acc_n * Eigen::Vector3d::Ones(), gyr_n * Eigen::Vector3d::Ones(),
        acc_n * Eigen::Vector3d::Ones(), gyr_n * Eigen::Vector3d::Ones(),
        acc_w * Eigen::Vector3d::Ones(), gyr_w * Eigen::Vector3d::Ones();

    // Jacobian: jacobian = F * jacobian
    // We use the structure of F to expand F * jacobian, be carefully to not
    // broke the data dependencies.
    //
    // structure of F:
    // clang-format off
        // |    | 0 | 3 | 6 | 9 | 12  |
        // |----+---+---+---+---+-----|
        // |  0 | 1 | x | x | x | x   |
        // |----+---+---+---+---+-----|
        // |  3 |   | x |   |   | -dt |
        // |----+---+---+---+---+-----|
        // |  6 |   | x | 1 | x | x   |
        // |----+---+---+---+---+-----|
        // |  9 |   |   |   | 1 |     |
        // |----+---+---+---+---+-----|
        // | 12 |   |   |   |   | 1   |
        // |----+---+---+---+---+-----|
    // clang-format on

    jacobian.topRows<3>() +=
        F.topRightCorner<3, 12>() * jacobian.bottomRows<12>();
    const Eigen::Matrix<double, 3, 15> old_jacobian_row3 =
        jacobian.middleRows<3>(3);
    jacobian.middleRows<3>(3) = F.block<3, 3>(3, 3) * old_jacobian_row3 -
                                _dt * jacobian.bottomRows<3>();
    jacobian.middleRows<3>(6) += F.block<3, 3>(6, 3) * old_jacobian_row3 +
                                 F.block<3, 6>(6, 9) * jacobian.bottomRows<6>();

    // Covariance
    // We use the structure of F, V to do optimization.
    //
    // structure of V:
    // clang-format off
        // |    | 0 | 3 | 6 | 9 | 12 | 15 |
        // |----+---+---+---+---+----+----|
        // |  0 | x | x | x | x |    |    |
        // |----+---+---+---+---+----+----|
        // |  3 |   | x |   | x |    |    |
        // |----+---+---+---+---+----+----|
        // |  6 | x | x | x | x |    |    |
        // |----+---+---+---+---+----+----|
        // |  9 |   |   |   |   | dt |    |
        // |----+---+---+---+---+----+----|
        // | 12 |   |   |   |   |    | dt |
        // |----+---+---+---+---+----+----|
    // clang-format on
    //      12 6
    // V = [U  O    9
    //      O dt]   6
    // V * diag(alpha^2, beta^2) * V.transpose()
    //   = [U * diag(alpha^2) * U.transpose()              O
    //                    O                         dt^2 * diag(beta^2)]

    // covariance = F * covariance
    covariance.topRows<3>() +=
        F.topRightCorner<3, 12>() * covariance.bottomRows<12>();
    const Eigen::Matrix<double, 3, 15> old_covariance_row3 =
        covariance.middleRows<3>(3);
    covariance.middleRows<3>(3) = F.block<3, 3>(3, 3) * old_covariance_row3 -
                                  _dt * covariance.bottomRows<3>();
    covariance.middleRows<3>(6) +=
        F.block<3, 3>(6, 3) * old_covariance_row3 +
        F.block<3, 6>(6, 9) * covariance.bottomRows<6>();

    // covariance = covariance * F.transpose()
    covariance.leftCols<3>() +=
        covariance.rightCols<12>() * F.topRightCorner<3, 12>().transpose();
    const Eigen::Matrix<double, 15, 3> old_covariance_col3 =
        covariance.middleCols<3>(3);
    covariance.middleCols<3>(3) =
        old_covariance_col3 * F.block<3, 3>(3, 3).transpose() -
        _dt * covariance.rightCols<3>();
    covariance.middleCols<3>(6) +=
        old_covariance_col3 * F.block<3, 3>(6, 3).transpose() +
        covariance.rightCols<6>() * F.block<3, 6>(6, 9).transpose();

    // covariance += V * noise^2 * V^T
    V = V * noise_.head<12>().asDiagonal();
    covariance.topLeftCorner<9, 9>() += V * V.transpose();
    covariance.bottomRightCorner<6, 6>().diagonal() +=
        _dt2 * noise_.tail<6>().cwiseProduct(noise_.tail<6>());
  }
}

Eigen::Matrix<double, 15, 1> IMUPreIntegration::ComputeRes(
    const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
    const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
    const Eigen::Vector3d &Bwi, const Eigen::Vector3d &Pj,
    const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
    const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bwj) {
  Eigen::Matrix<double, 15, 1> residuals;
  residuals.setZero();

  if (!just_propagate_bias) {
    Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbw = jacobian.block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbw = jacobian.block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbw = jacobian.block<3, 3>(O_V, O_BG);

    Eigen::Vector3d dba = Bai - linearized_ba;
    Eigen::Vector3d dbw = Bwi - linearized_bw;

    Eigen::Quaterniond corrected_delta_q =
        delta_q * STSLAMCommon::deltaQ(dq_dbw * dbw);
    Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbw * dbw;
    Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbw * dbw;

    residuals.block<3, 1>(O_P, 0) =
        Qi.inverse() *
            (0.5 * gravity * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) -
        corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) =
        2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) =
        Qi.inverse() * (gravity * sum_dt + Vj - Vi) - corrected_delta_v;
  }
  residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
  residuals.block<3, 1>(O_BG, 0) = Bwj - Bwi;
  return residuals;
}

}  // namespace SenseVIO
}  // namespace SenseSLAM
