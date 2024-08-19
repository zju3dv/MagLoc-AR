#ifndef SRC_PREDICT_IMUPREDICTOR_H_
#define SRC_PREDICT_IMUPREDICTOR_H_

#include <glog/logging.h>

#include <Eigen/Core>
#include <vector>

namespace SenseSLAM {

template <typename Scalar = float, int measurement_dim = 3>
class IMUPredictor {
  /// 3d measurements, such as gyro data
  std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > m_measures_3d;
  /// timestamps of each measurement
  std::vector<double> m_timestamps;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m_coff;

 public:
  void PushHistoryData(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &data,
                       const double &t) {
    m_measures_3d.emplace_back(data);
    m_timestamps.emplace_back(t);
  }

  void Clear() {
    m_measures_3d.clear();
    m_timestamps.clear();
  }

  /// generate a polynomial
  /// now it is a 2-order polynomial
  Eigen::Matrix<Scalar, 3, 1> GeneratePoly(const double &t) {
    return Eigen::Matrix<Scalar, 3, 1>(1.0, t, t * t);
  }

  /// solve A * x = b
  /// A [n,3]
  /// x [3,m]
  /// b [n,m]
  /// m is dim of measurement, for gyroscope, it is 3
  void SolveLS() {
    size_t data_len = m_measures_3d.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A(data_len, 3);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> b(data_len,
                                                            measurement_dim);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &x = m_coff;
    x = Eigen::Matrix<Scalar, 3, measurement_dim>::Zero();

    for (size_t i = 0; i < data_len; ++i) {
      A.row(i) = GeneratePoly(m_timestamps[i]);
      b.row(i) = m_measures_3d[i];
    }
    for (size_t dim = 0; dim < measurement_dim; ++dim) {
      x.col(dim) = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                       .solve(b.col(dim));
    }
    //        VLOG(3) << (A * m_coff - b).transpose();
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Predict(const double &t) {
    return GeneratePoly(t).transpose() * m_coff;
  }
};
}  // namespace SenseSLAM
#endif  // SRC_PREDICT_IMUPREDICTOR_H_
