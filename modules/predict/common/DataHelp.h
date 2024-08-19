#ifndef SENSESLAM2_SRC_UTILITY_DATAHELP_H_
#define SENSESLAM2_SRC_UTILITY_DATAHELP_H_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace STSLAMCommon {
struct IMUData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3f acc;
    Eigen::Vector3f gyr;
    double t;

    IMUData() = default;

    IMUData(const Eigen::Vector3f &_acc, const Eigen::Vector3f &_gyr, double _t) :
        acc(_acc), gyr(_gyr), t(_t) {}
};

struct IMUAttitude {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3f g;  ///< gravity
    Eigen::Quaternionf q;  ///< orientation
    double t = -1.0;
};

struct CameraCalibration {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool fisheye = false;

    double fx, fy;
    double cx, cy;
    double distort[8];
    int height = 0;
    int width = 0;
    Eigen::Matrix3f r_ic;
    Eigen::Vector3f t_ic;

    std::string PrintToString() const {
        std::ostringstream strm;
        char kEndl = '\n';
        strm << "fisheye: " << std::boolalpha << fisheye << kEndl
             << "[fx fy cx cy] = " << Eigen::RowVector4d(fx, fy, cx, cy) << kEndl
             << "distortion = " << Eigen::Map<const Eigen::Matrix<double, 1, 8>>(distort) << kEndl
             << "[width height] = " << width << " " << height << kEndl
             << "r_ic = " << r_ic.format(Eigen::IOFormat(Eigen::StreamPrecision, 0, ",", ";"))
             << kEndl << "t_ic = " << t_ic.transpose();
        return strm.str();
    }
};

struct IMUInstrinsic {
    Eigen::Vector3f acc_bias;
    Eigen::Vector3f gyo_bias;

    IMUInstrinsic() {
        acc_bias.setZero();
        gyo_bias.setZero();
    }
};

struct VIOState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int frame_id;
    double timestamp;
    Eigen::Vector3f t_wc;
    Eigen::Quaternionf r_wc;
    Eigen::Vector3f v;
    Eigen::Vector3f ba;
    Eigen::Vector3f bw;
    IMUData imu_data;

    VIOState() : frame_id(-1), timestamp(-1) {}
};

std::ostream& operator<<(std::ostream& os, const IMUData& obj);
std::ostream& operator<<(std::ostream& os, const IMUAttitude& obj);
std::ostream& operator<<(std::ostream& os, const CameraCalibration& obj);
std::ostream& operator<<(std::ostream& os, const IMUInstrinsic& obj);
}   // namespace STSLAMCommon
#endif //SENSESLAM2_SRC_UTILITY_DATAHELP_H_
