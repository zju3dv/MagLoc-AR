#ifndef STSLAM_UTILITY_H_
#define STSLAM_UTILITY_H_

#include <Eigen/Core>

#define STSLAM_SAFE_DELETE(p)\
do{\
    delete (p);\
    (p) = NULL;\
}while(0)

#define STSLAM_SAFE_DELETE_ARRAY(p)\
do{\
    delete[] (p);\
    (p) = NULL;\
}while(0)


///< 功能与x##y相同, 支持x, y为宏
#define STSLAM_PASTER_(x, y) x ## y
#define STSLAM_PASTER(x, y) STSLAM_PASTER_(x, y)

///< C字符串是否为空串
#define STSLAM_CSTR_IS_EMPTY(cstr) ((nullptr == (cstr)) || ('\0' == *(cstr)))

///< C++类, 禁止拷贝构造和赋值重载
#define STSLAM_DISABLE_COPY(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName &operator=(const ClassName&) = delete

#define SLIGHT_ANGLE    0.008726646                          // 0.5°

namespace STSLAMCommon {

template<typename Derived>
static Eigen::Quaternion<typename Derived::Scalar>
deltaQ(const Eigen::MatrixBase<Derived> &theta) {
    typedef typename Derived::Scalar Scalar_t;
    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    double norm = half_theta.norm();
    /// 0.5°
    if (norm > SLIGHT_ANGLE) {
        dq.w() = static_cast<Scalar_t>(cos(norm));
        dq.x() = sin(norm) * half_theta.x() / norm;
        dq.y() = sin(norm) * half_theta.y() / norm;
        dq.z() = sin(norm) * half_theta.z() / norm;
    } else {
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        dq.normalized();
    }
    return dq;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> Reorthogonalize(
    const Eigen::MatrixBase<Derived> &rotationMatrix
) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> result;

    result.col(0) = rotationMatrix.col(0).normalized();
    result.col(1) = rotationMatrix.col(1).normalized();
    result.col(2) = result.col(0).cross(result.col(1));
    result.col(2).normalize();
    result.col(0) = result.col(1).cross(result.col(2));
    result.col(0).normalize();
    return result;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
static SkewSymmetric(const Eigen::MatrixBase<Derived> &q) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
        q(2), typename Derived::Scalar(0), -q(0),
        -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar>
positify(const Eigen::QuaternionBase<Derived> &q) {
    return q;
}

template<typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4>
Qleft(const Eigen::QuaternionBase<Derived> &q) {
    Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec();
    ans.template block<3, 3>(1, 1) =
        qq.w() *
            Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
            SkewSymmetric(qq.vec());
    return ans;
}

template<typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4>
Qright(const Eigen::QuaternionBase<Derived> &p) {
    Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(1, 0) = pp.vec();
    ans.template block<3, 3>(1, 1) =
        pp.w() *
            Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
            SkewSymmetric(pp.vec());
    return ans;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> R2ypr(const Eigen::MatrixBase<Derived> &R) {
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Matrix<Scalar_t, 3, 1> n = R.col(0);
    Eigen::Matrix<Scalar_t, 3, 1> o = R.col(1);
    Eigen::Matrix<Scalar_t, 3, 1> a = R.col(2);

    Eigen::Matrix<Scalar_t, 3, 1> ypr(3);
    Scalar_t y = atan2(n(1), n(0));
    Scalar_t p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    Scalar_t r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    if (y < -1)
        y += 2*M_PI;
    if (p < -1)
        p += 2*M_PI;
    if (r < -1)
        r += 2*M_PI;

    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr) {
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0) / 180.0 * M_PI;
    Scalar_t p = ypr(1) / 180.0 * M_PI;
    Scalar_t r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
}
}  // namespace STSLAMCommon

#endif  //  STSLAM_UTILITY_H_