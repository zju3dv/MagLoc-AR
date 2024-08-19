/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-03-09 16:03:03
 * @Last Modified by: xuehua
 * @Last Modified time: 2022-03-11 11:18:57
 */
#ifndef STATE_ESTIMATION_SAMPLER_UNIFORM_3D_ORIENTATION_SAMPLER_H_
#define STATE_ESTIMATION_SAMPLER_UNIFORM_3D_ORIENTATION_SAMPLER_H_

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <ctime>

#include "util/misc.h"

namespace state_estimation {

namespace sampler {

class Uniform3DRotationAxisSampler {
 public:
  Eigen::Vector3d Sample(void) {
    // reference: https://marc-b-reynolds.github.io/distribution/2016/11/28/Uniform.html
    // Uniform points on the 3D unit sphere.
    std::uniform_real_distribution<double> theta_distribution(-M_PI, M_PI);
    std::uniform_real_distribution<double> a_distribution(0.0, 1.0);
    double a, theta, x, y, xx, yy, zz;
    a = a_distribution(this->generator_);
    theta = theta_distribution(this->generator_);
    x = std::sqrt(a) * std::cos(theta);
    y = std::sqrt(a) * std::sin(theta);
    xx = 2 * x * std::sqrt(1 - (x * x + y * y));
    yy = 2 * y * std::sqrt(1 - (x * x + y * y));
    zz = 1 - 2 * (x * x + y * y);
    Eigen::Vector3d rotation_vector_unit(xx, yy, zz);
    return rotation_vector_unit;
  }

  void Seed(int random_seed) {
    this->generator_.seed(static_cast<uint64_t>(random_seed));
  }

  Uniform3DRotationAxisSampler(void) {
    struct timespec tn;
    clock_gettime(CLOCK_REALTIME, &tn);
    this->generator_.seed(static_cast<uint64_t>(tn.tv_nsec));
  }

  ~Uniform3DRotationAxisSampler() {}

 private:
  std::default_random_engine generator_;
};

class Uniform3DOrientationSampler {
 public:
  Eigen::Quaterniond Sample(void) {
    // reference: https://marc-b-reynolds.github.io/distribution/2017/01/27/UniformRot.html
    std::uniform_real_distribution<double> theta_distribution(-M_PI, M_PI);
    std::uniform_real_distribution<double> a_distribution(0.0, 1.0);
    double a0, a1, theta0, theta1, x0, y0, x1, y1, d0, d1, s;
    a0 = a_distribution(this->generator_);
    theta0 = theta_distribution(this->generator_);
    x0 = std::sqrt(a0) * std::cos(theta0);
    y0 = std::sqrt(a0) * std::sin(theta0);
    a1 = a_distribution(this->generator_);
    theta1 = theta_distribution(this->generator_);
    x1 = std::sqrt(a1) * std::cos(theta1);
    y1 = std::sqrt(a1) * std::sin(theta1);

    d0 = x0 * x0 + y0 * y0;
    d1 = x1 * x1 + y1 * y1;

    s = std::sqrt((1 - d0) / d1);

    Eigen::Quaterniond q(x0, y0, x1 * s, y1 * s);
    return q;
  }

  Eigen::Quaterniond Sample1(void) {
    // uniformly sample orientations in my way.
    // uniformly sample rotation vectors within a 3D ball which has a radius of pi.
    // reference: https://marc-b-reynolds.github.io/distribution/2016/11/28/Uniform.html
    std::uniform_real_distribution<double> theta_distribution(-M_PI, M_PI);
    std::uniform_real_distribution<double> a_distribution(0.0, 1.0);
    double a, theta, x, y, xx, yy, zz;
    a = a_distribution(this->generator_);
    theta = theta_distribution(this->generator_);
    x = std::sqrt(a) * std::cos(theta);
    y = std::sqrt(a) * std::sin(theta);
    xx = 2 * x * std::sqrt(1 - (x * x + y * y));
    yy = 2 * y * std::sqrt(1 - (x * x + y * y));
    zz = 1 - 2 * (x * x + y * y);
    Eigen::Vector3d rotation_vector_unit(xx, yy, zz);

    std::uniform_real_distribution<double> omega_distribution(0, M_PI);
    double omega = omega_distribution(this->generator_);

    Eigen::AngleAxisd rotation_vector(omega, rotation_vector_unit);
    Eigen::Quaterniond q(rotation_vector);
    return q;
  }

  Eigen::Quaterniond Sample(Eigen::Quaterniond reference_orientation, double max_angular_distance, double* angular_distance = nullptr) {
    // reference: http://planning.cs.uiuc.edu/node198.html
    std::uniform_real_distribution<double> uniform_unit(0.0, 1.0);
    double u1, u2, u3, w, x, y, z;
    Eigen::Quaterniond q_unit;
    Eigen::Quaterniond q_error;
    Eigen::AngleAxisd angleaxis_error;
    double error_angle = max_angular_distance + 1.0;
    double epsilon = 1e-8;
    while (error_angle > max_angular_distance + epsilon) {
      u1 = uniform_unit(this->generator_);
      u2 = uniform_unit(this->generator_);
      u3 = uniform_unit(this->generator_);
      w = std::sqrt(1 - u1) * std::sin(2.0 * M_PI * u2);
      x = std::sqrt(1 - u1) * std::cos(2.0 * M_PI * u2);
      y = std::sqrt(u1) * std::sin(2.0 * M_PI * u3);
      z = std::sqrt(u1) * std::cos(2.0 * M_PI * u3);
      q_unit = Eigen::Quaterniond(w, x, y, z);
      q_error = q_unit * reference_orientation.conjugate();
      angleaxis_error = Eigen::AngleAxisd(q_error);
      error_angle = angleaxis_error.angle();
    }
    if (angular_distance) {
      *angular_distance = error_angle;
    }
    return q_unit;
  }

  Eigen::Quaterniond SampleByUniformOnAngleAxis(Eigen::Quaterniond reference_orientation, double max_diff_x, double max_diff_y, double max_diff_z) {
    assert(max_diff_x >= 0.0);
    assert(max_diff_y >= 0.0);
    assert(max_diff_z >= 0.0);
    std::uniform_real_distribution<double> uniform_distribution_x(-max_diff_x, max_diff_x);
    std::uniform_real_distribution<double> uniform_distribution_y(-max_diff_y, max_diff_y);
    std::uniform_real_distribution<double> uniform_distribution_z(-max_diff_z, max_diff_z);
    Eigen::Vector3d sample_orientation_jitter_log(uniform_distribution_x(this->generator_),
                                                  uniform_distribution_y(this->generator_),
                                                  uniform_distribution_z(this->generator_));
    return LogVector2Quaternion(sample_orientation_jitter_log) * reference_orientation;
  }

  double CalculateLogProbabilityByUniformOnAngleAxis(Eigen::Quaterniond reference_orientation, double max_diff_x, double max_diff_y, double max_diff_z, Eigen::Quaterniond sample_q) {
    assert(max_diff_x >= 0.0);
    assert(max_diff_y >= 0.0);
    assert(max_diff_z >= 0.0);
    Eigen::Quaterniond q_orientation_jitter = sample_q * reference_orientation.conjugate();
    Eigen::Vector3d orientation_jitter_log = Quaternion2LogVector(q_orientation_jitter);
    double log_prob = 0.0;
    if (max_diff_x > 1e-10) {
      if ((orientation_jitter_log(0) >= -max_diff_x) && (orientation_jitter_log(0) <= max_diff_x)) {
        log_prob += 1.0 / (2.0 * max_diff_x);
      } else {
        log_prob += std::log(0.0);
      }
    } else {
      if (std::abs(orientation_jitter_log(0)) <= 1e-10) {
        log_prob += std::log(1.0);
      } else {
        log_prob += std::log(0.0);

      }
    }
    if (max_diff_y > 1e-10) {
      if ((orientation_jitter_log(1) >= -max_diff_y) && (orientation_jitter_log(1) <= max_diff_y)) {
        log_prob += 1.0 / (2.0 * max_diff_y);
      } else {
        log_prob += std::log(0.0);
      }
    } else {
      if (std::abs(orientation_jitter_log(1)) <= 1e-10) {
        log_prob += std::log(1.0);
      } else {
        log_prob += std::log(0.0);

      }
    }
    if (max_diff_z > 1e-10) {
      if ((orientation_jitter_log(2) >= -max_diff_z) && (orientation_jitter_log(2) <= max_diff_z)) {
        log_prob += 1.0 / (2.0 * max_diff_z);
      } else {
        log_prob += std::log(0.0);
      }
    } else {
      if (std::abs(orientation_jitter_log(2)) <= 1e-10) {
        log_prob += std::log(1.0);
      } else {
        log_prob += std::log(0.0);

      }
    }
    return log_prob;
  }

  Eigen::Quaterniond SampleBySequentialUniformOnRotationMatrix(Eigen::Quaterniond reference_orientation, double max_angular_distance) {
    assert(max_angular_distance > 0.0);
    Eigen::Matrix3d reference_rotation_matrix = reference_orientation.normalized().toRotationMatrix();
    Eigen::Vector3d r_theta_phi_x = Cartesian2Spherical(reference_rotation_matrix.row(0));
    std::uniform_real_distribution<double> uniform_distribution_theta(-max_angular_distance, max_angular_distance);
    std::uniform_real_distribution<double> uniform_distribution_phi(-max_angular_distance, max_angular_distance);
    double delta_theta, delta_phi;
    while (true) {
      delta_theta = uniform_distribution_theta(this->generator_);
      delta_phi = uniform_distribution_theta(this->generator_);
      if ((delta_theta * delta_theta + delta_phi * delta_phi) < (max_angular_distance * max_angular_distance)) {
        break;
      }
    }
    Eigen::Vector3d sample_r_theta_phi_x(r_theta_phi_x(0), r_theta_phi_x(1) + delta_theta, r_theta_phi_x(2) + delta_phi);
    Eigen::Vector3d sample_xyz_x = Spherical2Cartesian(sample_r_theta_phi_x);
    Eigen::Vector3d xyz_y = reference_rotation_matrix.row(1);
    double sigma = std::asin(std::sqrt(std::pow(xyz_y.norm(), 2.0) - std::pow(xyz_y.dot(sample_xyz_x), 2.0)) / xyz_y.norm());
    double max_omega = std::sqrt(max_angular_distance * max_angular_distance - sigma * sigma);
    std::uniform_real_distribution<double> uniform_distribution_omege(-max_omega, max_omega);
    double sample_omega = uniform_distribution_omege(this->generator_);
    Eigen::Vector3d xyz_y_projection = xyz_y - (xyz_y.dot(sample_xyz_x) * sample_xyz_x);
    Eigen::Vector3d sample_xyz_y = Eigen::AngleAxisd(sample_omega, sample_xyz_x) * xyz_y_projection;
    assert(std::abs(sample_xyz_y.dot(sample_xyz_x)) < 1e-10);
    Eigen::Matrix3d sample_rotation_matrix;
    sample_rotation_matrix.row(0) = sample_xyz_x;
    sample_rotation_matrix.row(1) = sample_xyz_y;
    sample_rotation_matrix.row(2) = sample_xyz_x.cross(sample_xyz_y);
    return Eigen::Quaterniond(sample_rotation_matrix);
  }

  void Seed(int random_seed) {
    this->generator_.seed(static_cast<uint64_t>(random_seed));
  }

  Uniform3DOrientationSampler(void) {
    struct timespec tn;
    clock_gettime(CLOCK_REALTIME, &tn);
    this->generator_.seed(static_cast<uint64_t>(tn.tv_nsec));
  }

  ~Uniform3DOrientationSampler() {}

 private:
  std::default_random_engine generator_;
};

}  // namespace sampler

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_SAMPLER_UNIFORM_3D_ORIENTATION_SAMPLER_H_
