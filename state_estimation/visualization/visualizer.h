/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-01-18 10:52:52
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-12 14:13:30
 */
#ifndef STATE_ESTIMATION_VISUALIZATION_VISUALIZER_H_
#define STATE_ESTIMATION_VISUALIZATION_VISUALIZER_H_

#include <pangolin/pangolin.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace state_estimation {

namespace visualization {

struct VecXd {
  Eigen::VectorXd vec_ = Eigen::Vector3d::Zero();
};

inline std::ostream &operator<<(std::ostream &out, const VecXd &r) {
  int N = r.vec_.size();
  out.setf(std::ios::fixed);
  out << '=' << " [";
  for (int i = 0; i < N - 1; i++) {
    out << std::setprecision(2) << r.vec_(i) << ", ";
  }
  out << r.vec_(N - 1) << "]";
  return out;
}

inline std::istream &operator>>(std::istream &in, VecXd &r) {
  return in;
}

class SimpleVisualizer {
 public:
  explicit SimpleVisualizer(int width = 752,
                            int height = 480,
                            std::string window_title = "camera_pose")
      : window_width_(width),
        window_height_(height),
        window_title_(window_title) {}

  ~SimpleVisualizer() {}

  void InitDraw(double ex, double ey, double ez, double lx, double ly, double lz, double ux, double uy, double uz);

  void ActiveAllView();

  void DrawCubeTest();

  void DrawCamWithPose(const Eigen::Vector3d &pos,
                       const Eigen::Quaterniond &quat);

  void DrawTraj(const std::vector<Eigen::Vector3d> &traj,
                double color_r,
                double color_g,
                double color_b);

  void DrawParticles(const std::vector<Eigen::Vector3d> &particles);

  void DrawParticles(const std::vector<Eigen::Vector3d> &particles, std::vector<double> &weights);

  void DrawCam(const float scale = 1.0);

  void DrawCoordinate();

  void DrawReferenceGridXY(double x_min, double x_max,
                           double y_min, double y_max,
                           double grid_interval);

  void DisplayData(const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat);

  void RegisterUICallback();

 private:
  pangolin::OpenGlRenderState s_cam_;
  pangolin::View d_cam_, d_img_, d_track_;
  pangolin::GlTexture imageTexture_, trackTexture_;
  pangolin::DataLog pose_log_;

  std::vector<pangolin::Var<bool>> ui_set_;
  std::vector<pangolin::Var<VecXd>> data_set_;
  bool camera_visible_ = true;
  bool traj_visible_ = true;
  bool coordinate_visible_ = true;
  bool particles_visible_ = true;
  bool reference_grid_visible_ = true;

  int window_width_;
  int window_height_;
  std::string window_title_;
};

}  // namespace visualization

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_VISUALIZATION_VISUALIZER_H_
