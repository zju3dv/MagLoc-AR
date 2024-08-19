/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-01-18 14:56:42
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-03-12 15:16:33
 */
#include "visualization/visualizer.h"

namespace state_estimation {

namespace visualization {

void SimpleVisualizer::InitDraw(double ex, double ey, double ez, double lx, double ly, double lz, double ux, double uy, double uz) {
  pangolin::CreateWindowAndBind(
      this->window_title_,
      this->window_width_,
      this->window_height_);

  glEnable(GL_DEPTH_TEST);

  s_cam_ = pangolin::OpenGlRenderState(
      pangolin::ProjectionMatrix(this->window_width_, this->window_height_,
                                 420, 420,
                                 (this->window_width_ / 2.0), (this->window_height_ / 2.0),
                                 0.01, 10000),
      pangolin::ModelViewLookAt(ex, ey, ez, lx, ly, lz, ux, uy, uz));

  int panel_width = this->window_width_ / 4.0;
  int panel_height = this->window_height_ / 4.0;

  // trajectory panel
  d_cam_ = pangolin::CreateDisplay()
               .SetBounds(
                   0.0, 1.0,
                   0.0, 1.0,
                   -static_cast<float>(this->window_width_) /
                       static_cast<float>(this->window_height_))
               .SetHandler(new pangolin::Handler3D(s_cam_));

  // data panel
  pangolin::CreatePanel("data")
      .SetBounds(
          pangolin::Attach::Pix(3.0f * panel_height), 1.0,
          0.0, pangolin::Attach::Pix(panel_width),
          static_cast<float>(this->window_width_) /
              static_cast<float>(this->window_height_));
  this->data_set_.clear();
  pangolin::Var<VecXd> current_position("data.position", VecXd());
  this->data_set_.push_back(current_position);
  pangolin::Var<VecXd> current_attitude("data.euler_angle", VecXd());
  this->data_set_.push_back(current_attitude);

  // ui panel
  pangolin::CreatePanel("ui")
      .SetBounds(pangolin::Attach::Pix(2.0f * panel_height),
                 pangolin::Attach::Pix(3.0f * panel_height),
                 0.0, pangolin::Attach::Pix(panel_width),
                 static_cast<float>(this->window_width_) /
                     static_cast<float>(this->window_height_));

  this->ui_set_.clear();
  pangolin::Var<bool> show_cam("ui.show_cam", true, true);
  this->ui_set_.push_back(show_cam);
  pangolin::Var<bool> show_traj("ui.show_traj", true, true);
  this->ui_set_.push_back(show_traj);
  pangolin::Var<bool> show_coordinate("ui.show_coordinate", true, true);
  this->ui_set_.push_back(show_coordinate);
  pangolin::Var<bool> show_particles("ui.show_particles", true, true);
  this->ui_set_.push_back(show_particles);
  pangolin::Var<bool> show_grid("ui.show_grid", true, true);
  this->ui_set_.push_back(show_grid);
}

void SimpleVisualizer::ActiveAllView() {
  d_cam_.Activate(s_cam_);
}

void SimpleVisualizer::DrawCubeTest() {
  // Render some stuff
  glColor3f(1.0, 0.0, 1.0);
  pangolin::glDrawColouredCube();
}

void SimpleVisualizer::DrawCam(const float scale) {
  if (scale < 0) {
    std::cerr << "scale should be positive!\n";
    return;
  }

  const float w = 0.2 * scale;
  const float h = w * 0.75;
  const float z = w * 0.8;

  glLineWidth(2 * scale);
  glBegin(GL_LINES);
  glColor3f(0.0f, 1.0f, 1.0f);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);
  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);
  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);
  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();

  return;
}

void SimpleVisualizer::DrawCamWithPose(
    const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat) {
  if (!this->camera_visible_) {
    return;
  }

  Eigen::Matrix3d R = quat.toRotationMatrix();

  glPushMatrix();
  std::vector<GLdouble> Twc = {R(0, 0), R(1, 0), R(2, 0), 0.,
                               R(0, 1), R(1, 1), R(2, 1), 0.,
                               R(0, 2), R(1, 2), R(2, 2), 0.,
                               pos.x(), pos.y(), pos.z(), 1.};
  glMultMatrixd(Twc.data());
  DrawCam();
  glPopMatrix();
}

void SimpleVisualizer::DrawTraj(const std::vector<Eigen::Vector3d> &traj,
                                double color_r,
                                double color_g,
                                double color_b) {
  if (!this->traj_visible_) {
    return;
  }
  glLineWidth(2);
  glBegin(GL_LINES);
  glColor3f(color_r, color_g, color_b);
  for (size_t i = 0; i < traj.size() - 1; i++) {
    glVertex3d(traj[i].x(), traj[i].y(), traj[i].z());
    glVertex3d(traj[i + 1].x(), traj[i + 1].y(), traj[i + 1].z());
  }
  glEnd();
}

void SimpleVisualizer::DrawParticles(
    const std::vector<Eigen::Vector3d> &particles) {
  if (!this->particles_visible_) {
    return;
  }

  // draw points
  glPointSize(2.0f);
  glBegin(GL_POINTS);
  glColor3f(1.0f, 0.0f, 0.0f);

  for (size_t i = 0; i < particles.size(); i++) {
    glVertex3d(particles[i].x(), particles[i].y(), particles[i].z());
  }
  glEnd();
}

void SimpleVisualizer::DrawParticles(const std::vector<Eigen::Vector3d> &particles, std::vector<double> &weights) {
  if (!this->particles_visible_) {
    return;
  }

  assert(particles.size() == weights.size());

  double max_weight = 0.0;
  for (int i = 0; i < weights.size(); i++) {
    if (weights.at(i) < max_weight) {
      continue;
    }
    max_weight = weights.at(i);
  }
  for (int i = 0; i < weights.size(); i++) {
    weights.at(i) = weights.at(i) / max_weight;
  }

  // draw points
  glPointSize(2.0f);
  glBegin(GL_POINTS);

  for (size_t i = 0; i < particles.size(); i++) {
    glColor3f(static_cast<float>(weights.at(i)), 0.0f, 0.0f);
    glVertex3d(particles[i].x(), particles[i].y(), particles[i].z());
  }
  glEnd();
}

void SimpleVisualizer::DrawCoordinate() {
  if (!this->coordinate_visible_) {
    return;
  }
  glLineWidth(3);
  glBegin(GL_LINES);
  glColor3f(1.0f, 0.f, 0.f);
  glVertex3f(0, 0, 0);
  glVertex3f(1, 0, 0);
  glColor3f(0.f, 1.0f, 0.f);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 1, 0);
  glColor3f(0.f, 0.f, 1.f);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 0, 1);
  glEnd();
}

void SimpleVisualizer::DrawReferenceGridXY(
    double x_min, double x_max,
    double y_min, double y_max,
    double grid_interval) {
  if (!this->reference_grid_visible_) {
    return;
  }

  // draw horizontal plane and grids.
  glLineWidth(1);
  glBegin(GL_LINES);
  glColor3f(211.0f, 211.0f, 211.0f);

  for (int x = x_min; x <= x_max; x++) {
    glVertex3f(x * grid_interval, y_min * grid_interval, 0.0);
    glVertex3f(x * grid_interval, y_max * grid_interval, 0.0);
  }

  for (int y = y_min; y <= y_max; y++) {
    glVertex3f(x_min * grid_interval, y * grid_interval, 0.0);
    glVertex3f(x_max * grid_interval, y * grid_interval, 0.0);
  }

  glEnd();
}

void SimpleVisualizer::DisplayData(const Eigen::Vector3d &pos,
                                   const Eigen::Quaterniond &quat) {
  VecXd tmp_pose, tmp_euler;
  tmp_pose.vec_ = pos;
  tmp_euler.vec_ = quat.matrix().eulerAngles(2, 1, 0);

  tmp_euler.vec_ *= (180 / M_PI);
  data_set_[0] = tmp_pose;
  data_set_[1] = tmp_euler;
}

void SimpleVisualizer::RegisterUICallback() {
  camera_visible_ = ui_set_[0] ? true : false;
  traj_visible_ = ui_set_[1] ? true : false;
  coordinate_visible_ = ui_set_[2] ? true : false;
  particles_visible_ = ui_set_[3] ? true : false;
  reference_grid_visible_ = ui_set_[4] ? true : false;
}

}  // namespace visualization

}  // namespace state_estimation
