/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2020-12-24 16:16:08
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-05-16 19:20:56
 */
#ifndef STATE_ESTIMATION_FILTER_PARTICLE_FILTER_H_
#define STATE_ESTIMATION_FILTER_PARTICLE_FILTER_H_

#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#if (defined ENABLE_MULTITHREADING_PARTICLE_UPDATE) || (defined ENABLE_MULTITHREADING_PARTICLE_PREDICT)
#include <mutex>
#endif
#include <random>
#include <string>
#include <set>
#include <list>
#if (defined ENABLE_MULTITHREADING_PARTICLE_UPDATE) || (defined ENABLE_MULTITHREADING_PARTICLE_PREDICT)
#include <thread>
#endif
#include <utility>
#include <vector>
#include <unordered_map>

#include "util/misc.h"
#include "util/queue_memory.h"
#include "variable/position.h"
#include "prediction_model/base.h"
#include "observation_model/base.h"
#include "observation_model/geomagnetism_observation_model.h"
#include "sampler/gaussian_sampler.h"

namespace state_estimation {

namespace filter {

enum class ParticleFilterModeType {
  kNormal = 0,
  kLocalization,
  kTracking,
};

template <typename ElementType>
void GetPointerVectorOfVectorElements(std::vector<ElementType>& element_vector, std::vector<ElementType*>* pointer_vector) {
  pointer_vector->clear();
  for (int i = 0; i < element_vector.size(); i++) {
    pointer_vector->emplace_back(&(element_vector.at(i)));
  }
}

#ifdef ENABLE_MULTITHREADING_PARTICLE_PREDICT
template <typename PredictionModelState, typename PredictionModelControlInput, typename PredictionModel>
void PredictParticle(std::vector<PredictionModelState> *particle_states_tminus,
                     PredictionModelControlInput *control_input_t,
                     double dt,
                     std::vector<PredictionModelState> *particle_states_t,
                     PredictionModel *prediction_model,
                     int start_particle_index,
                     int end_particle_index,
                     bool include_ideal_prediction) {
  assert(particle_states_t->size() == particle_states_tminus->size());
  for (int i = start_particle_index; i < end_particle_index; i++) {
    bool sub_ideal_prediction = false;
    if (include_ideal_prediction && i == 0) {
      sub_ideal_prediction = true;
    }
    PredictionModelState particle_state_t;
    prediction_model->Predict(&particle_state_t, &(particle_states_tminus->at(i)), control_input_t, dt, sub_ideal_prediction);
    particle_states_t->at(i) = particle_state_t;
  }
}
#endif

#ifdef ENABLE_MULTITHREADING_PARTICLE_UPDATE
template <typename PredictionModelState,
          typename Observation,
          typename ObservationModelState,
          typename ObservationModel>
void UpdateParticle(std::vector<PredictionModelState> *particle_states_t,
                    ObservationModel *observation_model,
                    Observation *observation,
                    std::vector<double> *particle_weights_tminus,
                    double *epsilon,
                    double *max_particle_weight_log_t,
                    std::vector<double> *particle_weights_log_t,
                    std::vector<double> *particle_update_log_probabilities_t,
                    int start_particle_index,
                    int end_particle_index,
                    std::mutex *my_mutex) {
  assert(end_particle_index >= start_particle_index);
  assert(end_particle_index <= particle_states_t->size());
  assert(particle_weights_log_t->size() == particle_weights_tminus->size());
  assert(particle_weights_log_t->size() == particle_update_log_probabilies_t->size());
  double temp_max_weight_log = 1.0;
  for (int i = start_particle_index; i < end_particle_index; i++) {
    PredictionModelState prediction_model_state = particle_states_t->at(i);
    ObservationModelState observation_model_state;
    observation_model_state.FromPredictionModelState(&prediction_model_state);
    double particle_weight_log_t = observation_model->GetProbabilityObservationConditioningStateLog(observation, &observation_model_state);
    particle_update_log_probabilities_t->at(i) = particle_weight_log_t;
    particle_weight_log_t += std::log(particle_weights_tminus->at(i));
    particle_weights_log_t->at(i) = particle_weight_log_t;
    if ((temp_max_weight_log > *epsilon) || (particle_weight_log_t > temp_max_weight_log)) {
      temp_max_weight_log = particle_weight_log_t;
    }
  }

  {
    std::lock_guard<std::mutex> guard(*my_mutex);
    if ((*max_particle_weight_log_t > *epsilon) || (temp_max_weight_log > *max_particle_weight_log_t)) {
      *max_particle_weight_log_t = temp_max_weight_log;
    }
  }
}
#endif

template <typename PredictionModelControlInput>
class ParticleFilterControlInput {
 public:
  const PredictionModelControlInput& control_input(void) const {
    return this->control_input_;
  }

  PredictionModelControlInput* control_input_ptr(void) {
    return &(this->control_input_);
  }

  void control_input(const PredictionModelControlInput& control_input) {
    this->control_input_ = control_input;
  }

  const double& timestamp(void) const {
    return this->timestamp_;
  }

  void timestamp(const double& timestamp) {
    this->timestamp_ = timestamp;
  }

  ParticleFilterControlInput(void) {}
  ~ParticleFilterControlInput() {}

 private:
  PredictionModelControlInput control_input_;
  double timestamp_;
};

template <typename PredictionModelState,
          typename PredictionModelControlInput,
          typename ObservationModelObservation>
class ParticleFilterState {
 public:
  const std::vector<PredictionModelState>& particle_states(void) const {
    return this->particle_states_;
  }

  std::vector<PredictionModelState>* particle_states_ptr(void) {
    return &(this->particle_states_);
  }

  const PredictionModelState& particle_states(int particle_index) const {
    return this->particle_states_[particle_index];
  }

  void particle_states(const std::vector<PredictionModelState>& particle_states) {
    this->particle_states_ = particle_states;
  }

  void particle_states(std::vector<PredictionModelState>&& particle_states) {
    this->particle_states_ = std::forward<std::vector<PredictionModelState>>(particle_states);
  }

  const std::vector<double>& particle_weights(void) const {
    return this->particle_weights_;
  }

  const double& particle_weights(int particle_index) const {
    return this->particle_weights_[particle_index];
  }

  void particle_weights(const std::vector<double>& particle_weights) {
    this->particle_weights_ = particle_weights;
  }

  void particle_weights(std::vector<double>&& particle_weights) {
    this->particle_weights_ = std::forward<std::vector<double>>(particle_weights);
  }

  const ParticleFilterControlInput<PredictionModelControlInput>& filter_control_input(void) const {
    return this->filter_control_input_;
  }

  PredictionModelControlInput* prediction_model_control_input_ptr(void) {
    return this->filter_control_input_.control_input_ptr();
  }

  void filter_control_input(const ParticleFilterControlInput<PredictionModelControlInput>& filter_control_input) {
    this->filter_control_input_ = filter_control_input;
  }

  const ObservationModelObservation& observation(void) const {
    return this->observation_;
  }

  void observation(const ObservationModelObservation& observation) {
    this->observation_ = observation;
  }

  const double& timestamp(void) const {
    return this->timestamp_;
  }

  void timestamp(const double& timestamp) {
    this->timestamp_ = timestamp;
  }

  const int& number_of_particles(void) const {
    return this->number_of_particles_;
  }

  void number_of_particles(const int& number_of_particles) {
    this->number_of_particles_ = number_of_particles;
  }

  const bool& need_update(void) const {
    return this->need_update_;
  }

  void need_update(const bool& need_update) {
    this->need_update_ = need_update;
  }

  const std::vector<int>& backtrack_indices(void) const {
    return this->backtrack_indices_;
  }

  void backtrack_indices(const std::vector<int>& backtrack_indices) {
    this->backtrack_indices_ = backtrack_indices;
  }

  int Dump(std::ofstream &out_fs) {
    int dump_size = 0;
    int n_particles = this->particle_states_.size();
    out_fs.write(reinterpret_cast<char*>(&n_particles), sizeof(n_particles));
    dump_size += sizeof(n_particles);
    for (int i = 0; i < n_particles; i++) {
      // write position
      variable::Position temp_position = this->particle_states_.at(i).position();
      out_fs.write(reinterpret_cast<char*>(&temp_position), sizeof(temp_position));
      dump_size += sizeof(temp_position);
      // write yaw
      double temp_yaw = this->particle_states_.at(i).yaw();
      out_fs.write(reinterpret_cast<char*>(&temp_yaw), sizeof(temp_yaw));
      dump_size += sizeof(temp_yaw);
      // write bluetooth_offset
      double temp_bluetooth_offset = this->particle_states_.at(i).bluetooth_offset();
      out_fs.write(reinterpret_cast<char*>(&temp_bluetooth_offset), sizeof(temp_bluetooth_offset));
      dump_size += sizeof(temp_bluetooth_offset);
      // write wifi_offset
      double temp_wifi_offset = this->particle_states_.at(i).wifi_offset();
      out_fs.write(reinterpret_cast<char*>(&temp_wifi_offset), sizeof(temp_wifi_offset));
      dump_size += sizeof(temp_wifi_offset);
      // write geomagnetism_bias
      Eigen::Vector3d temp_geomagnetism_bias = this->particle_states_.at(i).geomagnetism_bias();
      double b_x = temp_geomagnetism_bias(0);
      double b_y = temp_geomagnetism_bias(1);
      double b_z = temp_geomagnetism_bias(2);
      out_fs.write(reinterpret_cast<char*>(&b_x), sizeof(b_x));
      dump_size += sizeof(b_x);
      out_fs.write(reinterpret_cast<char*>(&b_y), sizeof(b_y));
      dump_size += sizeof(b_y);
      out_fs.write(reinterpret_cast<char*>(&b_z), sizeof(b_z));
      dump_size += sizeof(b_z);
      // write state_prediction_log_probability
      double state_prediction_log_probability = this->particle_states_.at(i).state_prediction_log_probability();
      out_fs.write(reinterpret_cast<char*>(&state_prediction_log_probability), sizeof(state_prediction_log_probability));
      dump_size += sizeof(state_prediction_log_probability);
      // write state_update_log_probability
      double state_update_log_probability = this->particle_states_.at(i).state_update_log_probability();
      out_fs.write(reinterpret_cast<char*>(&state_update_log_probability), sizeof(state_update_log_probability));
      dump_size += sizeof(state_update_log_probability);
      // write particle_weight
      out_fs.write(reinterpret_cast<char*>(&(this->particle_weights_.at(i))), sizeof(this->particle_weights_.at(i)));
      dump_size += sizeof(this->particle_weights_.at(i));
      // write backtrack_indices
      int temp_backtrack_index = -1;
      if (this->backtrack_indices_.size() > 0) {
        temp_backtrack_index = this->backtrack_indices_.at(i);
      }
      out_fs.write(reinterpret_cast<char*>(&(temp_backtrack_index)), sizeof(temp_backtrack_index));
      dump_size += sizeof(temp_backtrack_index);
    }
    return dump_size;
  }

  int Load(std::ifstream &in_fs) {
    int n_particles;
    int load_size = 0;
    std::vector<PredictionModelState> particle_states;
    std::vector<double> particle_weights;
    std::vector<int> backtrack_indices;
    in_fs.read(reinterpret_cast<char*>(&n_particles), sizeof(n_particles));
    load_size += sizeof(n_particles);
    for (int i = 0; i < n_particles; i++) {
      PredictionModelState particle_state;
      double particle_weight;
      variable::Position temp_position;
      double temp_yaw;
      double temp_bluetooth_offset;
      double temp_wifi_offset;
      double geomagnetism_bias_x, geomagnetism_bias_y, geomagnetism_bias_z;
      double state_prediction_log_probability;
      double state_update_log_probability;
      int temp_backtrack_index;
      // read position
      in_fs.read(reinterpret_cast<char*>(&temp_position), sizeof(temp_position));
      load_size += sizeof(temp_position);
      // read yaw
      in_fs.read(reinterpret_cast<char*>(&temp_yaw), sizeof(temp_yaw));
      load_size += sizeof(temp_yaw);
      // read bluetooth_offset
      in_fs.read(reinterpret_cast<char*>(&temp_bluetooth_offset), sizeof(temp_bluetooth_offset));
      load_size += sizeof(temp_bluetooth_offset);
      // read wifi_offset
      in_fs.read(reinterpret_cast<char*>(&temp_wifi_offset), sizeof(temp_wifi_offset));
      load_size += sizeof(temp_wifi_offset);
      // read geomagnetism_bias
      in_fs.read(reinterpret_cast<char*>(&geomagnetism_bias_x), sizeof(geomagnetism_bias_x));
      load_size += sizeof(geomagnetism_bias_x);
      in_fs.read(reinterpret_cast<char*>(&geomagnetism_bias_y), sizeof(geomagnetism_bias_y));
      load_size += sizeof(geomagnetism_bias_y);
      in_fs.read(reinterpret_cast<char*>(&geomagnetism_bias_z), sizeof(geomagnetism_bias_z));
      load_size += sizeof(geomagnetism_bias_z);
      // read state_prediction_log_probability
      in_fs.read(reinterpret_cast<char*>(&state_prediction_log_probability), sizeof(state_prediction_log_probability));
      load_size += sizeof(state_prediction_log_probability);
      // read state_update_log_probability
      in_fs.read(reinterpret_cast<char*>(&state_update_log_probability), sizeof(state_update_log_probability));
      load_size += sizeof(state_update_log_probability);
      // read particle_weight
      in_fs.read(reinterpret_cast<char*>(&particle_weight), sizeof(particle_weight));
      load_size += sizeof(particle_weight);
      // read backtrack_index
      in_fs.read(reinterpret_cast<char*>(&temp_backtrack_index), sizeof(temp_backtrack_index));
      load_size += sizeof(temp_backtrack_index);
      particle_state.position(temp_position);
      particle_state.yaw(temp_yaw);
      particle_state.state_prediction_log_probability(state_prediction_log_probability);
      particle_state.state_update_log_probability(state_update_log_probability);
      particle_states.emplace_back(particle_state);
      particle_weights.emplace_back(particle_weight);
      backtrack_indices.emplace_back(temp_backtrack_index);
    }
    this->particle_states_ = particle_states;
    this->particle_weights_ = particle_weights;
    this->backtrack_indices_ = backtrack_indices;
    return load_size;
  }

  ParticleFilterState(const ParticleFilterState& particle_filter_state) {
    this->particle_states_ = particle_filter_state.particle_states_;
    this->particle_weights_ = particle_filter_state.particle_weights_;
    this->backtrack_indices_ = particle_filter_state.backtrack_indices_;
    this->filter_control_input_ = particle_filter_state.filter_control_input_;
    this->observation_ = particle_filter_state.observation_;
    this->timestamp_ = particle_filter_state.timestamp_;
    this->number_of_particles_ = particle_filter_state.number_of_particles_;
    this->need_update_ = particle_filter_state.need_update_;
  }

  ParticleFilterState& operator=(const ParticleFilterState& particle_filter_state) {
    this->particle_states_ = particle_filter_state.particle_states_;
    this->particle_weights_ = particle_filter_state.particle_weights_;
    this->backtrack_indices_ = particle_filter_state.backtrack_indices_;
    this->filter_control_input_ = particle_filter_state.filter_control_input_;
    this->observation_ = particle_filter_state.observation_;
    this->timestamp_ = particle_filter_state.timestamp_;
    this->number_of_particles_ = particle_filter_state.number_of_particles_;
    this->need_update_ = particle_filter_state.need_update_;
    return *this;
  }

  ParticleFilterState& operator=(ParticleFilterState&& particle_filter_state) {
    this->particle_states_ = std::move(particle_filter_state.particle_states_);
    this->particle_weights_ = std::move(particle_filter_state.particle_weights_);
    this->backtrack_indices_ = std::move(particle_filter_state.backtrack_indices_);
    this->filter_control_input_ = particle_filter_state.filter_control_input_;
    this->observation_ = particle_filter_state.observation_;
    this->timestamp_ = particle_filter_state.timestamp_;
    this->number_of_particles_ = particle_filter_state.number_of_particles_;
    this->need_update_ = particle_filter_state.need_update_;
    return *this;
  }

  ParticleFilterState(void) {}
  ~ParticleFilterState() {}

 private:
  std::vector<PredictionModelState> particle_states_;
  std::vector<double> particle_weights_;
  std::vector<int> backtrack_indices_;
  ParticleFilterControlInput<PredictionModelControlInput> filter_control_input_;
  ObservationModelObservation observation_;
  double timestamp_ = 0.0;
  int number_of_particles_ = 0;
  bool need_update_ = true;
};

template <typename PredictionModelState,
          typename PredictionModelControlInput,
          typename PredictionModelStateSampler,
          typename PredictionModel,
          typename ObservationModelObservation,
          typename ObservationModelState,
          typename ObservationModel>
class ParticleFilter {
 public:
  void Init(const PredictionModel& prediction_model,
            const PredictionModelStateSampler& prediction_model_state_sampler,
            const ObservationModel& observation_model,
            const int& initial_number_of_particles,
            const double& effective_population_ratio,
            const double& population_expansion_ratio,
            const int& filter_state_memory_size,
            const int& max_number_of_particles,
            const int& window_size,
            const bool& use_relative_observation,
            const bool& use_dense_relative_observation,
            const bool& use_geomagnetism_bias_exponential_averaging,
            const bool& use_orientation_geomagnetism_bias_correlated_jittering,
            const int& smoothed_filter_state_memory_size = -1) {
    this->prediction_model_ = prediction_model;
    this->prediction_model_state_sampler_ = prediction_model_state_sampler;
    this->observation_model_ = observation_model;
    this->initial_number_of_particles_ = initial_number_of_particles;
    this->effective_population_ratio_ = effective_population_ratio;
    this->population_expansion_ratio_ = population_expansion_ratio;
    this->filter_state_memory_size_ = filter_state_memory_size;
    this->max_number_of_particles_ = max_number_of_particles;
    this->window_size_ = window_size;
    this->use_relative_observation_ = use_relative_observation;
    this->use_dense_relative_observation_ = use_dense_relative_observation;
    this->use_geomagnetism_bias_exponential_averaging_ = use_geomagnetism_bias_exponential_averaging;
    this->use_orientation_geomagnetism_bias_correlated_jittering_ = use_orientation_geomagnetism_bias_correlated_jittering;
    this->position_prior_sampler_.Init(0.0, 0.25);

    ParticleFilterState<PredictionModelState,
                        PredictionModelControlInput,
                        ObservationModelObservation>
        particle_filter_state;
    this->filter_state_ = particle_filter_state;
    this->filter_state_memory_.Init(filter_state_memory_size);
    this->filter_state_memory_.Clear();

    if (smoothed_filter_state_memory_size < 0) {
      this->smoothed_filter_state_memory_size_ = filter_state_memory_size;
    } else if (smoothed_filter_state_memory_size > filter_state_memory_size + 1) {
      this->smoothed_filter_state_memory_size_ = filter_state_memory_size + 1;
    } else {
      this->smoothed_filter_state_memory_size_ = smoothed_filter_state_memory_size;
    }
    this->smoothed_filter_state_memory_.Init(this->smoothed_filter_state_memory_size_);
    this->smoothed_filter_state_memory_.Clear();
  }

  void Reset(const PredictionModelStateSampler& prediction_model_state_sampler) {
    this->prediction_model_state_sampler_ = prediction_model_state_sampler;
    this->Reset();
  }

  void Reset(void) {
    this->filter_state_  = ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>();
    this->filter_state_memory_.Clear();
    this->smoothed_filter_state_memory_.Clear();
  }

  double SetControlInput(const ParticleFilterControlInput<PredictionModelControlInput>& control_input_t) {
    // set control_input_t into the current particle_filter_state.
    // return the forwarding time difference bewteen the control-input and current filter-state.
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::SetControlInput" << std::endl;
#endif
    this->filter_state_.filter_control_input(control_input_t);
    return control_input_t.timestamp() - this->filter_state_.timestamp();
  }

  int SetObservation(const ObservationModelObservation& observation) {
    // set the observation into the current particle_filter_state.
    this->filter_state_.observation(observation);
    return 1;
  }

  int SetUpdateFlag(const bool& need_update) {
    this->filter_state_.need_update(need_update);
    return 1;
  }

  int InjectSpecifiedState(const PredictionModelState& specified_state, const int& particle_index) {
    std::vector<PredictionModelState> particle_states = this->filter_state_.particle_states();
    particle_states.at(particle_index) = specified_state;
    this->filter_state_.particle_states(particle_states);
    return particle_index;
  }

  const PredictionModelState& GetSpecifiedParticleState(const int& particle_index) {
    return this->filter_state_.particle_states(particle_index);
  }

  // Predict with prior.
  void BlindPredict(ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>* filter_state_t,
                    int number_of_particles) {
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::BlindPredict" << std::endl;
#endif
    std::vector<PredictionModelState> particle_states_t;
    std::vector<double> particle_weights_t;
    for (int i = 0; i < number_of_particles; i++) {
      particle_states_t.emplace_back(PredictionModelState());
      this->prediction_model_state_sampler_.Sample(&(particle_states_t.back()));
      particle_weights_t.emplace_back(1.0 / number_of_particles);
    }
    filter_state_t->particle_states(std::move(particle_states_t));
    filter_state_t->particle_weights(std::move(particle_weights_t));
    filter_state_t->timestamp(this->filter_state_.filter_control_input().timestamp());
    filter_state_t->number_of_particles(number_of_particles);
  }

  // Predict using this->filter_state_,
  // put the result into the temporal filter_state_t.
  void Predict(ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>* filter_state_t, bool include_ideal_prediction = false) {
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::Predict" << std::endl;
#endif
#ifdef RUNTIME_PROFILE_PARTICLE_FILTER_PREDICT
    MyTimer my_timer;
    my_timer.Start();
#endif
    std::vector<PredictionModelState>* particle_states_tminus = this->filter_state_.particle_states_ptr();
    std::vector<double> particle_weights_tminus = this->filter_state_.particle_weights();
    int number_of_particles = 0;
    double timestamp_tminus = this->filter_state_.timestamp();
    std::vector<PredictionModelState> particle_states_t;
    for (int i = 0; i < particle_states_tminus->size(); i++) {
      particle_states_t.emplace_back(PredictionModelState());
    }

    double dt = this->filter_state_.filter_control_input().timestamp() - timestamp_tminus;
#ifdef DEBUG_PARTICLE_FILTER_PREDICT_DT
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(5);
    std::cout << "ParitcleFilter::Predict: dt " << dt << std::endl;
#endif
    PredictionModelControlInput* control_input_t = this->filter_state_.prediction_model_control_input_ptr();
#ifdef ENABLE_MULTITHREADING_PARTICLE_PREDICT
    std::vector<std::thread> threads;
    int processor_count = std::thread::hardware_concurrency();
    if (processor_count == 0) {
      std::cout << "Concurrency: ParticleFilter::Predict: cannot detect the number of cores." << std::endl;
    }
    assert(processor_count > 0);
    int particles_per_processor = std::floor(particle_states_t.size() / processor_count);
    for (int i = 0; i < processor_count; i++) {
      int start_particle_index = i * particles_per_processor;
      int end_particle_index;
      if (i == processor_count - 1) {
        end_particle_index = particle_states_t.size();
      } else {
        end_particle_index = (i + 1) * particles_per_processor;
      }
      threads.emplace_back(std::thread(PredictParticle<PredictionModelState,
                                                       PredictionModelControlInput,
                                                       PredictionModel>, &particle_states_tminus, &control_input_t, dt, &particle_states_t, &this->prediction_model_, start_particle_index, end_particle_index, include_ideal_prediction));
    }
    for (auto& my_thread : threads) my_thread.join();
#else
    bool sub_ideal_prediction;
    for (int i = 0; i < this->filter_state_.number_of_particles(); i++) {
      sub_ideal_prediction = false;
      if (include_ideal_prediction && i == 0) {
        sub_ideal_prediction = true;
      }
      this->prediction_model_.Predict(&(particle_states_t.at(i)),
                                      &(particle_states_tminus->at(i)),
                                      control_input_t,
                                      dt,
                                      sub_ideal_prediction);
    }
#endif
    number_of_particles = this->filter_state_.number_of_particles();

    filter_state_t->particle_states(std::move(particle_states_t));
    filter_state_t->particle_weights(std::move(particle_weights_tminus));
    filter_state_t->timestamp(this->filter_state_.filter_control_input().timestamp());
    filter_state_t->number_of_particles(number_of_particles);

#ifdef RUNTIME_PROFILE_PARTICLE_FILTER_PREDICT
    std::cout << "RUNTIME_PROFILE: ParticleFilter::Predict: time_consumed: " << my_timer.TimePassedStr() << std::endl;
    my_timer.Close();
#endif
  }

  // given the temporal filter_state_t,
  // compute weights for particles in it,
  // (the observation which will be used for updating is expected to be inluded in this->filter_state_),
  // if the update is valid,
  // push the current this->filter_state_ into this->filter_state_memory_
  // and set the filter_state_t as this->filter_state_;
  int Update(ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>&& filter_state_t) {
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::Update" << std::endl;
#endif
#ifdef RUNTIME_PROFILE_PARTICLE_FILTER_UPDATE
    MyTimer my_timer;
    my_timer.Start();
#endif
    std::vector<PredictionModelState>* particle_states_t_ptr = filter_state_t.particle_states_ptr();
    const std::vector<double>& particle_weights_tminus = filter_state_t.particle_weights();

    // get the loaded observation from the current filter_state.
    ObservationModelObservation observation = this->filter_state_.observation();

    double particle_weight_log_t = 0.0;
    double max_particle_weight_log_t = 1.0;
    double epsilon = 1e-8;
    std::vector<double> particle_weights_log_t;
    std::vector<double> particle_update_log_probabilities_t;
    for (int i = 0; i < particle_states_t_ptr->size(); i++) {
      particle_weights_log_t.emplace_back(0.0);
      particle_update_log_probabilities_t.emplace_back(0.0);
    }
#ifdef ENABLE_MULTITHREADING_PARTICLE_UPDATE
    std::mutex my_mutex;
    std::vector<std::thread> threads;
    int processor_count = std::thread::hardware_concurrency();
    if (processor_count == 0) {
      std::cout << "Concurrency: ParticleFilter::Update: cannot detect the number of cores." << std::endl;
    }
    assert(processor_count > 0);
    int particles_per_processor = std::floor(particle_states_t.size() / processor_count);
    for (int i = 0; i < processor_count; i++) {
      int start_particle_index = i * particles_per_processor;
      int end_particle_index;
      if (i == processor_count - 1) {
        end_particle_index = particle_states_t.size();
      } else {
        end_particle_index = (i + 1) * particles_per_processor;
      }
      threads.emplace_back(std::thread(UpdateParticle<PredictionModelState,
                                                      ObservationModelObservation,
                                                      ObservationModelState,
                                                      ObservationModel>, &particle_states_t, &(this->observation_model_), &observation, &particle_weights_tminus, &epsilon, &max_particle_weight_log_t, &particle_weights_log_t, &particle_update_log_probabilities_t, start_particle_index, end_particle_index, &my_mutex));
    }
    for (auto& my_thread : threads) my_thread.join();
#else
    std::list<observation_model::Observation*> observation_ptr_window;
    auto observation_window_iter = this->observation_window_.begin();
    while (observation_window_iter != this->observation_window_.end()) {
      observation_ptr_window.emplace_back(&(*observation_window_iter));
      observation_window_iter++;
    }
    // --------- construct containers for dense relative observation mode.
    std::list<ObservationModelState> observation_model_state_window;
    std::list<observation_model::State*> observation_model_state_ptr_window;
    if (this->use_dense_relative_observation_) {
      for (int i = 0; i < this->particle_state_ptrs_window_.size(); i++) {
        observation_model_state_window.emplace_back(ObservationModelState());
        observation_model_state_ptr_window.emplace_back(&(observation_model_state_window.back()));
      }
    }
    // ---------------------------------------------------------------------
    ObservationModelState observation_model_state;
    for (int i = 0; i < particle_states_t_ptr->size(); i++) {
      observation_model_state.position(particle_states_t_ptr->at(i).position());
      observation_model_state.bluetooth_dynamic_offset(particle_states_t_ptr->at(i).bluetooth_offset());
      observation_model_state.wifi_dynamic_offset(particle_states_t_ptr->at(i).wifi_offset());
      observation_model_state.geomagnetism_bias(particle_states_t_ptr->at(i).geomagnetism_bias());
      observation_model_state.yaw(particle_states_t_ptr->at(i).yaw());
      observation_model_state.q_ws(particle_states_t_ptr->at(i).q_ws());
      if (this->use_dense_relative_observation_) {
        // auto prediction_model_state_window_iter = this->particle_states_window_.begin();
        auto prediction_model_state_ptrs_window_iter = this->particle_state_ptrs_window_.begin();
        auto observation_model_state_window_iter = observation_model_state_window.begin();
        // while (prediction_model_state_window_iter != this->particle_states_window_.end()) {
        while (prediction_model_state_ptrs_window_iter != this->particle_state_ptrs_window_.end()) {
          // (*observation_model_state_ptr_window_iter)->FromPredictionModelState(&((*prediction_model_state_window_iter).at(i)));
          // (*observation_model_state_window_iter).position((*prediction_model_state_window_iter).at(i).position());
          // (*observation_model_state_window_iter).bluetooth_dynamic_offset((*prediction_model_state_window_iter).at(i).bluetooth_offset());
          // (*observation_model_state_window_iter).wifi_dynamic_offset((*prediction_model_state_window_iter).at(i).wifi_offset());
          // (*observation_model_state_window_iter).geomagnetism_bias((*prediction_model_state_window_iter).at(i).geomagnetism_bias());
          // (*observation_model_state_window_iter).yaw((*prediction_model_state_window_iter).at(i).yaw());
          // (*observation_model_state_window_iter).q_ws((*prediction_model_state_window_iter).at(i).q_ws());
          (*observation_model_state_window_iter).position((*prediction_model_state_ptrs_window_iter).at(i)->position());
          (*observation_model_state_window_iter).bluetooth_dynamic_offset((*prediction_model_state_ptrs_window_iter).at(i)->bluetooth_offset());
          (*observation_model_state_window_iter).wifi_dynamic_offset((*prediction_model_state_ptrs_window_iter).at(i)->wifi_offset());
          (*observation_model_state_window_iter).geomagnetism_bias((*prediction_model_state_ptrs_window_iter).at(i)->geomagnetism_bias());
          (*observation_model_state_window_iter).yaw((*prediction_model_state_ptrs_window_iter).at(i)->yaw());
          (*observation_model_state_window_iter).q_ws((*prediction_model_state_ptrs_window_iter).at(i)->q_ws());
          // prediction_model_state_window_iter++;
          prediction_model_state_ptrs_window_iter++;
          observation_model_state_window_iter++;
        }
        particle_weight_log_t =
            this->observation_model_.GetProbabilityObservationConditioningStateLog(
                observation_ptr_window,
                observation_model_state_ptr_window,
                &observation,
                &observation_model_state);
      } else if (this->use_relative_observation_) {
        // PredictionModelState prediction_model_state_previous = this->particle_states_window_.front().at(i);
        PredictionModelState prediction_model_state_previous = *(this->particle_state_ptrs_window_.front().at(i));
        ObservationModelState observation_model_state_previous;
        observation_model_state_previous.FromPredictionModelState(&prediction_model_state_previous);
        particle_weight_log_t =
            this->observation_model_.GetProbabilityObservationConditioningStateLog(
                &(this->observation_window_.front()),
                &observation_model_state_previous,
                &observation,
                &observation_model_state);
      } else if (this->use_geomagnetism_bias_exponential_averaging_) {
        // averging bias
        observation_model::GeomagnetismObservation geomagnetism_observation = observation.geomagnetism_observation();
        observation_model::GeomagnetismObservationYawState geomagnetism_observation_state = observation_model_state.geomagnetism_observation_state();
        Eigen::Vector3d state_bias = particle_states_t_ptr->at(i).geomagnetism_bias();
        Eigen::Vector3d map_bias = this->observation_model_.geomagnetism_observation_model_ptr()->CalculateGeomagnetismBias(geomagnetism_observation, geomagnetism_observation_state);
        Eigen::Vector3d state_update_bias = 0.8 * state_bias + 0.2 * map_bias;
        particle_states_t_ptr->at(i).geomagnetism_bias(state_update_bias);
        observation_model_state.FromPredictionModelState(&(particle_states_t_ptr->at(i)));
        particle_weight_log_t =
            this->observation_model_.GetProbabilityObservationConditioningStateLog(
                &observation,
                &observation_model_state);
      } else {
        particle_weight_log_t =
            this->observation_model_.GetProbabilityObservationConditioningStateLog(
                &observation,
                &observation_model_state);
      }
      particle_update_log_probabilities_t.at(i) = particle_weight_log_t;
#ifdef DEBUG_PARTICLE_FILTER_UPDATE_WEIGHT
      std::cout.precision(10);
      std::cout << "ParticleFilter::Update: particle_weight_log_t: " << particle_weight_log_t << std::endl;
      std::cout << "ParticleFilter::Update: particle_weight_t: " << std::exp(particle_weight_log_t) << std::endl;
#endif
      particle_weight_log_t += std::log(particle_weights_tminus.at(i));
      if ((max_particle_weight_log_t > epsilon) || (particle_weight_log_t > max_particle_weight_log_t)) {
        max_particle_weight_log_t = particle_weight_log_t;
      }
      particle_weights_log_t.at(i) = particle_weight_log_t;
    }
#endif

    double particle_weights_t_sum = 0.0;
    double particle_weight_t;
    std::vector<double> particle_weights_t;
    for (int i = 0; i < particle_weights_log_t.size(); i++) {
      particle_weight_t = std::exp(particle_weights_log_t.at(i) - max_particle_weight_log_t);
      particle_weights_t_sum += particle_weight_t;
      particle_weights_t.emplace_back(particle_weight_t);
    }

    int update_success = 0;
    if (particle_weights_t_sum > 0.0) {
#ifdef DEBUG_PARTICLE_FILTER_UPDATE_WEIGHTS_SUM
      std::cout.precision(10);
      std::cout << "ParticleFilter::Update: particle_weights_t_sum: " << particle_weights_t_sum << std::endl;
#endif
      update_success = 1;
      for (int i = 0; i < particle_weights_t.size(); i++) {
        particle_weights_t.at(i) /= particle_weights_t_sum;
      }

      filter_state_t.particle_weights(std::move(particle_weights_t));

      // update the update_log_probability of particles
      for (int i = 0; i < particle_states_t_ptr->size(); i++) {
        double temp_state_update_log_probability = particle_states_t_ptr->at(i).state_update_log_probability() + particle_update_log_probabilities_t.at(i);
        particle_states_t_ptr->at(i).state_update_log_probability(temp_state_update_log_probability);
      }
      this->filter_state_memory_.Push(this->filter_state_);

      this->filter_state_ = std::forward<ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>>(filter_state_t);
    }

#ifdef RUNTIME_PROFILE_PARTICLE_FILTER_UPDATE
    std::cout << "RUNTIME_PROFILE: ParticleFilter::Update: time_consumed: " << my_timer.TimePassedStr() << std::endl;
    my_timer.Close();
#endif

    return update_success;
  }

  double GetNumberOfEffectiveParticles(std::set<int> indices_to_skip = std::set<int>()) {
    // assertions
    int current_number_of_particles = this->filter_state_.number_of_particles();
    int max_index_to_skip = -1;
    for (auto it = indices_to_skip.begin(); it != indices_to_skip.end(); it++) {
      assert(*it >= 0);
      if (*it > max_index_to_skip) {
        max_index_to_skip = *it;
      }
    }
    assert(current_number_of_particles > max_index_to_skip);

    // get and re-normalize weights
    std::vector<double> included_particle_weights;
    for (int i = 0; i < current_number_of_particles; i++) {
      if (indices_to_skip.find(i) != indices_to_skip.end()) {
        continue;
      }
      included_particle_weights.emplace_back(this->filter_state_.particle_weights(i));
    }

    NormalizeWeights(included_particle_weights);
    double weights_norm = 0.0;
    for (int i = 0; i < included_particle_weights.size(); i++) {
      weights_norm += std::pow(included_particle_weights.at(i), 2.0);
    }
    return (1.0 / weights_norm);
  }

  void LocalityBasedParticlePartition(double region_lateral_length, std::unordered_map<std::string, std::set<int>> &regional_index_sets) {
    regional_index_sets.clear();
    std::vector<PredictionModelState> particle_states = this->filter_state_.particle_states();
    for (int i = 0; i < this->filter_state_.number_of_particles(); i++) {
      variable::Position temp_position = particle_states.at(i).position();
      temp_position.Round(region_lateral_length);
      std::string position_key = temp_position.ToKey();
      if (regional_index_sets.find(position_key) != regional_index_sets.end()) {
        regional_index_sets.at(position_key).insert(i);
      } else {
        regional_index_sets.insert(std::pair<std::string, std::set<int>>(position_key, std::set<int>({i})));
      }
    }
  }

  void LocalResample(int number_of_particles, bool resample_jitter_state = false, std::set<int> indices_to_skip = std::set<int>()) {
    // The implementation of the traditional stratified resampling.
    // number_of_particles: the number_of_particles after resampling.
    // indices_to_skip: the indices of particles that do not need to be resampled.
    // the logic is: keep the incides_to_skip unchanged and stay in the same indices, resample the other to get the total number of particles equal to number_of_particles.
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::LocalResample" << std::endl;
#endif
    // do the assertions
    // - max index to skip is less than the number_of_particles (so that they can stay in the same places).
    int max_index_to_skip = -1;
    for (auto it = indices_to_skip.begin(); it != indices_to_skip.end(); it++) {
      assert(*it >= 0);
      if (*it > max_index_to_skip) {
        max_index_to_skip = *it;
      }
    }
    assert(number_of_particles > max_index_to_skip);
    int current_number_of_particles = this->filter_state_.number_of_particles();
    assert(current_number_of_particles > max_index_to_skip);

    int number_of_particles_to_skip = indices_to_skip.size();
    int number_of_particles_to_resample = number_of_particles - number_of_particles_to_skip;

    std::vector<PredictionModelState> particle_states_resampled;
    std::vector<double> particle_weights_resampled;
    std::vector<int> backtrack_indices;

    std::list<std::vector<PredictionModelState*>> particle_state_ptrs_window_resampled;
    for (auto it = this->particle_state_ptrs_window_.begin(); it != this->particle_state_ptrs_window_.end(); it++) {
      particle_state_ptrs_window_resampled.emplace_back(std::vector<PredictionModelState*>());
    }

    // locally partition the particles
    std::unordered_map<std::string, std::set<int>> regional_particle_index_sets;
    LocalityBasedParticlePartition(this->local_resampling_region_size_in_meters_, regional_particle_index_sets);

    int remaining_quota = number_of_particles_to_resample;
    for (auto region_set_iter = regional_particle_index_sets.begin(); region_set_iter != regional_particle_index_sets.end(); region_set_iter++) {
      int regional_target_number_of_particles = std::round(region_set_iter->second.size() * number_of_particles_to_resample / current_number_of_particles);
      if (regional_target_number_of_particles > remaining_quota) {
        regional_target_number_of_particles = remaining_quota;
      }
      remaining_quota -= regional_target_number_of_particles;

      std::vector<int> indices_to_resample;

      // copy the indices to skip
      for (auto index_iter = region_set_iter->second.begin(); index_iter != region_set_iter->second.end(); index_iter++) {
        if (indices_to_skip.find(*index_iter) != indices_to_skip.end()) {
          particle_states_resampled.emplace_back(this->filter_state_.particle_states(*index_iter));
          particle_weights_resampled.emplace_back(this->filter_state_.particle_weights(*index_iter));
          backtrack_indices.emplace_back(*index_iter);
          auto it_resampled = particle_state_ptrs_window_resampled.begin();
          auto it_origin = this->particle_state_ptrs_window_.begin();
          while (it_origin != this->particle_state_ptrs_window_.end()) {
            (*it_resampled).emplace_back((*it_origin).at(*index_iter));
            it_resampled++;
            it_origin++;
          }
        } else {
          indices_to_resample.emplace_back(*index_iter);
        }
      }

      // resample the quota
      if (regional_target_number_of_particles <= 0) {
        continue;
      } else {
        double cumulative_sum = 0.0;
        std::vector<double> cumulative_weights;
        std::vector<double> resample_id;
        for (int i = 0; i < indices_to_resample.size(); i++) {
          cumulative_sum += this->filter_state_.particle_weights(indices_to_resample.at(i));
          cumulative_weights.emplace_back(cumulative_sum);
        }

        std::uniform_real_distribution<double> distribution(0.0, cumulative_sum / regional_target_number_of_particles);

        for (int i = 0; i < regional_target_number_of_particles; i++) {
          double slot_sample_value = i * cumulative_sum / regional_target_number_of_particles + distribution(this->random_generator_);
          slot_sample_value = std::min(slot_sample_value, cumulative_sum);
          resample_id.emplace_back(slot_sample_value);
        }

        int temp_index = 0;
        double resampled_weight = cumulative_sum / regional_target_number_of_particles;
        int resample_id_index = 0;
        for (int i = 0; i < regional_target_number_of_particles; i++) {
          while (resample_id[resample_id_index] > cumulative_weights[temp_index]) {
            temp_index += 1;
          }
          PredictionModelState particle_state_resampled = this->filter_state_.particle_states(indices_to_resample.at(temp_index));
          if (temp_index != i && resample_jitter_state) {
            if (this->use_orientation_geomagnetism_bias_correlated_jittering_) {
              this->prediction_model_.JitterState(&particle_state_resampled, this->geomagnetism_s_, this->gravity_s_);
            } else {
              this->prediction_model_.JitterState(&particle_state_resampled);
            }
          }
          particle_states_resampled.emplace_back(particle_state_resampled);
          auto it_resampled = particle_state_ptrs_window_resampled.begin();
          auto it_origin = this->particle_state_ptrs_window_.begin();
          while (it_origin != this->particle_state_ptrs_window_.end()) {
            (*it_resampled).emplace_back((*it_origin).at(indices_to_resample.at(temp_index)));
            it_resampled++;
            it_origin++;
          }
          particle_weights_resampled.emplace_back(resampled_weight);
          backtrack_indices.emplace_back(indices_to_resample.at(temp_index));
          resample_id_index++;
        }
      }
    }

    assert(particle_states_resampled.size() == number_of_particles);
    assert(particle_weights_resampled.size() == number_of_particles);
    assert(backtrack_indices.size() == number_of_particles);

    this->filter_state_.particle_states(std::move(particle_states_resampled));
    this->particle_state_ptrs_window_ = std::move(particle_state_ptrs_window_resampled);
    this->filter_state_.particle_weights(particle_weights_resampled);
    this->filter_state_.backtrack_indices(backtrack_indices);
    this->filter_state_.number_of_particles(number_of_particles);
  }

  void ResamplePositionGivenPrior(Eigen::Vector3d prior_p) {
    int current_number_of_particles = this->filter_state_.number_of_particles();
    if (current_number_of_particles <= 0) {
      return;
    }
    std::vector<PredictionModelState>* particle_states_t_ptr = this->filter_state_.particle_states_ptr();
    std::vector<double> particle_weights_resampled;
    double weight = 1.0 / current_number_of_particles;
    for (int i = 0; i < particle_states_t_ptr->size(); i++) {
      variable::Position original_position = particle_states_t_ptr->at(i).position();
      double sampled_prior_x = prior_p(0) + this->position_prior_sampler_.Sample();
      double sampled_prior_y = prior_p(1) + this->position_prior_sampler_.Sample();
      variable::Position prior_position;
      prior_position.x(sampled_prior_x);
      prior_position.y(sampled_prior_y);
      double x_diff = sampled_prior_x - original_position.x();
      double y_diff = sampled_prior_y - original_position.y();
      particle_states_t_ptr->at(i).position(prior_position);
      auto prediction_model_state_ptrs_window_iter = this->particle_state_ptrs_window_.begin();
      while (prediction_model_state_ptrs_window_iter != this->particle_state_ptrs_window_.end()) {
        variable::Position temp_position;
        temp_position = prediction_model_state_ptrs_window_iter->at(i)->position();
        double new_x = temp_position.x() + x_diff;
        double new_y = temp_position.y() + y_diff;
        temp_position.x(new_x);
        temp_position.y(new_y);
        prediction_model_state_ptrs_window_iter->at(i)->position(temp_position);
        prediction_model_state_ptrs_window_iter++;
      }
      particle_weights_resampled.emplace_back(weight);
    }
    this->filter_state_.particle_weights(particle_weights_resampled);
  }

  void Resample(int number_of_particles, bool resample_jitter_state = false, std::set<int> indices_to_skip = std::set<int>()) {
    // The implementation of the traditional stratified resampling.
    // number_of_particles: the number_of_particles after resampling.
    // indices_to_skip: the indices of particles that do not need to be resampled.
    // the logic is: keep the incides_to_skip unchanged and stay in the same indices, resample the other to get the total number of particles equal to number_of_particles.
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::Resample" << std::endl;
#endif
    // do the assertions
    // - max index to skip is less than the number_of_particles (so that they can stay in the same places).
    int max_index_to_skip = -1;
    for (auto it = indices_to_skip.begin(); it != indices_to_skip.end(); it++) {
      assert(*it >= 0);
      if (*it > max_index_to_skip) {
        max_index_to_skip = *it;
      }
    }
    assert(number_of_particles > max_index_to_skip);
    int current_number_of_particles = this->filter_state_.number_of_particles();
    assert(current_number_of_particles > max_index_to_skip);

    int number_of_particles_to_skip = indices_to_skip.size();
    int number_of_particles_to_resample = number_of_particles - number_of_particles_to_skip;

    std::vector<PredictionModelState> particle_states_resampled;
    std::vector<double> particle_weights_resampled;
    std::vector<double> cumulative_weights;
    std::vector<double> resample_id;

    std::list<std::vector<PredictionModelState*>> particle_state_ptrs_window_resampled;
    for (auto it = this->particle_state_ptrs_window_.begin(); it != this->particle_state_ptrs_window_.end(); it++) {
      particle_state_ptrs_window_resampled.emplace_back(std::vector<PredictionModelState*>());
    }

    double cumulative_sum = 0.0;
    std::vector<int> indices_to_resample;
    for (int i = 0; i < current_number_of_particles; i++) {
      if (indices_to_skip.find(i) != indices_to_skip.end()) {
        continue;
      }
      indices_to_resample.emplace_back(i);
      cumulative_sum += this->filter_state_.particle_weights(i);
      cumulative_weights.emplace_back(cumulative_sum);
    }

    std::uniform_real_distribution<double> distribution(0.0, cumulative_sum / number_of_particles_to_resample);

    for (int i = 0; i < number_of_particles_to_resample; i++) {
      resample_id.emplace_back(i * (cumulative_sum / number_of_particles_to_resample) + distribution(this->random_generator_));
    }

    int temp_index = 0;
    double resampled_weight = cumulative_sum / number_of_particles_to_resample;
    int resample_id_index = 0;
    std::vector<int> backtrack_indices;
    for (int i = 0; i < number_of_particles; i++) {
      if (indices_to_skip.find(i) != indices_to_skip.end()) {
        particle_states_resampled.emplace_back(this->filter_state_.particle_states(i));
        auto it_resampled = particle_state_ptrs_window_resampled.begin();
        auto it_origin = this->particle_state_ptrs_window_.begin();
        while (it_origin != this->particle_state_ptrs_window_.end()) {
          (*it_resampled).emplace_back((*it_origin).at(i));
          it_resampled++;
          it_origin++;
        }
        particle_weights_resampled.emplace_back(this->filter_state_.particle_weights(i));
        backtrack_indices.emplace_back(i);
        continue;
      }
      while (resample_id[resample_id_index] > cumulative_weights[temp_index]) {
        temp_index += 1;
      }
      PredictionModelState particle_state_resampled = this->filter_state_.particle_states(indices_to_resample.at(temp_index));
      if (indices_to_resample.at(temp_index) != i && resample_jitter_state) {
        if (this->use_orientation_geomagnetism_bias_correlated_jittering_) {
          this->prediction_model_.JitterState(&particle_state_resampled, this->geomagnetism_s_, this->gravity_s_);
        } else {
          this->prediction_model_.JitterState(&particle_state_resampled);
        }
      }
      particle_states_resampled.emplace_back(particle_state_resampled);
      auto it_resampled = particle_state_ptrs_window_resampled.begin();
      // auto it_ptrs_window = this->particle_state_ptrs_window_.begin();
      auto it_origin = this->particle_state_ptrs_window_.begin();
      while (it_origin != this->particle_state_ptrs_window_.end()) {
      // while (it_ptrs_window != this->particle_state_ptrs_window_.end()) {
        (*it_resampled).emplace_back((*it_origin).at(indices_to_resample.at(temp_index)));
        // (*it_ptrs_window).at(i) = (*it_ptrs_window).at(indices_to_resample.at(temp_index));
        it_resampled++;
        it_origin++;
        // it_ptrs_window++;
      }
      particle_weights_resampled.emplace_back(resampled_weight);
      backtrack_indices.emplace_back(indices_to_resample.at(temp_index));
      resample_id_index++;
    }

    this->filter_state_.particle_states(std::move(particle_states_resampled));
    this->particle_state_ptrs_window_ = std::move(particle_state_ptrs_window_resampled);
    this->filter_state_.particle_weights(particle_weights_resampled);
    this->filter_state_.backtrack_indices(backtrack_indices);
    this->filter_state_.number_of_particles(number_of_particles);
  }

  int DumpFilterState(std::ofstream &out_fs) {
    if (!out_fs) {
      std::cout << "ParticleFilter::DumpFilterState: invalid output_stream." << std::endl;
      return 0;
    }
    int dump_size = this->filter_state_.Dump(out_fs);
    return dump_size;
  }

  int DumpFilterStateMemory(std::string dump_filepath) {
    std::ofstream dump_file(dump_filepath, std::ofstream::binary);
    if (!dump_file) {
      std::cout << "ParticleFilter::DumpFilterStateMemory: cannot open file: " << dump_filepath << std::endl;
      return 0;
    }
    int dump_size = 0;
    int memory_size = this->filter_state_memory_.GetCurrentSize();
    dump_file.write(reinterpret_cast<char*>(&memory_size), sizeof(memory_size));
    dump_size += sizeof(memory_size);
    for (int i = 0; i < memory_size; i++) {
      dump_size += this->filter_state_memory_.at(i).Dump(dump_file);
    }
    dump_file.close();
    return dump_size;
  }

  int BackwardSimulation(void) {
    // calculate the smoothing distribution using backward-simulation.
    // the smoothing distribution is directly sampled by updating the particle-states and weights in the filter-state-memo.
    // the backward state transition distribution is calculated using the historical samples.
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::BackwardSimulation" << std::endl;
#endif
#ifdef PARTICLE_FILTER_VERBOSE
    std::cout << "Filtering backward:" << std::endl;
#endif
    this->filter_state_memory_.Push(this->filter_state_);
    util::QueueMemory<ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>> filter_state_memory = this->filter_state_memory_;
    this->smoothed_filter_state_memory_.Clear();
    while (filter_state_memory.GetCurrentSize() > 1 && (!this->smoothed_filter_state_memory_.IsFull())) {
#ifdef PARTICLE_FILTER_VERBOSE
      std::cout << ".";
      std::cout.flush();
#endif
      // the process assumes that the current this->filter_state_ is already the smoothed one and already be pushed into the smoothed_filter_state_memory.
      // each iteration is calculating state_tminus_smooth from state_t_smooth.
      if (this->smoothed_filter_state_memory_.IsEmpty()) {
        // nothing has been smoothed, which mean the filter_state_memory contains all the states including the last updated one.
        filter_state_memory.Pop(&this->filter_state_);
      } else {
        // the current this->filter_state_ is already the smoothed state at time t and saved in the this->smoothed_filter_state_memory_;
        // use smoothed_state_t to calculate smoothed_state_tminus.
        ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation> filter_state_tminus_smooth;
        filter_state_memory.Pop(&filter_state_tminus_smooth);
        std::vector<PredictionModelState> particle_states_tminus = filter_state_tminus_smooth.particle_states();
        std::vector<double> particle_weights_tminus_origin = filter_state_tminus_smooth.particle_weights();
        PredictionModelControlInput control_input_t = filter_state_tminus_smooth.filter_control_input().control_input();
        double dt = filter_state_tminus_smooth.filter_control_input().timestamp() - filter_state_tminus_smooth.timestamp();

        std::cout << "ParticleFilter::BackwardSimulation: dt: " << dt << std::endl;

        double weights_tminus_smooth_sum = 0.0;
        std::vector<double> particle_weights_tminus_smooth;
        while (weights_tminus_smooth_sum <= 0.0) {
          // update the smoothed weights
          std::vector<PredictionModelState> particle_states_t = this->filter_state_.particle_states();
          std::vector<double> particle_weights_t = this->filter_state_.particle_weights();
          std::vector<double> particle_weights_tminus_smooth_log;
          double max_weight_tminus_smooth_log = 1.0;
          double epsilon = 1e-8;
          for (int i = 0; i < particle_states_tminus.size(); i++) {
            PredictionModelState particle_state_tminus = particle_states_tminus.at(i);
            double particle_weight_tminus_smooth_log;
            double temp_marginal = 0.0;
            for (int j = 0; j < particle_states_t.size(); j++) {
              PredictionModelState particle_state_t = particle_states_t.at(j);
              temp_marginal += std::exp(this->prediction_model_.CalculateStateTransitionProbabilityLog(&particle_state_t, &particle_state_tminus, &control_input_t, dt)) * particle_weights_t.at(j);
            }
            particle_weight_tminus_smooth_log = std::log(particle_weights_tminus_origin.at(i)) + std::log(temp_marginal);

            particle_weights_tminus_smooth_log.emplace_back(particle_weight_tminus_smooth_log);
            if ((max_weight_tminus_smooth_log > epsilon) || (particle_weight_tminus_smooth_log > max_weight_tminus_smooth_log)) {
              max_weight_tminus_smooth_log = particle_weight_tminus_smooth_log;
            }
          }
          if (std::exp(max_weight_tminus_smooth_log) == 0.0) {
            continue;
          }
          particle_weights_tminus_smooth.clear();
          double particle_weight_tminus_smooth;
          weights_tminus_smooth_sum = 0.0;
          for (int i = 0; i < particle_weights_tminus_smooth_log.size(); i++) {
            particle_weight_tminus_smooth = std::exp(particle_weights_tminus_smooth_log.at(i) - max_weight_tminus_smooth_log);
            weights_tminus_smooth_sum += particle_weight_tminus_smooth;
            particle_weights_tminus_smooth.emplace_back(particle_weight_tminus_smooth);
          }
        }

        for (int i = 0; i < particle_weights_tminus_smooth.size(); i++) {
          particle_weights_tminus_smooth.at(i) /= weights_tminus_smooth_sum;
        }

        filter_state_tminus_smooth.particle_weights(particle_weights_tminus_smooth);

        this->filter_state_ = filter_state_tminus_smooth;
      }

      // resample
      double number_of_effective_particles = this->GetNumberOfEffectiveParticles();
      double threshold_number_of_effective_particles = this->filter_state_.number_of_particles() * this->effective_population_ratio_;
      if (number_of_effective_particles < threshold_number_of_effective_particles) {
        this->Resample(this->filter_state_.number_of_particles());
      }

      this->smoothed_filter_state_memory_.Push(this->filter_state_);
    }
#ifdef PARTICLE_FILTER_VERBOSE
    std::cout << std::endl;
#endif
    this->filter_state_memory_.Pop(&this->filter_state_);
    return this->smoothed_filter_state_memory_.GetCurrentSize();
  }

  int BackwardSimulationResample(std::vector<double> state_discretization_resolutions, int smooth_start_index = -1, int max_number_of_retries = 10) {
    // calculate the smoothing distribution using backward-simulation.
    // the smoothing distribution is directly sampled by updating the particle-states and weights in the filter-state-memo.
    // the backward state transition distribution is sampled.
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::BackwardSimulationResample" << std::endl;
#endif
#ifdef PARTICLE_FILTER_VERBOSE
    std::cout << "Filtering backward:" << std::endl;
#endif
    this->smoothed_filter_state_memory_.Clear();
    bool initialized = false;
    if (smooth_start_index > 0) {
      while (this->filter_state_memory_.GetCurrentSize() > smooth_start_index) {
        this->smoothed_filter_state_memory_.Push(this->filter_state_);
        this->filter_state_memory_.Pop(&this->filter_state_);
      }
    }
    this->filter_state_memory_.Push(this->filter_state_);
    util::QueueMemory<ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>> filter_state_memory = this->filter_state_memory_;
    while (filter_state_memory.GetCurrentSize() > 1 && (!this->smoothed_filter_state_memory_.IsFull())) {
#ifdef PARTICLE_FILTER_VERBOSE
      std::cout << ".";
      std::cout.flush();
#endif
      // I do not take the initial state into account because the initial state is only a prior, estimation is not conducted at the initial state.
      // we do not care about p(x_0).
      if (!initialized) {
        filter_state_memory.Pop(&this->filter_state_);
        initialized = true;
      } else {
        ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation> filter_state_tminus_origin;
        filter_state_memory.Pop(&filter_state_tminus_origin);
        std::string temp_particle_state_key;

        // prepare grid_distribution_tminus
        std::unordered_map<std::string, double> grid_distribution_tminus;
        std::vector<PredictionModelState> particle_states_tminus_origin = filter_state_tminus_origin.particle_states();
        std::vector<double> particle_weights_tminus_origin = filter_state_tminus_origin.particle_weights();
        for (int i = 0; i < filter_state_tminus_origin.number_of_particles(); i++) {
          temp_particle_state_key = particle_states_tminus_origin.at(i).ToKey(state_discretization_resolutions);
          if (grid_distribution_tminus.find(temp_particle_state_key) != grid_distribution_tminus.end()) {
            grid_distribution_tminus.at(temp_particle_state_key) += particle_weights_tminus_origin.at(i);
          } else if (particle_weights_tminus_origin.at(i) > 0.0) {
            grid_distribution_tminus.insert(std::pair<std::string, double>(temp_particle_state_key, particle_weights_tminus_origin.at(i)));
          }
        }
#ifdef DEBUG_PARTICLE_FILTER_BACKWARDSIMULATION_GRID_DISTRIBUTION_TMINUS
        std::cout << "ParticleFilter::BackwardSimulationResample: filter_state_tminus_origin: number_of_particles: " << filter_state_tminus_origin.number_of_particles() << std::endl;
        std::cout << "ParticleFilter::BackwardSimulationResample: particle_states_tminus_origin: number_of_particles: " << particle_states_tminus_origin.size() << std::endl;
        std::cout << "ParticleFilter::BackwardSimulationResample: grid_distribution_tminus_size: " << grid_distribution_tminus.size() << std::endl;
        std::cout << "ParticleFilter::BackwardSimulationResample: grid_distribution_tminus keys:" << std::endl;
        for (auto it = grid_distribution_tminus.begin(); it != grid_distribution_tminus.end(); it++) {
          std::cout << "ParticleFilter::BackwardSimulationResample: " << it->first << std::endl;
        }
#endif

        // reverse-predict for particle_state_tminus
        ParticleFilterControlInput<PredictionModelControlInput> filter_control_input_tminus = filter_state_tminus_origin.filter_control_input();
        filter_control_input_tminus.timestamp(filter_state_tminus_origin.timestamp());
#ifdef DEBUG_PARTICLE_FILTER_BACKWARDSIMULATION_REVERSE_CONTROL_INPUT_TS
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout.precision(13);
        std::cout << "ParticleFilter::BackwardSimulationResample: filter_state_t timestamp: " << this->filter_state_.timestamp() << std::endl;
        std::cout << "ParticleFilter::BackwardSimulationResample: control_input_tminus timestamp: " << filter_control_input_tminus.timestamp() << std::endl;
#endif
        double forwarding_time_difference = this->SetControlInput(filter_control_input_tminus);
#ifdef DEBUG_PARTICLE_FILTER_BACKWARDSIMULATION_FORWARDING_TIME_DIFFERENCE
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout.precision(13);
        std::cout << "ParticleFilter::BackwardSimulationResample: forwarding time difference: " << forwarding_time_difference << std::endl;
#endif
        ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation> filter_state_tminus_smooth;
        double weights_tminus_smooth_sum = 0.0;
        std::vector<double> particle_weights_tminus_smooth;
        int number_of_retries = -1;
        while (weights_tminus_smooth_sum <= 0.0) {
          if (number_of_retries > max_number_of_retries) {
#ifdef PARTICLE_FILTER_VERBOSE
            std::cout << std::endl;
#endif
            this->filter_state_memory_.Pop(&this->filter_state_);
            return this->smoothed_filter_state_memory_.GetCurrentSize();
          }
          number_of_retries += 1;
          this->Predict(&filter_state_tminus_smooth);
          // update the smoothed weights
          std::vector<PredictionModelState> particle_states_tminus_smooth = filter_state_tminus_smooth.particle_states();
          std::vector<double> particle_weights_t = filter_state_tminus_smooth.particle_weights();
          std::vector<double> particle_weights_tminus_smooth_log;
          double max_weight_tminus_smooth_log = 1.0;
          double epsilon = 1e-8;
          for (int i = 0; i < particle_states_tminus_smooth.size(); i++) {
#ifdef DEBUG_PARTICLE_FILTER_BACKWARDSIMULATION_OUT_OF_RANGE
            std::cout << "ParticleFilter::BackwardSimulationResample: size of particle_states_tminus_smooth: " << particle_states_tminus_smooth.size() << std::endl;
            std::cout << "ParticleFilter::BackwardSimulationResample: index of the current particle: " << i + 1 << std::endl;
            std::cout << "ParticleFilter::BackwardSimulationResample: size of particle_weights_t: " << particle_weights_t.size() << std::endl;
#endif
            temp_particle_state_key = particle_states_tminus_smooth.at(i).ToKey(state_discretization_resolutions);
#ifdef DEBUG_PARTICLE_FILTER_BACKWARDSIMULATION_REVERSE_PREDICT_PARTICLE_STATE_KEY
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout.precision(5);
            std::cout << "ParticleFilter::BackwardSimulationResample: temp_particle_state_key:" << temp_particle_state_key << std::endl;
            std::cout << "ParticleFilter::BackwardSimulationResample: predicted_particle_position: " << particle_states_tminus_smooth.at(i).position().x() << ", " << particle_states_tminus_smooth.at(i).position().y() << std::endl;
#endif
            double particle_weight_tminus_smooth_log;
            if (grid_distribution_tminus.find(temp_particle_state_key) != grid_distribution_tminus.end()) {
              particle_weight_tminus_smooth_log = std::log(particle_weights_t.at(i)) + std::log(grid_distribution_tminus.at(temp_particle_state_key));
            } else {
              particle_weight_tminus_smooth_log = std::log(0.0);
            }
            particle_weights_tminus_smooth_log.emplace_back(particle_weight_tminus_smooth_log);
            if ((max_weight_tminus_smooth_log > epsilon) || (particle_weight_tminus_smooth_log > max_weight_tminus_smooth_log)) {
              max_weight_tminus_smooth_log = particle_weight_tminus_smooth_log;
            }
          }
          if (std::exp(max_weight_tminus_smooth_log) == 0.0) {
            continue;
          }
          particle_weights_tminus_smooth.clear();
          double particle_weight_tminus_smooth;
          weights_tminus_smooth_sum = 0.0;
          for (int i = 0; i < particle_weights_tminus_smooth_log.size(); i++) {
            particle_weight_tminus_smooth = std::exp(particle_weights_tminus_smooth_log.at(i) - max_weight_tminus_smooth_log);
#ifdef DEBUG_PARTICLE_FILTER_WEIGHTS_EXP
            std::cout << "                      " << particle_weight_tminus_smooth << std::endl;
#endif
            weights_tminus_smooth_sum += particle_weight_tminus_smooth;
            particle_weights_tminus_smooth.emplace_back(particle_weight_tminus_smooth);
          }
        }

#ifdef DEBUG_PARTICLE_FILTER_BACKWARDSIMULATION_PARTICLE_WEIGHTS_MINUS_T
        std::cout.setf(std::ios::scientific, std::ios::floatfield);
        std::cout.precision(10);
        std::cout << "ParticleFilter::BackwardSimulationResample: smoothed weights sum: " << weights_tminus_smooth_sum << std::endl;
#endif

        for (int i = 0; i < particle_weights_tminus_smooth.size(); i++) {
          particle_weights_tminus_smooth.at(i) /= weights_tminus_smooth_sum;
#ifdef DEBUG_PARTICLE_FILTER_BACKWARDSIMULATION_PARTICLE_WEIGHTS_MINUS_T
          if (particle_weights_tminus_smooth.at(i) < 0.0) {
            std::cout.setf(std::ios::scientific, std::ios::floatfield);
            std::cout.precision(10);
            std::cout << "ParticleFilter::BackwardSimulationResample: particle_weight_tminus_smooth less than zero: " << particle_weights_tminus_smooth.at(i) << std::endl;
          }
#endif
        }

#ifdef DEBUG_PARTICLE_FILTER_BACKWARDSIMULATION_PARTICLE_WEIGHTS_MINUS_T
        std::cout << "ParticleFilter::BackwardSimulationResample: particle_weights_tminus_smooth size: " << particle_weights_tminus_smooth.size() << std::endl;
        double temp_sum = 0.0;
        for (int i = 0; i < particle_weights_tminus_smooth.size(); i++) {
          temp_sum += particle_weights_tminus_smooth.at(i);
        }
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout.precision(10);
        std::cout << "ParticleFilter::BackwardSimulationResample: smoothed particle weights sum: " << temp_sum << std::endl;
#endif

        filter_state_tminus_smooth.filter_control_input(filter_state_tminus_origin.filter_control_input());
        filter_state_tminus_smooth.observation(filter_state_tminus_origin.observation());
        filter_state_tminus_smooth.need_update(filter_state_tminus_origin.need_update());
        filter_state_tminus_smooth.particle_weights(particle_weights_tminus_smooth);

        this->filter_state_ = filter_state_tminus_smooth;
      }

      // resample
      double number_of_effective_particles = this->GetNumberOfEffectiveParticles();
      double threshold_number_of_effective_particles = this->filter_state_.number_of_particles() * this->effective_population_ratio_;
      if (number_of_effective_particles < threshold_number_of_effective_particles) {
        this->Resample(this->filter_state_.number_of_particles());
      }

      this->smoothed_filter_state_memory_.Push(this->filter_state_);

      // PredictionModelState temp_state;
      // this->EstimateState(&temp_state);
      // std::cout.setf(std::ios::fixed, std::ios::floatfield);
      // std::cout.precision(13);
      // std::cout << "ParticleFilter::BackwardSimulationResample: " << this->filter_state_.timestamp() << std::endl;
      // std::cout << "ParticleFilter::BackwardSimulationResample: " << temp_state.position().x() << "," << temp_state.position().y() << std::endl;
    }
#ifdef PARTICLE_FILTER_VERBOSE
    std::cout << std::endl;
#endif
    this->filter_state_memory_.Pop(&this->filter_state_);
    return this->smoothed_filter_state_memory_.GetCurrentSize();
  }

  int StartLocalizationStage(const ParticleFilterControlInput<PredictionModelControlInput>& filter_control_input_t,
                             const ObservationModelObservation& observation) {
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::StartLocalizationStage" << std::endl;
#endif
    this->SetControlInput(filter_control_input_t);
    this->SetObservation(observation);
    this->SetUpdateFlag(true);

    ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation> filter_state_t;

    std::vector<PredictionModelState> particle_states_t;
    std::vector<double> particle_weights_t;

    this->prediction_model_state_sampler_.ResetTraverseState();
    while (!(this->prediction_model_state_sampler_.IsTraverseFinished())) {
      particle_states_t.emplace_back(PredictionModelState());
      this->prediction_model_state_sampler_.Traverse(&(particle_states_t.back()));
      particle_weights_t.emplace_back(1.0);
    }

    int number_of_travesed_particles = particle_states_t.size();

    filter_state_t.particle_states(std::move(particle_states_t));
    filter_state_t.particle_weights(std::move(particle_weights_t));
    filter_state_t.timestamp(this->filter_state_.filter_control_input().timestamp());
    filter_state_t.number_of_particles(number_of_travesed_particles);

    this->particle_states_window_.clear();
    this->particle_state_ptrs_window_.clear();
    this->observation_window_.clear();

    this->particle_states_window_.emplace_back(filter_state_t.particle_states());
    std::vector<PredictionModelState*> particle_state_ptrs;
    GetPointerVectorOfVectorElements(this->particle_states_window_.back(), &particle_state_ptrs);
    this->particle_state_ptrs_window_.emplace_back(std::move(particle_state_ptrs));
    this->observation_window_.emplace_back(observation);
    this->filter_state_ = std::move(filter_state_t);

    this->running_mode_ = ParticleFilterModeType::kLocalization;

    return number_of_travesed_particles;
  }

  void StartTrackingStage(int number_of_tracking_stage_particles, bool resample_jitter_state, const ObservationModelObservation& observation) {
    this->particle_states_window_.clear();
    this->particle_state_ptrs_window_.clear();
    this->observation_window_.clear();
    if (this->filter_state_.number_of_particles() > 0) {
      this->Resample(number_of_tracking_stage_particles, resample_jitter_state);
      this->particle_states_window_.emplace_back(this->filter_state_.particle_states());
      std::vector<PredictionModelState*> particle_state_ptrs;
      GetPointerVectorOfVectorElements(this->particle_states_window_.back(), &particle_state_ptrs);
      this->particle_state_ptrs_window_.emplace_back(std::move(particle_state_ptrs));
      this->observation_window_.emplace_back(observation);
    }
    this->running_mode_ = ParticleFilterModeType::kTracking;
  }

  void DeadReckoningStep(ParticleFilterControlInput<PredictionModelControlInput> filter_control_input_t,
                         bool include_ideal_prediction = false,
                         int number_of_particles = 0) {
    // step purely depend on the control_input, step without update.
    // load the bullets
    this->SetControlInput(filter_control_input_t);
    this->SetUpdateFlag(false);

    // the temporal filter state
    ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation> filter_state_t;

    // take the shot
    if (this->filter_state_.number_of_particles() == 0) {
      if (number_of_particles == 0) {
        this->BlindPredict(&filter_state_t, this->initial_number_of_particles_);
      } else {
        this->BlindPredict(&filter_state_t, number_of_particles);
      }
    } else {
      this->Predict(&filter_state_t, include_ideal_prediction);
    }

    this->filter_state_memory_.Push(this->filter_state_);

    this->filter_state_ = filter_state_t;

    this->particle_states_window_.emplace_back(filter_state_t.particle_states());
    std::vector<PredictionModelState*> particle_state_ptrs;
    GetPointerVectorOfVectorElements(this->particle_states_window_.back(), &particle_state_ptrs);
    this->particle_state_ptrs_window_.emplace_back(std::move(particle_state_ptrs));
    assert(this->particle_states_window_.size() == this->particle_state_ptrs_window_.size());
    while (this->particle_states_window_.size() > this->window_size_) {
      this->particle_states_window_.pop_front();
      this->particle_state_ptrs_window_.pop_front();
    }
  }

  int SimpleStep(ParticleFilterControlInput<PredictionModelControlInput> filter_control_input_t,
                 ObservationModelObservation observation,
                 bool resample_jitter_state = false,
                 bool include_ideal_prediction = false,
                 int number_of_particles = 0,
                 bool prior_p_valid = false,
                 Eigen::Vector3d prior_p = Eigen::Vector3d::Zero()) {
    // load the bullets
    this->SetControlInput(filter_control_input_t);
    this->SetObservation(observation);
    this->SetUpdateFlag(true);

    // the temporal filter state
    ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation> filter_state_t;

    // the success indicator
    int update_success = 0;

    // take the shot
#ifdef PROFILE_1
    MyTimer timer;
    timer.Start();
#endif
    if (this->filter_state_.number_of_particles() == 0) {
      if (number_of_particles == 0) {
        this->BlindPredict(&filter_state_t, this->initial_number_of_particles_);
      } else {
        this->BlindPredict(&filter_state_t, number_of_particles);
      }
      this->particle_states_window_.emplace_back(filter_state_t.particle_states());
      std::vector<PredictionModelState*> particle_state_ptrs;
      GetPointerVectorOfVectorElements(this->particle_states_window_.back(), &particle_state_ptrs);
      this->particle_state_ptrs_window_.emplace_back(std::move(particle_state_ptrs));
      this->filter_state_ = std::move(filter_state_t);
      this->observation_window_.emplace_back(observation);
      return 1;
    } else {
      this->Predict(&filter_state_t, include_ideal_prediction);
    }

    if (prior_p_valid) {
      this->ResamplePositionGivenPrior(prior_p);
    }

#ifdef PROFILE_1
    std::cout << "PROFILE_1::ParticleFilter::SimpleStep: time consumed by prediciton: " << timer.TimePassedStr() << std::endl;
    timer.Start();
#endif
    update_success = this->Update(std::move(filter_state_t));
#ifdef PROFILE_1
    std::cout << "PROFILE_1::ParticleFilter::SimpleStep: time consumed by update: " << timer.TimePassedStr() << std::endl;
#endif

    this->geomagnetism_s_ = observation.geomagnetism_observation_ptr()->GetFeatureValuesVector();
    this->gravity_s_ = observation.geomagnetism_observation_ptr()->gravity();

    // clean-up
    if (update_success) {
      double number_of_effective_particles = 0;
      if (include_ideal_prediction) {
        number_of_effective_particles = this->GetNumberOfEffectiveParticles(std::set<int>({0}));
      } else {
        number_of_effective_particles = this->GetNumberOfEffectiveParticles();
      }
      double threshold_number_of_effective_particles = this->filter_state_.number_of_particles() * this->effective_population_ratio_;
      if (number_of_effective_particles < threshold_number_of_effective_particles) {
        if (this->local_resampling_) {
          if (include_ideal_prediction) {
            this->LocalResample(this->filter_state_.number_of_particles(), resample_jitter_state, std::set<int>({0}));
          } else {
            this->LocalResample(this->filter_state_.number_of_particles(), resample_jitter_state);
          }
        } else {
          if (include_ideal_prediction) {
            this->Resample(this->filter_state_.number_of_particles(), resample_jitter_state, std::set<int>({0}));
          } else {
            this->Resample(this->filter_state_.number_of_particles(), resample_jitter_state);
          }
        }
      }
    }

    this->particle_states_window_.emplace_back(this->filter_state_.particle_states());
    std::vector<PredictionModelState*> particle_state_ptrs;
    GetPointerVectorOfVectorElements(this->particle_states_window_.back(), &particle_state_ptrs);
    this->particle_state_ptrs_window_.emplace_back(std::move(particle_state_ptrs));
    this->observation_window_.emplace_back(observation);
    assert(this->particle_states_window_.size() == this->observation_window_.size());
    assert(this->particle_states_window_.size() == this->particle_state_ptrs_window_.size());
    while (this->particle_states_window_.size() > this->window_size_) {
      this->particle_states_window_.pop_front();
      this->particle_state_ptrs_window_.pop_front();
      this->observation_window_.pop_front();
    }

    return update_success;
  }

  int RobustStep(ParticleFilterControlInput<PredictionModelControlInput> filter_control_input_t, ObservationModelObservation observation) {
    // separate caches for saving control_inputs and observations
    std::vector<ParticleFilterControlInput<PredictionModelControlInput>> filter_control_input_cache;
    std::vector<ObservationModelObservation> observation_cache;

    // take the shot
    // when the update is failed or the number of effective particles is below a specified threshold,
    // follow the two-stage expand-and-roll-back strategy to improve robustness:
    //    *the strategy prefers population expansion rather than rolling-back.
    //    *since previous steps are also robust-steps, so we ought to have more confidence on them.
    //    first-stage (expansion):
    //      keep expanding the particle population and trying to update
    //      till either update successfully or the number of particles is above a specified threshold.
    //      if the expansion stops with insuccessful update, go to the next stage.
    //    second-stage (rolling-back):
    //      while (roll back to the t-1 step and forward -> still fail) {
    //        t -= 1;
    //        if (t < the farest memory) break;
    //      }
#ifdef VERBOSE
    std::cout << "# ParticleFilter::RobustStep: Start ####" << std::endl;
    std::cout << "## max_number_of_particles: " << this->max_number_of_particles_ << std::endl;
#endif
    int update_success = 0;    // the success indicator for update filter_state to t.
    int n_rollback_steps = 1;  // the number of steps rolling back from the next (t) filter_state.
    int normal_number_of_particles = this->initial_number_of_particles_;
    int expect_number_of_particles = normal_number_of_particles;
    while (!update_success) {
      if (this->filter_state_.number_of_particles() == 0) {
        // attempt at the initial stage
        if (expect_number_of_particles > this->max_number_of_particles_) {
          expect_number_of_particles = this->max_number_of_particles_;
        }
#ifdef VERBOSE
        std::cout << "### retry at the population initialization stage "
                  << expect_number_of_particles << "/" << this->max_number_of_particles_ << ": ";
#endif
        update_success = SimpleStep(filter_control_input_t, observation, expect_number_of_particles);
#ifdef VERBOSE
        std::cout << "update_success: " << update_success << std::endl;
#endif
        if (expect_number_of_particles >= this->max_number_of_particles_ && (!update_success)) {
          break;
        }
        expect_number_of_particles *= this->population_expansion_ratio_;
      } else if (expect_number_of_particles == normal_number_of_particles) {
        // retry without population expansion.
#ifdef VERBOSE
        std::cout << "### retry without population expansion "
                  << expect_number_of_particles << "/" << this->max_number_of_particles_ << ": ";
#endif
        update_success = SimpleStep(filter_control_input_t, observation);
        expect_number_of_particles *= this->population_expansion_ratio_;
#ifdef VERBOSE
        std::cout << "update_success: " << update_success << std::endl;
#endif
      } else if (expect_number_of_particles < this->max_number_of_particles_) {
        // expand the population of particles
#ifdef VERBOSE
        std::cout << "### retry with population expansion "
                  << expect_number_of_particles << "/" << this->max_number_of_particles_ << ": ";
#endif
        this->Resample(expect_number_of_particles);
        update_success = SimpleStep(filter_control_input_t, observation);
        expect_number_of_particles *= this->population_expansion_ratio_;
#ifdef VERBOSE
        std::cout << "update_success: " << update_success << std::endl;
#endif
      } else if (expect_number_of_particles != this->filter_state_.number_of_particles()) {
        // expand the population to the max number of particles
        expect_number_of_particles = this->max_number_of_particles_;
#ifdef VERBOSE
        std::cout << "### retry with maximum expansion of population"
                  << expect_number_of_particles << "/" << this->max_number_of_particles_ << ": ";
#endif
        this->Resample(expect_number_of_particles);
        update_success = SimpleStep(filter_control_input_t, observation);
#ifdef VERBOSE
        std::cout << "update_success: " << update_success << std::endl;
#endif
      } else {
        // roll back
        if (n_rollback_steps > this->filter_state_memory_.GetCurrentSize()) {
          // the rolling-back process is at its end.
          break;
        } else {
          // perform the rolling-back
          // push the afterward step into the temporal memory and checkout the corresponding step.
          filter_control_input_cache.clear();
          observation_cache.clear();
          // potential redundant copy of data
          for (int i = 0; i < n_rollback_steps; i++) {
            filter_control_input_cache.emplace_back(this->filter_state_.filter_control_input());
            observation_cache.emplace_back(this->filter_state_.observation());
            this->filter_state_memory_.Pop(&(this->filter_state_));
          }
          if (this->filter_state_.number_of_particles() < expect_number_of_particles) {
            this->Resample(expect_number_of_particles);
          }
          update_success = this->SimpleStep(this->filter_state_.filter_control_input(), this->filter_state_.observation());
          while (!filter_control_input_cache.empty() && update_success) {
            filter_control_input_t = filter_control_input_cache.back();
            filter_control_input_cache.pop_back();
            observation = observation_cache.back();
            observation_cache.pop_back();
            update_success = this->SimpleStep(filter_control_input_t, observation);
          }
#ifdef VERBOSE
          std::cout << "### retry by rolling-back " << n_rollback_steps << " steps: "
                    << "update_success: " << update_success << std::endl;
          if (!filter_control_input_cache.empty()) {
            std::cout << "#### failed on a previously visited filter state." << std::endl;
          }
#endif
          n_rollback_steps += 1;
        }
      }
    }
    if (this->filter_state_.number_of_particles() > normal_number_of_particles) {
#ifdef VERBOSE
      std::cout << "### shrink the population to normal size." << std::endl;
#endif
      this->Resample(normal_number_of_particles);
    }
#ifdef VERBOSE
    std::cout << "# ParticleFilter::RobustStep: End: update_success: "
              << update_success << " ####" << std::endl;
#endif
    return update_success;
  }

  void PushState(void) {
    this->filter_state_memory_.Push(this->filter_state_);
  }

  void PopState(void) {
    this->filter_state_memory_.Pop(&this->filter_state_);
  }

  int StepSimulationSmoothed(void) {
    if (this->smoothed_filter_state_memory_.IsEmpty()) {
      return 0;
    }
    this->smoothed_filter_state_memory_.Pop(&this->filter_state_);
    return 1;
  }

//   int EstimateState(PredictionModelState* est_state) {
// #ifdef DEBUG_FOCUSING
//     std::cout << "ParticleFilter::EstimateState" << std::endl;
// #endif
//     int number_of_particles = this->filter_state_.number_of_particles();
//     if (number_of_particles == 0) {
//       return 0;
//     } else {
//       std::vector<PredictionModelState> particle_states =
//           this->filter_state_.particle_states();
//       std::vector<double> particle_weights =
//           this->filter_state_.particle_weights();
//       est_state->operator=(particle_states[0]);
//       est_state->Multiply_scalar(particle_weights[0]);
//       PredictionModelState state;
//       for (int i = 1; i < number_of_particles; i++) {
// #ifdef DEBUG_PARTICLE_FILTER_ESTIMATESTATE
//         std::cout.precision(10);
//         std::cout << particle_weights[i] << std::endl;
// #endif
//         state.operator=(particle_states[i]);
//         state.Multiply_scalar(particle_weights[i]);
//         est_state->Add(&state);
//       }
//       return 1;
//     }
//   }

  int EstimateState(PredictionModelState* est_state, bool include_ideal_prediction = false) {
#ifdef DEBUG_FOCUSING
    std::cout << "ParticleFilter::EstimateState" << std::endl;
#endif
    int number_of_particles = this->filter_state_.number_of_particles();
    if (number_of_particles == 0) {
      return 0;
    } else {
      std::vector<PredictionModelState> particle_states = this->filter_state_.particle_states();
      std::vector<double> particle_weights = this->filter_state_.particle_weights();
      std::vector<prediction_model::State*> effective_particle_state_ptrs;
      std::vector<double> effective_particle_weights;
      for (int i = 0; i < particle_states.size(); i++) {
        if (include_ideal_prediction && i == 0) {
          continue;
        }
        effective_particle_state_ptrs.emplace_back(&particle_states.at(i));
        effective_particle_weights.emplace_back(particle_weights.at(i));
      }
      NormalizeWeights(effective_particle_weights);
      est_state->EstimateFromSamples(effective_particle_state_ptrs, effective_particle_weights);
      return 1;
    }
  }

  double EstimateDistanceVariance(PredictionModelState* est_state) {
    int number_of_particles = this->filter_state_.number_of_particles();
    if (number_of_particles == 0) {
      return 0;
    } else {
      std::vector<PredictionModelState> particle_states =
          this->filter_state_.particle_states();
      std::vector<double> particle_weights =
          this->filter_state_.particle_weights();
      double dist_variance = 0.0;
      state_estimation::variable::Position est_position = est_state->position();
      for (int i = 1; i < number_of_particles; i++) {
        state_estimation::variable::Position particle_position = particle_states[i].position();
        dist_variance += particle_weights[i] * (std::pow((est_position.x() - particle_position.x()), 2.0) + std::pow((est_position.y() - particle_position.y()), 2.0));
      }
      return dist_variance;
    }
  }

//   int CalculateCovariance(std::vector<std::vector<double>>* est_cov) {
//     PredictionModelState temp_state;
//     std::vector<double> temp_container;
//     int kNumberOfStateValues = temp_state.kNumberOfStateValues();
//     std::vector<double> means;
//     // setup the storage matrix.
//     for (int i = 0; i < kNumberOfStateValues; i++) {
//       means.emplace_back(0.0);
//       temp_container.clear();
//       for (int j = 0; j < kNumberOfStateValues; j++) {
//         temp_container.emplace_back(0.0);
//       }
//       est_cov->emplace_back(temp_container);
//     }
//
//     // compute values for the covariance matrix
//     std::vector<PredictionModelState> particle_states = this->filter_state_.particle_states();
//     std::vector<double> particle_weights = this->filter_state_.particle_weights();
//     std::vector<std::pair<std::string, double>> named_values;
//     int number_of_particles = particle_states.size();
//     double particle_weight = 1.0;
//     for (int i = 0; i < number_of_particles; i++) {
//       particle_states[i].GetAllNamedValues(&named_values);
//       particle_weight = particle_weights[i];
//       for (int j = 0; j < named_values.size(); j++) {
//         means[j] += named_values[j].second * particle_weight;
//         for (int k = j; k < named_values.size(); k++) {
//           est_cov->operator[](j)[k] += named_values[j].second * named_values[k].second * particle_weight;
//         }
//       }
//     }
//     for (int i = 0; i < kNumberOfStateValues; i++) {
//       for (int j = 0; j < kNumberOfStateValues; j++) {
//         if (j < i) {
//           est_cov->operator[](i)[j] = est_cov->operator[](j)[i];
//         } else {
//           est_cov->operator[](i)[j] = est_cov->operator[](i)[j] - means[i] * means[j];
//         }
//       }
//     }
//     return 1;
//   }

  const ParticleFilterState<PredictionModelState,
                            PredictionModelControlInput,
                            ObservationModelObservation>*
  filter_state(void) {
    return &(this->filter_state_);
  }

  const util::QueueMemory<
      ParticleFilterState<PredictionModelState,
                          PredictionModelControlInput,
                          ObservationModelObservation>>*
  filter_state_memory(void) {
    return &(this->filter_state_memory_);
  }

  int GetFilterStateMemoryCurrentSize(void) {
    return this->filter_state_memory_.GetCurrentSize();
  }

  const util::QueueMemory<
      ParticleFilterState<PredictionModelStateSampler,
                          PredictionModelControlInput,
                          ObservationModelObservation>>*
  smoothed_filter_state_memory(void) {
    return &(this->smoothed_filter_state_memory_);
  }

  int GetSmoothedFilterStateMemoryCurrentSize(void) {
    return this->smoothed_filter_state_memory_.GetCurrentSize();
  }

  void PushObservationToObservationWindow(ObservationModelObservation observation) {
    this->observation_window_.emplace_back(observation);
    while (this->observation_window_.size() > this->window_size_) {
      this->observation_window_.pop_front();
    }
  }

  int GetCurrentWindowSize(void) {
    assert(this->particle_states_window_.size() == this->observation_window_.size());
    assert(this->particle_states_window_.size() == this->particle_state_ptrs_window_.size());
    return this->particle_states_window_.size();
  }

  bool IsWindowFull(void) {
    assert(this->particle_states_window_.size() == this->observation_window_.size());
    assert(this->particle_states_window_.size() == this->particle_state_ptrs_window_.size());
    return (this->particle_states_window_.size() >= this->window_size_);
  }

  void use_relative_observation(bool use_relative_observation) {
    this->use_relative_observation_ = use_relative_observation;
  }

  bool use_relative_observation(void) {
    return this->use_relative_observation_;
  }

  void use_dense_relative_observation(bool use_dense_relative_observation) {
    this->use_dense_relative_observation_ = use_dense_relative_observation;
  }

  bool use_dense_relative_observation(void) {
    return this->use_dense_relative_observation_;
  }

  void Seed(int random_seed) {
    this->random_generator_.seed(static_cast<uint64_t>(random_seed));
  }

  PredictionModel* GetPredictionModelPtr(void) {
    return &(this->prediction_model_);
  }

  ParticleFilterModeType running_mode(void) {
    return this->running_mode_;
  }

  void SetObservationModel(ObservationModel observation_model) {
    this->observation_model_ = observation_model;
  }

  void SetLocalResampling(bool local_resampling) {
    this->local_resampling_ = local_resampling;
  }

  void SetLocalResamplingRegionSizeInMeters(double local_resampling_region_size_in_meters) {
    this->local_resampling_region_size_in_meters_ = local_resampling_region_size_in_meters;
  }

  ParticleFilter(void) {
    this->prediction_model_ = PredictionModel();
    this->prediction_model_state_sampler_ = PredictionModelStateSampler();
    this->observation_model_ = ObservationModel();
    this->initial_number_of_particles_ = 0;
    this->effective_population_ratio_ = 0.0;
    this->population_expansion_ratio_ = 0.0;
    this->filter_state_ = ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>();
    this->filter_state_memory_size_ = 0;
    this->filter_state_memory_ = util::QueueMemory<ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>>();
    this->smoothed_filter_state_memory_size_ = 0;
    this->smoothed_filter_state_memory_ = util::QueueMemory<ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>>();
    this->max_number_of_particles_ = 0;
    struct timespec tn;
    clock_gettime(CLOCK_REALTIME, &tn);
    this->random_generator_.seed(static_cast<uint64_t>(tn.tv_nsec));
    this->particle_states_window_ = std::list<std::vector<PredictionModelState>>();
    this->particle_state_ptrs_window_ = std::list<std::vector<PredictionModelState*>>();
    this->observation_window_ = std::list<ObservationModelObservation>();
    this->window_size_ = 0;
    this->use_relative_observation_ = true;
    this->use_dense_relative_observation_ = false;
    this->use_geomagnetism_bias_exponential_averaging_ = false;
    this->use_orientation_geomagnetism_bias_correlated_jittering_ = false;
    this->geomagnetism_s_ = Eigen::Vector3d::Zero();
    this->gravity_s_ = Eigen::Vector3d::Zero();
    this->running_mode_ = ParticleFilterModeType::kNormal;
    this->local_resampling_ = false;
    this->local_resampling_region_size_in_meters_ = 10.0;
  }
  ~ParticleFilter() {}

 private:
  PredictionModel prediction_model_;
  PredictionModelStateSampler prediction_model_state_sampler_;
  ObservationModel observation_model_;
  int initial_number_of_particles_;
  double effective_population_ratio_;
  double population_expansion_ratio_;
  ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation> filter_state_;
  int filter_state_memory_size_;
  util::QueueMemory<ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>> filter_state_memory_;
  int smoothed_filter_state_memory_size_;
  util::QueueMemory<ParticleFilterState<PredictionModelState, PredictionModelControlInput, ObservationModelObservation>> smoothed_filter_state_memory_;
  int max_number_of_particles_;
  std::default_random_engine random_generator_;
  std::list<std::vector<PredictionModelState>> particle_states_window_;
  // particle_state_ptrs_window_ is a structure for storing connections of particles between different filtering steps.
  // -- the reason for having it is that we do not want to directly change the elements of the particle_states_window, which may be a bottleneck of efficiency,
  // -- instead, we use particle_state_ptrs_window_ to keep track of ordering of particles while resampling.
  std::list<std::vector<PredictionModelState*>> particle_state_ptrs_window_;
  std::list<ObservationModelObservation> observation_window_;
  int window_size_;
  bool use_relative_observation_;
  bool use_dense_relative_observation_;
  bool use_geomagnetism_bias_exponential_averaging_;
  bool use_orientation_geomagnetism_bias_correlated_jittering_;
  Eigen::Vector3d geomagnetism_s_;
  Eigen::Vector3d gravity_s_;
  ParticleFilterModeType running_mode_;
  bool local_resampling_;
  double local_resampling_region_size_in_meters_;
  sampler::UnivariateGaussianSamplerStd position_prior_sampler_;
};

}  // namespace filter

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_FILTER_PARTICLE_FILTER_H_
