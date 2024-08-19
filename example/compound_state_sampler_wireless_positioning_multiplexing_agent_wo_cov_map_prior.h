#ifndef EXAMPLE_COMPOUND_STATE_SAMPLER_WIRELESS_POSITIONING_MULTIPLEXING_AGENT_WO_COV_MAP_PRIOR_H_
#define EXAMPLE_COMPOUND_STATE_SAMPLER_WIRELESS_POSITIONING_MULTIPLEXING_AGENT_WO_COV_MAP_PRIOR_H_

#include <iostream>

#include "prediction_model/base.h"
#include "prediction_model/compound_prediction_model.h"
#include "prediction_model/motion_model_2d_local_velocity_1d_rotation.h"
#include "prediction_model/parameter_model_random_walk.h"
#include "observation_model/geomagnetism_observation_model.h"
#include "distribution/probability_mapper_2d.h"
#include "sampler/gaussian_sampler.h"

namespace state_estimation {

namespace prediction_model {

const int kNumberOfSubmodelsInWPA = 2;

class CompoundPredictionModelStateSamplerWPA : public CompoundPredictionModelStateSampler {
 public:
  void Sample(State* state_sample) {
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelStateSamplerWPA::Sample" << std::endl;
#endif
    CompoundPredictionModelState* my_state = reinterpret_cast<CompoundPredictionModelState*>(state_sample);

    std::vector<StateSampler*> submodel_state_sampler_ptrs;
    this->submodel_state_sampler_ptrs(&submodel_state_sampler_ptrs);
    assert(submodel_state_sampler_ptrs.size() == kNumberOfSubmodelsInWPA);
    assert(my_state->GetNumberOfSubModelStates() == kNumberOfSubmodelsInWPA);

    double state_prediction_log_probability = 0.0;
    for (int i = 0; i < kNumberOfSubmodelsInWPA; i++) {
      submodel_state_sampler_ptrs.at(i)->Sample(my_state->at(i));
      state_prediction_log_probability += my_state->at(i)->state_prediction_log_probability();
    }

    my_state->state_prediction_log_probability(state_prediction_log_probability);
    my_state->state_update_log_probability(0.0);

    if (!this->geomagnetism_bias_use_map_prior_) {
      return;
    }

    // assign geomagnetism bias according to the map, position and orientation.
    MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler* state_sampler_0 = reinterpret_cast<MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler*>(submodel_state_sampler_ptrs.at(0));
    MotionModel2dLocalVelocity1dRotationState* my_state_0 = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(my_state->at(0));

    distribution::ProbabilityMapper2D geomagnetism_probability_mapper;
    state_sampler_0->geomagnetism_probability_mapper(&geomagnetism_probability_mapper);

    observation_model::GeomagnetismObservation geomagnetism_observation;
    observation_model::GeomagnetismObservationModelYaw<distribution::ProbabilityMapper2D> geomagnetism_observation_model;
    geomagnetism_observation_model.Init(geomagnetism_probability_mapper, state_sampler_0->position_griding_resolution());
    observation_model::GeomagnetismObservationYawState geomagnetism_observation_state;
    geomagnetism_observation_state.FromPredictionModelState(my_state_0);

    Eigen::Matrix3d R_mw;
    state_sampler_0->R_mw(&R_mw);

    Eigen::Vector3d gravity_s;
    state_sampler_0->gravity_s(&gravity_s);

    Eigen::Vector3d geomagnetism_s;
    state_sampler_0->geomagnetism_s(&geomagnetism_s);

    double yaw_w = geomagnetism_observation_state.yaw();

    Eigen::Quaterniond q_sgs = Eigen::Quaterniond::FromTwoVectors(gravity_s, Eigen::Vector3d({0.0, 0.0, 1.0}));
    Eigen::Quaterniond q_wsg(Eigen::AngleAxisd(yaw_w, Eigen::Vector3d({0.0, 0.0, 1.0})));
    q_sgs.normalize();
    q_wsg.normalize();

    Eigen::Quaterniond q_ms(R_mw * q_wsg * q_sgs);

    geomagnetism_observation.GetObservationFromGeomagnetismEigenVector(geomagnetism_s);

    Eigen::Vector3d geomagnetism_bias = geomagnetism_observation_model.CalculateGeomagnetismBias(geomagnetism_observation, geomagnetism_observation_state, q_ms);

    Eigen::Matrix3d bias_cov = Eigen::Matrix3d::Zero();
    bias_cov(0, 0) = 0.0;
    bias_cov(1, 1) = 0.0;
    bias_cov(2, 2) = 0.0;
    this->mvg_sampler_.SetParams(Eigen::Vector3d::Zero(), bias_cov);

    Eigen::Vector3d bias_noise = this->mvg_sampler_.Sample();

    geomagnetism_bias += bias_noise;

    ParameterModelRandomWalkState* my_state_1 = reinterpret_cast<ParameterModelRandomWalkState*>(my_state->at(1));
    Eigen::Matrix<double, 5, 1> parameters = my_state_1->parameters();
    parameters(2) = geomagnetism_bias(0);
    parameters(3) = geomagnetism_bias(1);
    parameters(4) = geomagnetism_bias(2);
    my_state_1->parameters(parameters);
  }

  void Traverse(State* state_sample) {
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelStateSamplerWPA::Traverse" << std::endl;
#endif
    if (this->IsTraverseFinished()) {
      std::cout << "CompoundPredictionModelStateSamplerWPA::Traverse: traversing finished." << std::endl;
      return;
    }
    CompoundPredictionModelState* my_state = reinterpret_cast<CompoundPredictionModelState*>(state_sample);

    std::vector<StateSampler*> submodel_state_sampler_ptrs;
    this->submodel_state_sampler_ptrs(&submodel_state_sampler_ptrs);
    assert(submodel_state_sampler_ptrs.size() == kNumberOfSubmodelsInWPA);
    assert(my_state->GetNumberOfSubModelStates() == kNumberOfSubmodelsInWPA);

    submodel_state_sampler_ptrs.at(0)->Traverse(my_state->at(0));
    submodel_state_sampler_ptrs.at(1)->Sample(my_state->at(1));
    double state_prediction_log_probability = my_state->at(0)->state_prediction_log_probability() + my_state->at(1)->state_prediction_log_probability();

    my_state->state_prediction_log_probability(state_prediction_log_probability);
    my_state->state_update_log_probability(0.0);

    if (!this->geomagnetism_bias_use_map_prior_) {
      return;
    }

    // assign geomagnetism bias according to the map, position and orientation.
    MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler* state_sampler_0 = reinterpret_cast<MotionModel2dLocalVelocity1dRotationMapWoCovStateSampler*>(submodel_state_sampler_ptrs.at(0));
    MotionModel2dLocalVelocity1dRotationState* my_state_0 = reinterpret_cast<MotionModel2dLocalVelocity1dRotationState*>(my_state->at(0));

    distribution::ProbabilityMapper2D geomagnetism_probability_mapper;
    state_sampler_0->geomagnetism_probability_mapper(&geomagnetism_probability_mapper);

    observation_model::GeomagnetismObservation geomagnetism_observation;
    observation_model::GeomagnetismObservationModelYaw<distribution::ProbabilityMapper2D> geomagnetism_observation_model;
    geomagnetism_observation_model.Init(geomagnetism_probability_mapper, state_sampler_0->position_griding_resolution());
    observation_model::GeomagnetismObservationYawState geomagnetism_observation_state;
    geomagnetism_observation_state.FromPredictionModelState(my_state_0);

    Eigen::Matrix3d R_mw;
    state_sampler_0->R_mw(&R_mw);

    Eigen::Vector3d gravity_s;
    state_sampler_0->gravity_s(&gravity_s);

    Eigen::Vector3d geomagnetism_s;
    state_sampler_0->geomagnetism_s(&geomagnetism_s);

    double yaw_w = geomagnetism_observation_state.yaw();

    Eigen::Quaterniond q_sgs = Eigen::Quaterniond::FromTwoVectors(gravity_s, Eigen::Vector3d({0.0, 0.0, 1.0}));
    Eigen::Quaterniond q_wsg(Eigen::AngleAxisd(yaw_w, Eigen::Vector3d({0.0, 0.0, 1.0})));
    q_sgs.normalize();
    q_wsg.normalize();

    Eigen::Quaterniond q_ms(R_mw * q_wsg * q_sgs);

    geomagnetism_observation.GetObservationFromGeomagnetismEigenVector(geomagnetism_s);

    Eigen::Vector3d geomagnetism_bias = geomagnetism_observation_model.CalculateGeomagnetismBias(geomagnetism_observation, geomagnetism_observation_state, q_ms);

    Eigen::Matrix3d bias_cov = Eigen::Matrix3d::Zero();
    bias_cov(0, 0) = 0.0;
    bias_cov(1, 1) = 0.0;
    bias_cov(2, 2) = 0.0;
    this->mvg_sampler_.SetParams(Eigen::Vector3d::Zero(), bias_cov);

    Eigen::Vector3d bias_noise = this->mvg_sampler_.Sample();

    geomagnetism_bias += bias_noise;

    ParameterModelRandomWalkState* my_state_1 = reinterpret_cast<ParameterModelRandomWalkState*>(my_state->at(1));
    Eigen::Matrix<double, 5, 1> parameters = my_state_1->parameters();
    parameters(2) = geomagnetism_bias(0);
    parameters(3) = geomagnetism_bias(1);
    parameters(4) = geomagnetism_bias(2);
    my_state_1->parameters(parameters);

  }

  void ResetTraverseState(void) {
    std::vector<StateSampler*> submodel_state_sampler_ptrs;
    this->submodel_state_sampler_ptrs(&submodel_state_sampler_ptrs);
    submodel_state_sampler_ptrs.at(0)->ResetTraverseState();
  }

  bool IsTraverseFinished(void) {
#ifdef DEBUG_FOCUSING
    std::cout << "CompoundPredictionModelStateSamplerWPA::IsTraverseFinished" << std::endl;
#endif
    std::vector<StateSampler*> submodel_state_sampler_ptrs;
    this->submodel_state_sampler_ptrs(&submodel_state_sampler_ptrs);
    return submodel_state_sampler_ptrs.at(0)->IsTraverseFinished();
  }

  void geomagnetism_bias_use_map_prior(bool geomagnetism_bias_use_map_prior) {
    this->geomagnetism_bias_use_map_prior_ = geomagnetism_bias_use_map_prior;
  }

  bool geomagnetism_bias_use_map_prior(void) {
    return this->geomagnetism_bias_use_map_prior_;
  }

  void Seed(int random_seed) {
    std::vector<StateSampler*> submodel_state_sampler_ptrs;
    this->submodel_state_sampler_ptrs(&submodel_state_sampler_ptrs);
    for (int i = 0; i < submodel_state_sampler_ptrs.size(); i++) {
      submodel_state_sampler_ptrs.at(i)->Seed(random_seed);
    }
    this->mvg_sampler_.Seed(random_seed);
  }

 private:
  bool geomagnetism_bias_use_map_prior_;
  sampler::MultivariateGaussianSampler mvg_sampler_;
};

}  // namespace prediction_model

}  // namespace state_estimation

#endif  // EXAMPLE_COMPOUND_STATE_SAMPLER_WIRELESS_POSITIONING_MULTIPLEXING_AGENT_WO_COV_MAP_PRIOR_H_
