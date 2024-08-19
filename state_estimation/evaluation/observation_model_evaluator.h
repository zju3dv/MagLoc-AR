/*
 * Copyright (c) 2020 SENSETIME
 * All rights reserved.
 * @Author: xuehua
 * @Date: 2021-02-01 17:19:27
 * @Last Modified by: xuehua
 * @Last Modified time: 2021-02-03 19:48:03
 */
#ifndef STATE_ESTIMATION_EVALUATION_OBSERVATION_MODEL_EVALUATOR_H_
#define STATE_ESTIMATION_EVALUATION_OBSERVATION_MODEL_EVALUATOR_H_

#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "evaluation/utils/Statistics.h"
#include "variable/position.h"

namespace state_estimation {

namespace evaluation {

template <typename ObservationModelObservation,
          typename ObservationModelState,
          typename ObservationModel>
class ObservationModelEvaluator {
 public:
  int Init(ObservationModel observation_model,
           std::vector<ObservationModelObservation> gt_observations,
           std::vector<ObservationModelState> gt_observation_states,
           std::vector<ObservationModelState> model_observation_states) {
    this->observation_model_ = observation_model;
    this->gt_observations_ = gt_observations;
    this->gt_observation_states_ = gt_observation_states;
    this->model_observation_states_ = model_observation_states;
    assert(this->gt_observations_.size() > 0);
    assert(this->gt_observations_.size() == this->gt_observation_states_.size());
    return 1;
  }

  void CalculateClosestPositionError(ov_eval::Statistics* closest_position_error) {
    closest_position_error->clear();

    for (int i = 0; i < this->gt_observation_states_.size(); i++) {
      // find the closest point in model_observation_states
      double closest_distance = -1.0;
      variable::Position target_position = this->gt_observation_states_.at(i).position();
      for (int j = 0; j < this->model_observation_states_.size(); j++) {
        variable::Position current_position = this->model_observation_states_.at(j).position();
        // calculate the 2D distance
        double current_distance = std::pow(
            std::pow(current_position.x() - target_position.x(), 2.0) +
                std::pow(current_position.y() - target_position.y(), 2.0),
            0.5);
        if (closest_distance < -0.5 || current_distance < closest_distance) {
          closest_distance = current_distance;
        }
      }
      closest_position_error->values.push_back(closest_distance);
    }

    closest_position_error->calculate();
  }

  void CalculateIndependentSumLogLikelihood(double* log_likelihood_combined, std::vector<double>* log_likelihood_separated) {
    // initial values
    *log_likelihood_combined = 0.0;
    log_likelihood_separated->clear();
    std::vector<std::pair<std::string, double>> sample_feature_vector = this->gt_observations_.at(0).GetFeatureVector();
    int sample_feature_dimensions = sample_feature_vector.size();
    for (int i = 0; i < sample_feature_dimensions; i++) {
      log_likelihood_separated->push_back(0.0);
    }
    // compute log likelihood
    for (int i = 0; i < this->gt_observations_.size(); i++) {
      // *likelihood_combined *= this->observation_model_.GetProbabilityObservationConditioningState(
      //     &(this->gt_observations_.at(i)),
      //     &(this->gt_observation_states_.at(i)));
      *log_likelihood_combined += std::log(this->observation_model_.GetProbabilityObservationConditioningState(
          &(this->gt_observations_.at(i)),
          &(this->gt_observation_states_.at(i))));
      std::vector<std::pair<std::string, double>> prob_separated;
      prob_separated = this->observation_model_.GetProbabilityObservationConditioningStateSeparated(
          &(this->gt_observations_.at(i)), &(this->gt_observation_states_.at(i)));
      for (int j = 0; j < sample_feature_dimensions; j++) {
        // log_likelihood_separated->at(j) *= prob_separated.at(j).second;
        log_likelihood_separated->at(j) += std::log(prob_separated.at(j).second);
      }
    }
  }

  void CalculateIndependentLikelihood(std::vector<ov_eval::Statistics>* likelihood_separated, ov_eval::Statistics* likelihood_combined) {
    // clear old values
    likelihood_separated->clear();
    likelihood_combined->clear();
    std::vector<std::pair<std::string, double>> sample_feature_vector = this->gt_observations_.at(0).GetFeatureVector();
    int sample_feature_dimensions = sample_feature_vector.size();
    for (int i = 0; i < sample_feature_dimensions; i++) {
      ov_eval::Statistics likelihood_statistics;
      likelihood_statistics.clear();
      likelihood_separated->push_back(likelihood_statistics);
    }

    for (int i = 0; i < this->gt_observations_.size(); i++) {
      double prob_combined = this->observation_model_.GetProbabilityObservationConditioningState(
          &(this->gt_observations_.at(i)),
          &(this->gt_observation_states_.at(i)));
      std::vector<std::pair<std::string, double>> prob_separated;
      prob_separated = this->observation_model_.GetProbabilityObservationConditioningStateSeparated(
          &(this->gt_observations_.at(i)), &(this->gt_observation_states_.at(i)));

      likelihood_combined->values.push_back(prob_combined);
      for (int j = 0; j < prob_separated.size(); j++) {
        likelihood_separated->at(j).values.push_back(prob_separated.at(j).second);
      }
    }

    likelihood_combined->calculate();
    for (int i = 0; i < likelihood_separated->size(); i++) {
      likelihood_separated->at(i).calculate();
    }
  }

  void CalculateIndependentNEES(ov_eval::Statistics* nees_combined, std::vector<ov_eval::Statistics>* nees_separated) {
    // clear old values
    nees_separated->clear();
    nees_combined->clear();
    std::vector<std::pair<std::string, double>> sample_feature_vector = this->gt_observations_.at(0).GetFeatureVector();
    int feature_dimensions = sample_feature_vector.size();
    for (int i = 0; i < feature_dimensions; i++) {
      ov_eval::Statistics nees_statistics;
      nees_statistics.clear();
      nees_separated->push_back(nees_statistics);
    }

    // compute independent nees
    for (int i = 0; i < this->gt_observations_.size(); i++) {
      std::vector<std::pair<std::string, double>> feature_vector = this->gt_observations_.at(i).GetFeatureVector();
      ObservationModelState observation_model_state = this->gt_observation_states_.at(i);
      const std::unordered_map<std::string, std::vector<double>>* distribution_params = nullptr;
      distribution_params = this->observation_model_.GetDistributionParams(&observation_model_state);
      if (!distribution_params) {
        continue;
      }
      double nees_all_features = 0.0;
      double nees_feature = 0.0;
      for (int j = 0; j < feature_vector.size(); j++) {
        std::string feature_key = feature_vector.at(j).first;
        double feature_value = feature_vector.at(j).second;
        if (distribution_params) {
          if (distribution_params->find(feature_key) == distribution_params->end()) {
            std::cout << "ObservationModelEvaluator::CalculateIndependentNEES: feature_key --"
                      << feature_key << "-- mismatch." << std::endl;
            return;
          }
          double est_mean = distribution_params->find(feature_key)->second[0];
          double est_std = distribution_params->find(feature_key)->second[1];
          nees_feature = std::pow((est_mean - feature_value), 2.0) / std::pow(est_std, 2.0);
        } else {
          nees_feature = -1.0;
        }
        nees_separated->at(j).timestamps.push_back(this->gt_observations_.at(i).timestamp());
        nees_separated->at(j).values.push_back(nees_feature);
        nees_all_features += nees_feature;
      }
      nees_combined->timestamps.push_back(this->gt_observations_.at(i).timestamp());
      nees_combined->values.push_back(nees_all_features);
    }

    nees_combined->calculate();
    for (int i = 0; i < feature_dimensions; i++) {
      nees_separated->at(i).calculate();
    }
  }

  void CalculateEstimation(std::vector<std::vector<double>>* model_means, std::vector<std::vector<double>>* model_stds, std::vector<std::vector<double>>* gt_means) {
    model_means->clear();
    model_stds->clear();
    gt_means->clear();

    // compute independent nees
    for (int i = 0; i < this->gt_observations_.size(); i++) {
      std::vector<std::pair<std::string, double>> feature_vector = this->gt_observations_.at(i).GetFeatureVector();
      ObservationModelState gt_observation_state = this->gt_observation_states_.at(i);
      const std::unordered_map<std::string, std::vector<double>>* distribution_params = nullptr;
      distribution_params = this->observation_model_.GetDistributionParams(&gt_observation_state);
      if (!distribution_params) {
        continue;
      }
      std::vector<double> est_means;
      std::vector<double> est_stds;
      std::vector<double> gt_values;
      for (int j = 0; j < feature_vector.size(); j++) {
        std::string feature_key = feature_vector.at(j).first;
        double feature_value = feature_vector.at(j).second;
        if (distribution_params->find(feature_key) == distribution_params->end()) {
          std::cout << "ObservationModelEvaluator::CalculateIndependentNEES: feature_key --"
                    << feature_key << "-- mismatch." << std::endl;
          continue;
        }
        double est_mean = distribution_params->find(feature_key)->second[0];
        double est_std = distribution_params->find(feature_key)->second[1];
        est_means.push_back(est_mean);
        est_stds.push_back(est_std);
        gt_values.push_back(feature_value);
      }
      model_means->push_back(est_means);
      model_stds->push_back(est_stds);
      gt_means->push_back(gt_values);
    }
  }

  void Evaluate(void) {
    this->CalculateClosestPositionError(&(this->closest_position_error_));
    this->CalculateIndependentSumLogLikelihood(&(this->sum_log_likelihood_combined_), &(this->sum_log_likelihood_separated_));
    this->CalculateIndependentNEES(&(this->nees_combined_), &(this->nees_separated_));
    this->CalculateEstimation(&(this->model_means_), &(this->model_stds_), &(this->gt_means_));
    this->CalculateIndependentLikelihood(&(this->likelihood_separated_), &(this->likelihood_combined_));
  }

  void OutputResults(std::string results_folderpath) {
    if (!std::filesystem::exists(results_folderpath)) {
      std::cout << "ObservationModelEvaluator::OutputResults: results_folderpath: "
                << results_folderpath << " does not exist." << std::endl;
    } else {
      // the current result folder is named by date
      time_t now = time(0);
      tm* local_time = std::localtime(&now);
      std::string current_foldername = std::to_string(local_time->tm_year + 1900) +
                                       "-" + std::to_string(local_time->tm_mon) +
                                       "-" + std::to_string(local_time->tm_mday) +
                                       "-" + std::to_string(local_time->tm_hour) +
                                       "-" + std::to_string(local_time->tm_min) +
                                       "-" + std::to_string(local_time->tm_sec);
      std::string current_folderpath;
      if (results_folderpath[results_folderpath.size() - 1] != '/') {
        current_folderpath = results_folderpath + "/" + current_foldername;
      } else {
        current_folderpath = results_folderpath + current_foldername;
      }
      if (std::filesystem::exists(current_folderpath)) {
        std::cout << "ObservationModelEvaluator::OutputResults: output_result_folderpath conflicts." << std::endl;
      } else {
        if (!std::filesystem::create_directory(current_folderpath)) {
          std::cout << "ObservationModelEvaluator::OutputResutls: create directory " << current_folderpath << " failed." << std::endl;
          return;
        }
        // write evaluation results into corresponding files.
        // write statistics resutls
        std::string statistics_result_filename = "statistics_result.csv";
        std::string statistics_result_filepath = current_folderpath + "/" + statistics_result_filename;
        std::ofstream statistics_result_file;
        statistics_result_file.open(statistics_result_filepath);
        if (!statistics_result_file) {
          std::cout << "ObservationModelEvaluator::OutputResults: create file error." << std::endl;
        } else {
          // write csv header
          for (int i = 0; i < this->sum_log_likelihood_separated_.size(); i++) {
            statistics_result_file << "likelihood_" << i << ",";
          }
          statistics_result_file << "likelihood\n";
          statistics_result_file.precision(15);
          for (int i = 0; i < this->sum_log_likelihood_separated_.size(); i++) {
            statistics_result_file << this->sum_log_likelihood_separated_.at(i) << ",";
          }
          statistics_result_file << this->sum_log_likelihood_combined_ << "\n";
          statistics_result_file.close();
        }

        // write sample-wise results
        std::string sample_wise_result_filename = "sample_wise_result.csv";
        std::string sample_wise_result_filepath = current_folderpath + "/" + sample_wise_result_filename;
        std::ofstream sample_wise_result_file;
        sample_wise_result_file.open(sample_wise_result_filepath);
        if (!sample_wise_result_file) {
          std::cout << "ObservationModelEvaluator::OutputResults: create file error." << std::endl;
        } else {
          // write csv header
          for (int i = 0; i < this->nees_separated_.size(); i++) {
            sample_wise_result_file << "nees_" << i << ",";
          }
          sample_wise_result_file << "nees,";
          for (int i = 0; i < this->likelihood_separated_.size(); i++) {
            sample_wise_result_file << "likelihood_" << i << ",";
          }
          sample_wise_result_file << "likelihood\n";
          sample_wise_result_file.precision(9);
          for (int i = 0; i < this->gt_observations_.size(); i++) {
            for (int j = 0; j < this->nees_separated_.size(); j++) {
              sample_wise_result_file << this->nees_separated_.at(j).values.at(i) << ",";
            }
            sample_wise_result_file << this->nees_combined_.values.at(i) << ",";
            for (int j = 0; j < this->likelihood_separated_.size(); j++) {
              sample_wise_result_file << this->likelihood_separated_.at(j).values.at(i) << ",";
            }
            sample_wise_result_file << this->likelihood_combined_.values.at(i) << "\n";
          }
          sample_wise_result_file.close();
        }

        // write gt-est results
        std::string gt_est_result_filename = "gt_est_result.csv";
        std::string gt_est_result_filepath = current_folderpath + "/" + gt_est_result_filename;
        std::ofstream gt_est_result_file;
        gt_est_result_file.open(gt_est_result_filepath);
        if (!gt_est_result_file) {
          std::cout << "ObservationModelEvaluator::OutputResults: create file error." << std::endl;
        } else {
          // write csv header
          assert(this->gt_means_.size() == this->model_means_.size());
          assert(this->model_means_.size() == this->model_stds_.size());
          for (int i = 0; i < this->gt_means_.at(0).size(); i++) {
            gt_est_result_file << "gt_feature_" << i << ",";
            gt_est_result_file << "est_mean_feature_" << i << ",";
            if (i == this->gt_means_.at(0).size() - 1) {
              gt_est_result_file << "est_std_feature_" << i << "\n";
            } else {
              gt_est_result_file << "est_std_feature_" << i << ",";
            }
          }
          // write csv values
          gt_est_result_file.precision(9);
          for (int i = 0; i < this->gt_means_.size(); i++) {
            for (int j = 0; j < this->gt_means_.at(0).size(); j++) {
              gt_est_result_file << this->gt_means_.at(i).at(j) << ",";
              gt_est_result_file << this->model_means_.at(i).at(j) << ",";
              if (j == this->gt_means_.at(0).size() - 1) {
                gt_est_result_file << this->model_stds_.at(i).at(j) << "\n";
              } else {
                gt_est_result_file << this->model_stds_.at(i).at(j) << ",";
              }
            }
          }
          gt_est_result_file.close();
        }
      }
    }
  }

  ObservationModelEvaluator(void) {}

  ~ObservationModelEvaluator() {}

 private:
  ObservationModel observation_model_;
  std::vector<ObservationModelObservation> gt_observations_;
  std::vector<ObservationModelState> gt_observation_states_;
  std::vector<std::vector<double>> model_means_;
  std::vector<std::vector<double>> model_stds_;
  std::vector<std::vector<double>> gt_means_;
  std::vector<ObservationModelState> model_observation_states_;
  std::vector<ov_eval::Statistics> nees_separated_;
  ov_eval::Statistics nees_combined_;
  std::vector<double> sum_log_likelihood_separated_;
  double sum_log_likelihood_combined_;
  std::vector<ov_eval::Statistics> likelihood_separated_;
  ov_eval::Statistics likelihood_combined_;
  ov_eval::Statistics closest_position_error_;
};

}  // namespace evaluation

}  // namespace state_estimation

#endif  // STATE_ESTIMATION_EVALUATION_OBSERVATION_MODEL_EVALUATOR_H_
