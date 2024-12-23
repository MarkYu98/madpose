#ifndef HYBRID_POSE_TWO_FOCAL_ESTIMATION_H
#define HYBRID_POSE_TWO_FOCAL_ESTIMATION_H

#include <RansacLib/ransac.h>
#include "optimizer.h"
#include "hybrid_ransac.h"
#include "solver.h"
#include "estimator_config.h"
#include <iostream>

namespace acmpose {

class HybridTwoFocalPoseEstimator {
public:
    HybridTwoFocalPoseEstimator(const std::vector<Eigen::Vector2d> &x0_norm, const std::vector<Eigen::Vector2d> &x1_norm,
                        const std::vector<double> &depth0, const std::vector<double> &depth1, 
                        const Eigen::Vector2d &min_depth, 
                        const double &norm_scale = 1.0,
                        const double &sampson_squared_weight = 1.0,
                        const std::vector<double> &squared_inlier_thresholds = {},
                        const EstimatorConfig &est_config = EstimatorConfig()) : 
                        sampson_squared_weight_(sampson_squared_weight), norm_scale_(norm_scale),
                        min_depth_(min_depth), squared_inlier_thresholds_(squared_inlier_thresholds), est_config_(est_config) {
                            assert(x0_norm.size() == x1_norm.size() && x0_norm.size() == depth0.size() && x0_norm.size() == depth1.size());
                            
                            d0_ = Eigen::Map<const Eigen::VectorXd>(depth0.data(), depth0.size());
                            d1_ = Eigen::Map<const Eigen::VectorXd>(depth1.data(), depth1.size());

                            x0_norm_ = Eigen::MatrixXd(3, x0_norm.size());
                            x1_norm_ = Eigen::MatrixXd(3, x1_norm.size());
                            for (int i = 0; i < x0_norm.size(); i++) {
                                x0_norm_.col(i) = x0_norm[i].homogeneous();
                                x1_norm_.col(i) = x1_norm[i].homogeneous();
                            }

                            solver_time = std::vector<std::vector<int>>(2);
                            nonmin_time = std::vector<std::vector<int>>(2);
                            lsq_time = std::vector<std::vector<int>>(2);
                        }  

    ~HybridTwoFocalPoseEstimator() {}

    inline int num_minimal_solvers() const { return 2; }

    inline int min_sample_size() const { return 7; }

    void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const {
        min_sample_sizes->resize(2);
        min_sample_sizes->at(0) = {4, 0};
        min_sample_sizes->at(1) = {0, 7};
    }

    inline int num_data_types() const { return 2; }

    void num_data(std::vector<int>* num_data) const {
        num_data->resize(2);
        num_data->at(0) = x0_norm_.cols();
        num_data->at(1) = x0_norm_.cols();
    }

    void solver_probabilities(std::vector<double>* probabilities) const {
        probabilities->resize(2);
        probabilities->at(0) = 1.0;
        probabilities->at(1) = 1.0;
        if (est_config_.solver_type == EstimatorOption::EPI_ONLY) {
            probabilities->at(0) = 0.0;
        }
        else if (est_config_.solver_type == EstimatorOption::MD_ONLY) {
            probabilities->at(1) = 0.0;
        }
    }

    inline int non_minimal_sample_size() const { return 36; }

    int MinimalSolver(const std::vector<std::vector<int>>& sample,
                      const int solver_idx, std::vector<PoseScaleOffsetTwoFocal>* models) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetTwoFocal* model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffsetTwoFocal& model, int t, int i, bool is_for_inlier=false) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetTwoFocal* model) const;

    double EvaluateModel(const PoseScaleOffsetTwoFocal& model) const;

    void GetInliers(const PoseScaleOffsetTwoFocal& model, std::vector<int>* inliers) const;

    std::string GetTimingInfo() const {
        std::stringstream ss;
        int solver_sum_0 = 0, solver_sum_1 = 0;
        for (auto &t : solver_time[0]) solver_sum_0 += t;
        for (auto &t : solver_time[1]) solver_sum_1 += t;
        double solver_avg_0 = double(solver_sum_0) / solver_time[0].size();
        double solver_avg_1 = double(solver_sum_1) / solver_time[1].size();

        ss << "Solver 0: sum " << solver_sum_0 << "ms; avg " << solver_avg_0 << "ms" << std::endl;
        ss << "Solver 1: sum " << solver_sum_1 << "ms; avg " << solver_avg_1 << "ms" << std::endl;

        int nonmin_sum_0 = 0, nonmin_sum_1 = 0;
        for (auto &t : nonmin_time[0]) nonmin_sum_0 += t;
        for (auto &t : nonmin_time[1]) nonmin_sum_1 += t;
        double nonmin_avg_0 = double(nonmin_sum_0) / nonmin_time[0].size();
        double nonmin_avg_1 = double(nonmin_sum_1) / nonmin_time[1].size();

        ss << "Nonmin 0: sum " << nonmin_sum_0 << "ms; avg " << nonmin_avg_0 << "ms" << std::endl;
        ss << "Nonmin 1: sum " << nonmin_sum_1 << "ms; avg " << nonmin_avg_1 << "ms" << std::endl;

        int lsq_sum_0 = 0, lsq_sum_1 = 0;
        for (auto &t : lsq_time[0]) lsq_sum_0 += t;
        for (auto &t : lsq_time[1]) lsq_sum_1 += t;
        double lsq_avg_0 = double(lsq_sum_0) / lsq_time[0].size();
        double lsq_avg_1 = double(lsq_sum_1) / lsq_time[1].size();

        ss << "LSQ 0: sum " << lsq_sum_0 << "ms; avg " << lsq_avg_0 << "ms" << std::endl;
        ss << "LSQ 1: sum " << lsq_sum_1 << "ms; avg " << lsq_avg_1 << "ms" << std::endl;

        return ss.str();
    }

protected:
    Eigen::MatrixXd x0_norm_, x1_norm_;
    Eigen::VectorXd d0_, d1_;
    Eigen::Vector2d min_depth_; 
    double sampson_squared_weight_;

    EstimatorConfig est_config_;
    std::vector<double> squared_inlier_thresholds_;
    double norm_scale_;

    std::vector<std::vector<int>> solver_time;
    std::vector<std::vector<int>> nonmin_time;
    std::vector<std::vector<int>> lsq_time;
};

class HybridTwoFocalPoseEstimator3 : public HybridTwoFocalPoseEstimator {
public:
    HybridTwoFocalPoseEstimator3(const std::vector<Eigen::Vector2d> &x0_norm, const std::vector<Eigen::Vector2d> &x1_norm,
                         const std::vector<double> &depth0, const std::vector<double> &depth1, 
                         const Eigen::Vector2d &min_depth, 
                         const double &norm_scale = 1.0,
                         const double &sampson_squared_weight = 1.0,
                         const std::vector<double> &squared_inlier_thresholds = {},
                         const EstimatorConfig &est_config = EstimatorConfig()) : 
                            HybridTwoFocalPoseEstimator(x0_norm, x1_norm, depth0, depth1, min_depth, 
                                norm_scale, sampson_squared_weight, squared_inlier_thresholds, est_config) {}

    void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const {
        min_sample_sizes->resize(2);
        min_sample_sizes->at(0) = {4, 4, 0};
        min_sample_sizes->at(1) = {0, 0, 7};
    }

    inline int num_data_types() const { return 3; }

    void num_data(std::vector<int>* num_data) const {
        num_data->resize(3);
        num_data->at(0) = x0_norm_.cols();
        num_data->at(1) = x0_norm_.cols();
        num_data->at(2) = x0_norm_.cols();
    }

    int MinimalSolver(const std::vector<std::vector<int>>& sample,
                      const int solver_idx, std::vector<PoseScaleOffsetTwoFocal>* models) const {
        std::vector<std::vector<int>> sample_2 = {sample[0], sample[2]};
        return HybridTwoFocalPoseEstimator::MinimalSolver(sample_2, solver_idx, models);
    }

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetTwoFocal* model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffsetTwoFocal& model, int t, int i, bool is_for_inlier=false) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetTwoFocal* model) const;
};

std::pair<PoseScaleOffsetTwoFocal, ransac_lib::HybridRansacStatistics> 
HybridEstimatePoseScaleOffsetTwoFocal(const std::vector<Eigen::Vector2d> &x0_norm, const std::vector<Eigen::Vector2d> &x1_norm,
                              const std::vector<double> &depth0, const std::vector<double> &depth1, 
                              const Eigen::Vector2d &min_depth, 
                              const Eigen::Vector2d &pp0, const Eigen::Vector2d &pp1,
                              const ExtendedHybridLORansacOptions& options, 
                              const EstimatorConfig &estimator_config = EstimatorConfig());

} // namespace acmpose

#endif // HYBRID_POSE_TWO_FOCAL_ESTIMATION_H
