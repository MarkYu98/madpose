#ifndef HYBRID_POSE_SHARED_FOCAL_ESTIMATION_H
#define HYBRID_POSE_SHARED_FOCAL_ESTIMATION_H

#include <RansacLib/ransac.h>
#include "optimizer.h"
#include "hybrid_ransac.h"
#include "solver.h"
#include "estimator_config.h"

namespace madpose {

class HybridSharedFocalPoseEstimator {
public:
    HybridSharedFocalPoseEstimator(const std::vector<Eigen::Vector2d> &x0_norm, const std::vector<Eigen::Vector2d> &x1_norm,
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
                        }  

    ~HybridSharedFocalPoseEstimator() {}

    inline int num_minimal_solvers() const { return 2; }

    inline int min_sample_size() const { return 6; }

    void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const {
        min_sample_sizes->resize(2);
        min_sample_sizes->at(0) = {4, 4, 0};
        min_sample_sizes->at(1) = {0, 0, 6};
    }

    inline int num_data_types() const { return 3; }

    void num_data(std::vector<int>* num_data) const {
        num_data->resize(3);
        num_data->at(0) = x0_norm_.cols();
        num_data->at(1) = x0_norm_.cols();
        num_data->at(2) = x0_norm_.cols();
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
                      const int solver_idx, std::vector<PoseScaleOffsetSharedFocal>* models) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetSharedFocal* model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffsetSharedFocal& model, int t, int i, bool is_for_inlier=false) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetSharedFocal* model) const;

protected:
    Eigen::MatrixXd x0_norm_, x1_norm_;
    Eigen::VectorXd d0_, d1_;
    Eigen::Vector2d min_depth_; 
    double sampson_squared_weight_;

    EstimatorConfig est_config_;

    std::vector<double> squared_inlier_thresholds_;
    double norm_scale_;
};

std::pair<PoseScaleOffsetSharedFocal, ransac_lib::HybridRansacStatistics> 
HybridEstimatePoseScaleOffsetSharedFocal(const std::vector<Eigen::Vector2d> &x0_norm, const std::vector<Eigen::Vector2d> &x1_norm,
                              const std::vector<double> &depth0, const std::vector<double> &depth1, 
                              const Eigen::Vector2d &min_depth,
                              const Eigen::Vector2d &pp0, const Eigen::Vector2d &pp1,
                              const ExtendedHybridLORansacOptions& options, 
                              const EstimatorConfig &est_config = EstimatorConfig());

} // namespace madpose

#endif // HYBRID_POSE_SHARED_FOCAL_ESTIMATION_H
