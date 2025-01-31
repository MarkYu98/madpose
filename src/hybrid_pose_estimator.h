#pragma once

#include "estimator_config.h"
#include "hybrid_ransac.h"
#include "optimizer.h"
#include "solver.h"

#include <RansacLib/ransac.h>

namespace madpose {

class HybridPoseEstimator {
  public:
    HybridPoseEstimator(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                        const std::vector<double> &depth0, const std::vector<double> &depth1,
                        const Eigen::Vector2d &min_depth, const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                        const double &sampson_squared_weight = 1.0,
                        const std::vector<double> &squared_inlier_thresholds = {},
                        const EstimatorConfig &est_config = EstimatorConfig())
        : K0_(K0), K1_(K1), sampson_squared_weight_(sampson_squared_weight), K0_inv_(K0.inverse()),
          K1_inv_(K1.inverse()), min_depth_(min_depth), squared_inlier_thresholds_(squared_inlier_thresholds),
          est_config_(est_config) {
        assert(x0.size() == x1.size() && x0.size() == depth0.size() && x0.size() == depth1.size());

        d0_ = Eigen::Map<const Eigen::VectorXd>(depth0.data(), depth0.size());
        d1_ = Eigen::Map<const Eigen::VectorXd>(depth1.data(), depth1.size());

        x0_ = Eigen::MatrixXd(3, x0.size());
        x1_ = Eigen::MatrixXd(3, x1.size());
        for (int i = 0; i < x0.size(); i++) {
            x0_.col(i) = x0[i].homogeneous();
            x1_.col(i) = x1[i].homogeneous();
        }

        sampson_loss_scale_ = 1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1));
        sampson_loss_scale_ = 1.0 / std::pow(sampson_loss_scale_, 2);
    }

    ~HybridPoseEstimator() {}

    inline int num_minimal_solvers() const { return 2; }

    inline int min_sample_size() const { return 5; }

    void min_sample_sizes(std::vector<std::vector<int>> *min_sample_sizes) const {
        min_sample_sizes->resize(2);
        min_sample_sizes->at(0) = {3, 3, 0};
        min_sample_sizes->at(1) = {0, 0, 5};
    }

    inline int num_data_types() const { return 3; }

    void num_data(std::vector<int> *num_data) const {
        num_data->resize(3);
        num_data->at(0) = x0_.cols();
        num_data->at(1) = x0_.cols();
        num_data->at(2) = x0_.cols();
    }

    void solver_probabilities(std::vector<double> *probabilities) const {
        probabilities->resize(2);
        probabilities->at(0) = 1.0;
        probabilities->at(1) = 1.0;
        if (est_config_.solver_type == EstimatorOption::EPI_ONLY) {
            probabilities->at(0) = 0.0;
        } else if (est_config_.solver_type == EstimatorOption::MD_ONLY) {
            probabilities->at(1) = 0.0;
        }
    }

    inline int non_minimal_sample_size() const { return 35; }

    int MinimalSolver(const std::vector<std::vector<int>> &sample, const int solver_idx,
                      std::vector<PoseScaleOffset> *models) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>> &sample, const int solver_idx,
                         PoseScaleOffset *model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffset &model, int t, int i, bool is_for_inlier = false) const;

    // Linear least squares solver.
    void LeastSquares(const std::vector<std::vector<int>> &sample, const int solver_idx, PoseScaleOffset *model) const;

  protected:
    Eigen::Matrix3d K0_, K1_;
    Eigen::Matrix3d K0_inv_, K1_inv_;
    Eigen::MatrixXd x0_, x1_;
    Eigen::VectorXd d0_, d1_;
    Eigen::Vector2d min_depth_;
    double sampson_squared_weight_;
    double sampson_loss_scale_;

    EstimatorConfig est_config_;
    std::vector<double> squared_inlier_thresholds_;
};

class HybridPoseEstimatorScaleOnly {
  public:
    HybridPoseEstimatorScaleOnly(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                                 const std::vector<double> &depth0, const std::vector<double> &depth1,
                                 const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                                 const double &sampson_squared_weight = 1.0,
                                 const std::vector<double> &squared_inlier_thresholds = {},
                                 const EstimatorConfig &est_config = EstimatorConfig())
        : K0_(K0), K1_(K1), sampson_squared_weight_(sampson_squared_weight), K0_inv_(K0.inverse()),
          K1_inv_(K1.inverse()), squared_inlier_thresholds_(squared_inlier_thresholds), est_config_(est_config) {
        assert(x0.size() == x1.size() && x0.size() == depth0.size() && x0.size() == depth1.size());

        d0_ = Eigen::Map<const Eigen::VectorXd>(depth0.data(), depth0.size());
        d1_ = Eigen::Map<const Eigen::VectorXd>(depth1.data(), depth1.size());

        x0_ = Eigen::MatrixXd(3, x0.size());
        x1_ = Eigen::MatrixXd(3, x1.size());
        for (int i = 0; i < x0.size(); i++) {
            x0_.col(i) = x0[i].homogeneous();
            x1_.col(i) = x1[i].homogeneous();
        }

        sampson_loss_scale_ = 1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1));
        sampson_loss_scale_ = 1.0 / std::pow(sampson_loss_scale_, 2);
    }

    ~HybridPoseEstimatorScaleOnly() {}

    inline int num_minimal_solvers() const { return 2; }

    inline int min_sample_size() const { return 5; }

    void min_sample_sizes(std::vector<std::vector<int>> *min_sample_sizes) const {
        min_sample_sizes->resize(2);
        min_sample_sizes->at(0) = {3, 3, 0};
        min_sample_sizes->at(1) = {0, 0, 5};
    }

    inline int num_data_types() const { return 3; }

    void num_data(std::vector<int> *num_data) const {
        num_data->resize(3);
        num_data->at(0) = x0_.cols();
        num_data->at(1) = x0_.cols();
        num_data->at(2) = x0_.cols();
    }

    void solver_probabilities(std::vector<double> *probabilities) const {
        probabilities->resize(2);
        probabilities->at(0) = 1.0;
        probabilities->at(1) = 1.0;
        if (est_config_.solver_type == EstimatorOption::EPI_ONLY) {
            probabilities->at(0) = 0.0;
        } else if (est_config_.solver_type == EstimatorOption::MD_ONLY) {
            probabilities->at(1) = 0.0;
        }
    }

    inline int non_minimal_sample_size() const { return 35; }

    int MinimalSolver(const std::vector<std::vector<int>> &sample, const int solver_idx,
                      std::vector<PoseAndScale> *models) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>> &sample, const int solver_idx, PoseAndScale *model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseAndScale &model, int t, int i, bool is_for_inlier = false) const;

    // Linear least squares solver.
    void LeastSquares(const std::vector<std::vector<int>> &sample, const int solver_idx, PoseAndScale *model) const;

  protected:
    Eigen::Matrix3d K0_, K1_;
    Eigen::Matrix3d K0_inv_, K1_inv_;
    Eigen::MatrixXd x0_, x1_;
    Eigen::VectorXd d0_, d1_;
    double sampson_squared_weight_;

    EstimatorConfig est_config_;
    std::vector<double> squared_inlier_thresholds_;
};

std::pair<PoseScaleOffset, ransac_lib::HybridRansacStatistics>
HybridEstimatePoseScaleOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                              const std::vector<double> &depth0, const std::vector<double> &depth1,
                              const Eigen::Vector2d &min_depth, const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                              const ExtendedHybridLORansacOptions &options,
                              const EstimatorConfig &est_config = EstimatorConfig());

std::pair<PoseAndScale, ransac_lib::HybridRansacStatistics> HybridEstimatePoseAndScale(
    const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1, const std::vector<double> &depth0,
    const std::vector<double> &depth1, const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
    const ExtendedHybridLORansacOptions &options, const EstimatorConfig &est_config = EstimatorConfig());

} // namespace madpose
