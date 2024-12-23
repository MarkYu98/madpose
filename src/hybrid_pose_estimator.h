#ifndef HYBRID_POSE_ESTIMATION_H
#define HYBRID_POSE_ESTIMATION_H

#include <RansacLib/ransac.h>
#include "hybrid_ransac.h"
#include "pose_scale_shift_estimator.h"
#include "solver.h"
#include "optimizer.h"
#include "estimator_config.h"

namespace acmpose {

class HybridPoseEstimator {
public:
    HybridPoseEstimator(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                        const std::vector<double> &depth0, const std::vector<double> &depth1, 
                        const Eigen::Vector2d &min_depth, 
                        const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, 
                        const double &sampson_squared_weight = 1.0,
                        const std::vector<double> &squared_inlier_thresholds = {},
                        const EstimatorConfig &est_config = EstimatorConfig()) : 
                        K0_(K0), K1_(K1), sampson_squared_weight_(sampson_squared_weight),
                        K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), min_depth_(min_depth),  
                        squared_inlier_thresholds_(squared_inlier_thresholds), est_config_(est_config) {
        assert(x0.size() == x1.size() && x0.size() == depth0.size() && x0.size() == depth1.size());
        
        d0_ = Eigen::Map<const Eigen::VectorXd>(depth0.data(), depth0.size());
        d1_ = Eigen::Map<const Eigen::VectorXd>(depth1.data(), depth1.size());

        x0_ = Eigen::MatrixXd(3, x0.size());
        x1_ = Eigen::MatrixXd(3, x1.size());
        for (int i = 0; i < x0.size(); i++) {
            x0_.col(i) = x0[i].homogeneous();
            x1_.col(i) = x1[i].homogeneous();
        }
    }  

    ~HybridPoseEstimator() {}

    inline int num_minimal_solvers() const { return 2; }

    inline int min_sample_size() const { return 5; }

    void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const {
        min_sample_sizes->resize(2);
        min_sample_sizes->at(0) = {3, 0};
        min_sample_sizes->at(1) = {0, 5};
    }

    inline int num_data_types() const { return 2; }

    void num_data(std::vector<int>* num_data) const {
        num_data->resize(2);
        num_data->at(0) = x0_.cols();
        num_data->at(1) = x0_.cols();
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

    inline int non_minimal_sample_size() const { return 35; }

    int MinimalSolver(const std::vector<std::vector<int>>& sample,
                      const int solver_idx, std::vector<PoseScaleOffset>* models) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffset& model, int t, int i, bool is_for_inlier=false) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* model) const;

protected:
    Eigen::Matrix3d K0_, K1_;
    Eigen::Matrix3d K0_inv_, K1_inv_;
    Eigen::MatrixXd x0_, x1_;
    Eigen::VectorXd d0_, d1_;
    Eigen::Vector2d min_depth_; 
    double sampson_squared_weight_;

    EstimatorConfig est_config_;
    std::vector<double> squared_inlier_thresholds_;
};

class HybridPoseEstimator3 : public HybridPoseEstimator {
public:
    HybridPoseEstimator3(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                         const std::vector<double> &depth0, const std::vector<double> &depth1, 
                         const Eigen::Vector2d &min_depth, 
                         const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, 
                         const double &sampson_squared_weight = 1.0,
                         const std::vector<double> &squared_inlier_thresholds = {},
                         const EstimatorConfig &est_config = EstimatorConfig()) : 
                            HybridPoseEstimator(x0, x1, depth0, depth1, min_depth, 
                                K0, K1, sampson_squared_weight, squared_inlier_thresholds, est_config) {}

    void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const {
        min_sample_sizes->resize(2);
        min_sample_sizes->at(0) = {3, 3, 0};
        min_sample_sizes->at(1) = {0, 0, 5};
    }

    inline int num_data_types() const { return 3; }

    void num_data(std::vector<int>* num_data) const {
        num_data->resize(3);
        num_data->at(0) = x0_.cols();
        num_data->at(1) = x0_.cols();
        num_data->at(2) = x0_.cols();
    }

    int MinimalSolver(const std::vector<std::vector<int>>& sample,
                      const int solver_idx, std::vector<PoseScaleOffset>* models) const {
        std::vector<std::vector<int>> sample_2 = {sample[0], sample[2]};
        return HybridPoseEstimator::MinimalSolver(sample_2, solver_idx, models);
    }

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffset& model, int t, int i, bool is_for_inlier=false) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* model) const;
};

class HybridPoseEstimatorScaleOnly3 {
public:
    HybridPoseEstimatorScaleOnly3(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                                const std::vector<double> &depth0, const std::vector<double> &depth1, 
                                const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, 
                                const double &sampson_squared_weight = 1.0,
                                const std::vector<double> &squared_inlier_thresholds = {},
                                const EstimatorConfig &est_config = EstimatorConfig()) : 
                                K0_(K0), K1_(K1), sampson_squared_weight_(sampson_squared_weight),
                                K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), 
                                squared_inlier_thresholds_(squared_inlier_thresholds), est_config_(est_config) {
        assert(x0.size() == x1.size() && x0.size() == depth0.size() && x0.size() == depth1.size());
        
        d0_ = Eigen::Map<const Eigen::VectorXd>(depth0.data(), depth0.size());
        d1_ = Eigen::Map<const Eigen::VectorXd>(depth1.data(), depth1.size());

        x0_ = Eigen::MatrixXd(3, x0.size());
        x1_ = Eigen::MatrixXd(3, x1.size());
        for (int i = 0; i < x0.size(); i++) {
            x0_.col(i) = x0[i].homogeneous();
            x1_.col(i) = x1[i].homogeneous();
        }
    }  

    ~HybridPoseEstimatorScaleOnly3() {}

    inline int num_minimal_solvers() const { return 2; }

    inline int min_sample_size() const { return 5; }

    void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const {
        min_sample_sizes->resize(2);
        min_sample_sizes->at(0) = {3, 3, 0};
        min_sample_sizes->at(1) = {0, 0, 5};
    }

    inline int num_data_types() const { return 3; }

    void num_data(std::vector<int>* num_data) const {
        num_data->resize(3);
        num_data->at(0) = x0_.cols();
        num_data->at(1) = x0_.cols();
        num_data->at(2) = x0_.cols();
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

    inline int non_minimal_sample_size() const { return 35; }

    int MinimalSolver(const std::vector<std::vector<int>>& sample,
                      const int solver_idx, std::vector<PoseAndScale>* models) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseAndScale* model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseAndScale& model, int t, int i, bool is_for_inlier=false) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseAndScale* model) const;

protected:
    Eigen::Matrix3d K0_, K1_;
    Eigen::Matrix3d K0_inv_, K1_inv_;
    Eigen::MatrixXd x0_, x1_;
    Eigen::VectorXd d0_, d1_;
    double sampson_squared_weight_;

    EstimatorConfig est_config_;
    std::vector<double> squared_inlier_thresholds_;
};

class PoseAndScaleOptimizer3 {
protected:
    const Eigen::Matrix3d &K0_, &K1_, K0_inv_, K1_inv_;
    const Eigen::MatrixXd &x0_, &x1_;
    const Eigen::VectorXd &d0_, &d1_;
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;
    double scale_, offset0_, offset1_;
    OptimizerConfig config_;

    const std::vector<int> &indices_reproj_0_, &indices_reproj_1_;
    const std::vector<int> &indices_sampson_;

    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;

public:
    PoseAndScaleOptimizer3(const Eigen::MatrixXd &x0, const Eigen::MatrixXd &x1, 
                     const Eigen::VectorXd &depth0, const Eigen::VectorXd &depth1,
                     const std::vector<int> &indices_reproj_0, const std::vector<int> &indices_reproj_1,
                     const std::vector<int> &indices_sampson,
                     const PoseAndScale &pose, 
                     const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, 
                     const OptimizerConfig& config = OptimizerConfig()) : 
                     K0_(K0), K1_(K1), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), 
                     x0_(x0), x1_(x1), d0_(depth0), d1_(depth1),
                     indices_reproj_0_(indices_reproj_0), indices_reproj_1_(indices_reproj_1), indices_sampson_(indices_sampson),
                     config_(config) {
        qvec_ = RotationMatrixToQuaternion<double>(pose.R());
        tvec_ = pose.t();
        offset0_ = 0;
        offset1_ = 0;
        scale_ = pose.scale;

        if (config_.geom_loss_function.get() == nullptr)
            config_.geom_loss_function.reset(new ceres::TrivialLoss());
        if (config_.reproj_loss_function.get() == nullptr)
            config_.reproj_loss_function.reset(new ceres::TrivialLoss());
        if (config_.sampson_loss_function.get() == nullptr)
            config_.sampson_loss_function.reset(new ceres::TrivialLoss());
    }

    void SetUp() {
        problem_.reset(new ceres::Problem(config_.problem_options));

        ceres::LossFunction* geo_loss_func = config_.geom_loss_function.get();
        ceres::LossFunction* proj_loss_func = config_.reproj_loss_function.get();
        ceres::LossFunction* sampson_loss_func = config_.sampson_loss_function.get();

        if (config_.use_reprojection) {
            for (auto &i : indices_reproj_0_) {
                ceres::CostFunction* reproj_cost_0 = LiftProjectionFunctor0::Create(
                    K0_inv_ * x0_.col(i), x1_.col(i), d0_(i), K1_);
                problem_->AddResidualBlock(reproj_cost_0, proj_loss_func, &offset0_, qvec_.data(), tvec_.data());
            }
            for (auto &i : indices_reproj_1_) {
                ceres::CostFunction* reproj_cost_1 = LiftProjectionFunctor1::Create(
                    K1_inv_ * x1_.col(i), x0_.col(i), d1_(i), K0_);
                problem_->AddResidualBlock(reproj_cost_1, proj_loss_func, &scale_, &offset1_, qvec_.data(), tvec_.data());
            }
        }

        if (config_.use_sampson) {
            for (auto &i : indices_sampson_) {
                Eigen::Vector3d x0 = K0_inv_ * x0_.col(i);
                Eigen::Vector3d x1 = K1_inv_ * x1_.col(i);
                ceres::CostFunction* sampson_cost = SampsonErrorFunctor::Create(x0, x1, K0_, K1_, config_.weight_sampson);
                problem_->AddResidualBlock(sampson_cost, sampson_loss_func, qvec_.data(), tvec_.data());
            }
        }

        if (problem_->HasParameterBlock(&scale_)) {
            problem_->SetParameterLowerBound(&scale_, 0, 1e-2); // scale >= 0
        }

        if (problem_->HasParameterBlock(&offset0_)) 
            problem_->SetParameterBlockConstant(&offset0_);
        if (problem_->HasParameterBlock(&offset1_))
            problem_->SetParameterBlockConstant(&offset1_);

        if (problem_->HasParameterBlock(qvec_.data())) {
            if (config_.constant_pose) {
                problem_->SetParameterBlockConstant(qvec_.data());
                problem_->SetParameterBlockConstant(tvec_.data());
            }
            else {
            #ifdef CERES_PARAMETERIZATION_ENABLED
                ceres::LocalParameterization* quaternion_parameterization = 
                    new ceres::QuaternionParameterization;
                problem_->SetParameterization(qvec_.data(), quaternion_parameterization);
            #else
                ceres::Manifold* quaternion_manifold = 
                    new ceres::QuaternionManifold;
                problem_->SetManifold(qvec_.data(), quaternion_manifold);
            #endif
            }
        }
    }

    bool Solve() {
        if (problem_->NumResiduals() == 0) return false;
        ceres::Solver::Options solver_options = config_.solver_options;
    
        solver_options.linear_solver_type = ceres::DENSE_QR;

        solver_options.num_threads = 1; 
        // colmap::GetEffectiveNumThreads(solver_options.num_threads);
        #if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads =
            colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
        #endif  // CERES_VERSION_MAJOR
        
        std::string solver_error;
        CHECK(solver_options.IsValid(&solver_error)) << solver_error;

        ceres::Solve(solver_options, problem_.get(), &summary_);
        return true;
    }

    PoseAndScale GetSolution() {
        Eigen::Matrix3d R = QuaternionToRotationMatrix<double>(qvec_);
        return PoseAndScale(R, tvec_, scale_);
    }
};

std::pair<PoseScaleOffset, ransac_lib::HybridRansacStatistics> 
HybridEstimatePoseScaleOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                              const std::vector<double> &depth0, const std::vector<double> &depth1, 
                              const Eigen::Vector2d &min_depth, 
                              const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                              const ExtendedHybridLORansacOptions& options, 
                              const EstimatorConfig &est_config = EstimatorConfig());

std::pair<PoseAndScale, ransac_lib::HybridRansacStatistics>
HybridEstimatePoseAndScale(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                           const std::vector<double> &depth0, const std::vector<double> &depth1, 
                           const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                           const ExtendedHybridLORansacOptions& options, 
                           const EstimatorConfig &est_config = EstimatorConfig());

} // namespace acmpose

#endif // HYBRID_POSE_ESTIMATION_H