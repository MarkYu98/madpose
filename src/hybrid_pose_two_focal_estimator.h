#ifndef HYBRID_POSE_TWO_FOCAL_ESTIMATION_H
#define HYBRID_POSE_TWO_FOCAL_ESTIMATION_H

#include <RansacLib/ransac.h>
#include "optimizer.h"
#include "hybrid_ransac.h"
#include "solver.h"
#include "estimator_config.h"
#include <iostream>

namespace acmpose {

class HybridPoseTwoFocalEstimator {
public:
    HybridPoseTwoFocalEstimator(const std::vector<Eigen::Vector2d> &x0_norm, const std::vector<Eigen::Vector2d> &x1_norm,
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

    ~HybridPoseTwoFocalEstimator() {}

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

class HybridPoseTwoFocalEstimator3 : public HybridPoseTwoFocalEstimator {
public:
    HybridPoseTwoFocalEstimator3(const std::vector<Eigen::Vector2d> &x0_norm, const std::vector<Eigen::Vector2d> &x1_norm,
                         const std::vector<double> &depth0, const std::vector<double> &depth1, 
                         const Eigen::Vector2d &min_depth, 
                         const double &norm_scale = 1.0,
                         const double &sampson_squared_weight = 1.0,
                         const std::vector<double> &squared_inlier_thresholds = {},
                         const EstimatorConfig &est_config = EstimatorConfig()) : 
                            HybridPoseTwoFocalEstimator(x0_norm, x1_norm, depth0, depth1, min_depth, 
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
        return HybridPoseTwoFocalEstimator::MinimalSolver(sample_2, solver_idx, models);
    }

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetTwoFocal* model) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffsetTwoFocal& model, int t, int i, bool is_for_inlier=false) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetTwoFocal* model) const;
};

class TwoFocalOptimizer3 {
protected:
    const Eigen::MatrixXd &x0_, &x1_;
    const Eigen::VectorXd &d0_, &d1_;
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;
    double focal0_, focal1_;
    double scale_, offset0_, offset1_;
    Eigen::Vector2d min_depth_;
    TwoFocalOptimizerConfig config_;

    const std::vector<int> &indices_reproj_0_, &indices_reproj_1_;
    const std::vector<int> &indices_sampson_;

    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
public:
    TwoFocalOptimizer3(const Eigen::MatrixXd &x0, const Eigen::MatrixXd &x1, const Eigen::VectorXd &depth0, const Eigen::VectorXd &depth1,
                          const std::vector<int> &indices_reproj_0, const std::vector<int> &indices_reproj_1, const std::vector<int> &indices_sampson,
                          const Eigen::Vector2d &min_depth, 
                          const PoseScaleOffsetTwoFocal &pose, 
                          const TwoFocalOptimizerConfig& config = TwoFocalOptimizerConfig()) : 
                     x0_(x0), x1_(x1), d0_(depth0), d1_(depth1),
                     indices_reproj_0_(indices_reproj_0), indices_reproj_1_(indices_reproj_1), indices_sampson_(indices_sampson),
                     min_depth_(min_depth), config_(config) {
        qvec_ = RotationMatrixToQuaternion<double>(pose.R());
        tvec_ = pose.t();
        offset0_ = pose.offset0;
        offset1_ = pose.offset1;
        scale_ = pose.scale;
        focal0_ = pose.focal0;
        focal1_ = pose.focal1;

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
        
        // geo_loss_func = new ceres::ScaledLoss(geo_loss_func, config_.weight_geometric, ceres::DO_NOT_TAKE_OWNERSHIP);
        // sampson_loss_func = new ceres::ScaledLoss(sampson_loss_func, config_.weight_sampson, ceres::DO_NOT_TAKE_OWNERSHIP);

        if (config_.use_reprojection) {
            for (auto &i : indices_reproj_0_) {
                ceres::LossFunction* weighted_loss = new ceres::ScaledLoss(proj_loss_func, 1.0, ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::CostFunction* reproj_cost_0 = LiftProjectionTwoFocalFunctor0::Create(x0_.col(i), x1_.col(i), d0_(i));
                problem_->AddResidualBlock(reproj_cost_0, weighted_loss, &offset0_, qvec_.data(), tvec_.data(), &focal0_, &focal1_);
            }
            for (auto &i : indices_reproj_1_) {
                ceres::LossFunction* weighted_loss = new ceres::ScaledLoss(proj_loss_func, 1.0, ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::CostFunction* reproj_cost_1 = LiftProjectionTwoFocalFunctor1::Create(x1_.col(i), x0_.col(i), d1_(i));
                problem_->AddResidualBlock(reproj_cost_1, weighted_loss, &scale_, &offset1_, qvec_.data(), tvec_.data(), &focal0_, &focal1_);
            }
        }

        for (auto &i : indices_sampson_) {
            if (config_.use_sampson) {
                ceres::LossFunction* weighted_loss = new ceres::ScaledLoss(sampson_loss_func, 1.0, ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::CostFunction* sampson_cost = SampsonErrorTwoFocalFunctor::Create(x0_.col(i), x1_.col(i), config_.weight_sampson);
                problem_->AddResidualBlock(sampson_cost, weighted_loss, qvec_.data(), tvec_.data(), &focal0_, &focal1_);
            }
        }

        if (problem_->HasParameterBlock(&scale_)) {
            problem_->SetParameterLowerBound(&scale_, 0, 1e-2); // scale >= 0
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset0_)) {
            problem_->SetParameterLowerBound(&offset0_, 0, -min_depth_(0) + 1e-2); // offset0 >= -min_depth_(0)
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset1_)) {
            problem_->SetParameterLowerBound(&offset1_, 0, -min_depth_(1) + 1e-2); // offset1 >= -min_depth_(1)
        }
        if (!config_.use_shift) {
            if (problem_->HasParameterBlock(&offset0_)) problem_->SetParameterBlockConstant(&offset0_);
            if (problem_->HasParameterBlock(&offset1_)) problem_->SetParameterBlockConstant(&offset1_);
        }

        if (problem_->HasParameterBlock(&focal0_)) {
            problem_->SetParameterLowerBound(&focal0_, 0, 1e-6); // focal0 >= 0
            problem_->SetParameterLowerBound(&focal1_, 0, 1e-6); // focal1 >= 0
        }

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

    PoseScaleOffsetTwoFocal GetSolution() {
        Eigen::Matrix3d R = QuaternionToRotationMatrix<double>(qvec_);
        return PoseScaleOffsetTwoFocal(R, tvec_, scale_, offset0_, offset1_, focal0_, focal1_);
    }
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
