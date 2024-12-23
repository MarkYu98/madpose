#ifndef POSE_SCALE_SHIFT_ESTIMATOR_H
#define POSE_SCALE_SHIFT_ESTIMATOR_H

#include <RansacLib/ransac.h>
#include "optimizer.h"

namespace acmpose {

class PoseScaleOffsetEstimator {
public:
    PoseScaleOffsetEstimator(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                            const std::vector<double> &depth0, const std::vector<double> &depth1, 
                            const Eigen::Vector2d &min_depth,
                            const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1) : 
                            K0_(K0), K1_(K1), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), min_depth_(min_depth) { 
                            assert(x0.size() == x1.size() && x0.size() == depth0.size() && x0.size() == depth1.size());
                            
                            d0_ = Eigen::Map<const Eigen::VectorXd>(depth0.data(), depth0.size());
                            d1_ = Eigen::Map<const Eigen::VectorXd>(depth1.data(), depth1.size());

                            x0_ = Eigen::MatrixXd(3, x0.size());
                            x1_ = Eigen::MatrixXd(3, x1.size());
                            for (int i = 0; i < x0.size(); i++) {
                                x0_.col(i) = Eigen::Vector3d(x0[i](0), x0[i](1), 1.0);
                                x1_.col(i) = Eigen::Vector3d(x1[i](0), x1[i](1), 1.0);
                            }
                         }  

    inline int min_sample_size() const { return 3; }

    inline int non_minimal_sample_size() const { return 10; }

    inline int num_data() const { return x0_.cols(); }

    int MinimalSolver(const std::vector<int>& sample, std::vector<PoseScaleOffset>* solutions) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<int>& sample, PoseScaleOffset* solution) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffset& solution, int i) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<int>& sample, PoseScaleOffset* solution) const;
    
    const size_t sample_sz = 3;
    double squared_inlier_thres;

private:
    Eigen::Matrix3d K0_, K1_;
    Eigen::Matrix3d K0_inv_, K1_inv_;
    Eigen::MatrixXd x0_, x1_;
    Eigen::VectorXd d0_, d1_;
    Eigen::Vector2d min_depth_;
};

class PoseScaleOffsetOptimizer {
protected:
    const Eigen::Matrix3d &K0_, &K1_, K0_inv_, K1_inv_;
    const Eigen::MatrixXd &x0_, &x1_;
    const Eigen::VectorXd &d0_, &d1_;
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;
    double scale_, offset0_, offset1_;
    double min_depth_0_, min_depth_1_;
    OptimizerConfig config_;

    const std::vector<int> &indices_reproj_;
    const std::vector<int> &indices_sampson_;

    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
public:
    PoseScaleOffsetOptimizer(const Eigen::MatrixXd &x0, const Eigen::MatrixXd &x1, const Eigen::VectorXd &depth0, const Eigen::VectorXd &depth1,
                     const std::vector<int> &indices_reproj, const std::vector<int> &indices_sampson,
                     const double &min_depth_0, const double &min_depth_1, 
                     const PoseScaleOffset &pose, 
                     const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                     const OptimizerConfig& config = OptimizerConfig()) : 
                     K0_(K0), K1_(K1), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), 
                     x0_(x0), x1_(x1), d0_(depth0), d1_(depth1), 
                     indices_reproj_(indices_reproj), indices_sampson_(indices_sampson),
                     min_depth_0_(min_depth_0), min_depth_1_(min_depth_1), 
                     config_(config) {
        qvec_ = RotationMatrixToQuaternion<double>(pose.R());
        tvec_ = pose.t();
        offset0_ = pose.offset0;
        offset1_ = pose.offset1;
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


        for (auto &i : indices_reproj_) {
            if (config_.use_reprojection) {
                Eigen::Vector3d x0_normalized = K0_inv_ * x0_.col(i);
                Eigen::Vector3d x1_normalized = K1_inv_ * x1_.col(i);
                x0_normalized = x0_normalized / x0_normalized[2];
                x1_normalized = x1_normalized / x1_normalized[2];

                ceres::CostFunction* reproj_cost = LiftProjectionFunctor::Create(
                    x0_.col(i), x1_.col(i), d0_(i), d1_(i), K0_, K1_, config_.squared_cost);
                problem_->AddResidualBlock(reproj_cost, proj_loss_func, &scale_, &offset0_, &offset1_, qvec_.data(), tvec_.data());
            }
            if (config_.use_geometric) {
                ceres::CostFunction* geo_cost = LiftGeometryFunctor::Create(
                    x0_.col(i), x1_.col(i), d0_(i), d1_(i), K0_, K1_, config_.weight_geometric);
                problem_->AddResidualBlock(geo_cost, geo_loss_func, &scale_, &offset0_, &offset1_, qvec_.data(), tvec_.data());
            }
        }

        for (auto &i : indices_sampson_) {
            if (config_.use_sampson) {
                Eigen::Vector3d x0 = K0_inv_ * x0_.col(i);
                Eigen::Vector3d x1 = K1_inv_ * x1_.col(i);
                ceres::CostFunction* sampson_cost = SampsonErrorFunctor::Create(x0, x1, K0_, K1_, config_.weight_sampson);
                problem_->AddResidualBlock(sampson_cost, sampson_loss_func, qvec_.data(), tvec_.data());
            }
        }

        if (problem_->HasParameterBlock(&scale_)) {
            problem_->SetParameterLowerBound(&scale_, 0, 1e-2); // scale >= 0.01
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset0_)) {
            problem_->SetParameterLowerBound(&offset0_, 0, -min_depth_0_ + 1e-2); // offset0 >= -min_depth_(0)
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset1_)) {
            problem_->SetParameterLowerBound(&offset1_, 0, -min_depth_1_ + 1e-2); // offset1 >= -min_depth_(1)
        }
        if (!config_.use_shift) {
            if (problem_->HasParameterBlock(&offset0_)) problem_->SetParameterBlockConstant(&offset0_);
            if (problem_->HasParameterBlock(&offset1_)) problem_->SetParameterBlockConstant(&offset1_);
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

    PoseScaleOffset GetSolution() {
        Eigen::Matrix3d R = QuaternionToRotationMatrix<double>(qvec_);
        return PoseScaleOffset(R, tvec_, scale_, offset0_, offset1_);
    }
};

class PoseScaleOffsetOptimizer3 {
protected:
    const Eigen::Matrix3d &K0_, &K1_, K0_inv_, K1_inv_;
    const Eigen::MatrixXd &x0_, &x1_;
    const Eigen::VectorXd &d0_, &d1_;
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;
    double scale_, offset0_, offset1_;
    Eigen::Vector2d min_depth_; 
    OptimizerConfig config_;

    const std::vector<int> &indices_reproj_0_, &indices_reproj_1_;
    const std::vector<int> &indices_sampson_;

    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;

public:
    PoseScaleOffsetOptimizer3(const Eigen::MatrixXd &x0, const Eigen::MatrixXd &x1, 
                     const Eigen::VectorXd &depth0, const Eigen::VectorXd &depth1,
                     const std::vector<int> &indices_reproj_0, const std::vector<int> &indices_reproj_1,
                     const std::vector<int> &indices_sampson,
                     const Eigen::Vector2d &min_depth,
                     const PoseScaleOffset &pose, 
                     const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                     const OptimizerConfig& config = OptimizerConfig()) : 
                     K0_(K0), K1_(K1), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), 
                     x0_(x0), x1_(x1), d0_(depth0), d1_(depth1), 
                     indices_reproj_0_(indices_reproj_0), indices_reproj_1_(indices_reproj_1), indices_sampson_(indices_sampson),
                     min_depth_(min_depth), config_(config) {
        qvec_ = RotationMatrixToQuaternion<double>(pose.R());
        tvec_ = pose.t();
        offset0_ = pose.offset0;
        offset1_ = pose.offset1;
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

    PoseScaleOffset GetSolution() {
        Eigen::Matrix3d R = QuaternionToRotationMatrix<double>(qvec_);
        return PoseScaleOffset(R, tvec_, scale_, offset0_, offset1_);
    }
};

std::pair<PoseScaleOffset, ransac_lib::RansacStatistics>
EstimatePoseScaleOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                        const std::vector<double> &depth0, const std::vector<double> &depth1, const Eigen::Vector2d &min_depth,
                        const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                        const ransac_lib::LORansacOptions& options);

} // namespace acmpose

#endif // POSE_SCALE_SHIFT_ESTIMATOR_H