#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "utils.h"
#include "pose.h"
#include "cost_functions.h"

namespace acmpose {

class OptimizerConfig {
public:
    OptimizerConfig() {
        solver_options.function_tolerance = 1e-6;
        solver_options.gradient_tolerance = 1e-10;
        solver_options.parameter_tolerance = 1e-8;
        solver_options.minimizer_progress_to_stdout = true;
        solver_options.max_num_iterations = 100;
        solver_options.use_nonmonotonic_steps = true;
        solver_options.num_threads = -1;
        solver_options.logging_type = ceres::SILENT;
    #if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads = -1;
    #endif  // CERES_VERSION_MAJOR
        problem_options.loss_function_ownership = ceres::TAKE_OWNERSHIP;
        problem_options.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    }

    OptimizerConfig(py::dict dict): OptimizerConfig() {
        ASSIGN_PYDICT_ITEM(dict, constant_pose, bool);
        ASSIGN_PYDICT_ITEM(dict, constant_scale, bool);
        ASSIGN_PYDICT_ITEM(dict, constant_offset, bool);
        if (dict.contains("solver_options"))
            AssignSolverOptionsFromDict(solver_options, dict["solver_options"]);
    }
    ceres::Solver::Options solver_options;

    bool constant_pose = false;
    bool constant_scale = false;
    bool constant_offset = false;
    bool use_reprojection = true;
    bool use_sampson = false;
    bool min_depth_constraint = true;
    bool use_shift = true;

    bool squared_cost = false;

    double weight_geometric = 1.0;
    double weight_sampson = 1.0;
    std::shared_ptr<ceres::LossFunction> reproj_loss_function;
    std::shared_ptr<ceres::LossFunction> geom_loss_function;
    std::shared_ptr<ceres::LossFunction> sampson_loss_function;

    // These are not set from py::dict;
    ceres::Problem::Options problem_options;
};

class SharedFocalOptimizerConfig : public OptimizerConfig {
public:
    SharedFocalOptimizerConfig() : OptimizerConfig() {}
    bool constant_focal = false;
};

typedef SharedFocalOptimizerConfig TwoFocalOptimizerConfig;


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

} // namespace acmpose

#endif // OPTIMIZER_H

