#pragma once

#include "cost_functions.h"
#include "optimizer_config.h"
#include "pose.h"

namespace madpose {

class HybridPoseOptimizer {
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
    HybridPoseOptimizer(const Eigen::MatrixXd &x0, const Eigen::MatrixXd &x1, const Eigen::VectorXd &depth0,
                        const Eigen::VectorXd &depth1, const std::vector<int> &indices_reproj_0,
                        const std::vector<int> &indices_reproj_1, const std::vector<int> &indices_sampson,
                        const Eigen::Vector2d &min_depth, const PoseScaleOffset &pose, const Eigen::Matrix3d &K0,
                        const Eigen::Matrix3d &K1, const OptimizerConfig &config = OptimizerConfig())
        : K0_(K0), K1_(K1), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), x0_(x0), x1_(x1), d0_(depth0), d1_(depth1),
          indices_reproj_0_(indices_reproj_0), indices_reproj_1_(indices_reproj_1), indices_sampson_(indices_sampson),
          min_depth_(min_depth), config_(config) {
        qvec_ = RotationMatrixToQuaternion<double>(pose.R());
        tvec_ = pose.t();
        offset0_ = pose.offset0;
        offset1_ = pose.offset1;
        scale_ = pose.scale;

        if (config_.reproj_loss_function.get() == nullptr)
            config_.reproj_loss_function.reset(new ceres::TrivialLoss());
        if (config_.sampson_loss_function.get() == nullptr)
            config_.sampson_loss_function.reset(new ceres::TrivialLoss());
    }

    void SetUp() {
        problem_.reset(new ceres::Problem(config_.problem_options));

        ceres::LossFunction *proj_loss_func = config_.reproj_loss_function.get();
        ceres::LossFunction *sampson_loss_func = config_.sampson_loss_function.get();

        if (config_.use_reprojection) {
            for (auto &i : indices_reproj_0_) {
                ceres::CostFunction *reproj_cost_0 =
                    LiftProjectionFunctor0::Create(K0_inv_ * x0_.col(i), x1_.col(i), d0_(i), K1_);
                problem_->AddResidualBlock(reproj_cost_0, proj_loss_func, &offset0_, qvec_.data(), tvec_.data());
            }
            for (auto &i : indices_reproj_1_) {
                ceres::CostFunction *reproj_cost_1 =
                    LiftProjectionFunctor1::Create(K1_inv_ * x1_.col(i), x0_.col(i), d1_(i), K0_);
                problem_->AddResidualBlock(reproj_cost_1, proj_loss_func, &scale_, &offset1_, qvec_.data(),
                                           tvec_.data());
            }
        }

        if (config_.use_sampson) {
            for (auto &i : indices_sampson_) {
                Eigen::Vector3d x0 = K0_inv_ * x0_.col(i);
                Eigen::Vector3d x1 = K1_inv_ * x1_.col(i);
                ceres::CostFunction *sampson_cost = SampsonErrorFunctor::Create(x0, x1, weight);
                problem_->AddResidualBlock(sampson_cost, sampson_loss_func, qvec_.data(), tvec_.data());
            }
        }

        if (problem_->HasParameterBlock(&scale_)) {
            problem_->SetParameterLowerBound(&scale_, 0, 1e-2); // scale >= 0
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset0_)) {
            problem_->SetParameterLowerBound(&offset0_, 0,
                                             -min_depth_(0) + 1e-2); // offset0 >= -min_depth_(0)
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset1_)) {
            problem_->SetParameterLowerBound(&offset1_, 0,
                                             -min_depth_(1) + 1e-2); // offset1 >= -min_depth_(1)
        }
        if (!config_.use_shift) {
            if (problem_->HasParameterBlock(&offset0_))
                problem_->SetParameterBlockConstant(&offset0_);
            if (problem_->HasParameterBlock(&offset1_))
                problem_->SetParameterBlockConstant(&offset1_);
        }

        if (problem_->HasParameterBlock(qvec_.data())) {
            if (config_.constant_pose) {
                problem_->SetParameterBlockConstant(qvec_.data());
                problem_->SetParameterBlockConstant(tvec_.data());
            } else {
#ifdef CERES_PARAMETERIZATION_ENABLED
                ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                problem_->SetParameterization(qvec_.data(), quaternion_parameterization);
#else
                ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold;
                problem_->SetManifold(qvec_.data(), quaternion_manifold);
#endif
            }
        }
    }

    bool Solve() {
        if (problem_->NumResiduals() == 0)
            return false;
        ceres::Solver::Options solver_options = config_.solver_options;

        solver_options.linear_solver_type = ceres::DENSE_QR;
        solver_options.num_threads = 1;

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

class HybridPoseOptimizerScaleOnly {
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
    HybridPoseOptimizerScaleOnly(const Eigen::MatrixXd &x0, const Eigen::MatrixXd &x1, const Eigen::VectorXd &depth0,
                                 const Eigen::VectorXd &depth1, const std::vector<int> &indices_reproj_0,
                                 const std::vector<int> &indices_reproj_1, const std::vector<int> &indices_sampson,
                                 const PoseAndScale &pose, const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                                 const OptimizerConfig &config = OptimizerConfig())
        : K0_(K0), K1_(K1), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), x0_(x0), x1_(x1), d0_(depth0), d1_(depth1),
          indices_reproj_0_(indices_reproj_0), indices_reproj_1_(indices_reproj_1), indices_sampson_(indices_sampson),
          config_(config) {
        qvec_ = RotationMatrixToQuaternion<double>(pose.R());
        tvec_ = pose.t();
        offset0_ = 0;
        offset1_ = 0;
        scale_ = pose.scale;

        if (config_.reproj_loss_function.get() == nullptr)
            config_.reproj_loss_function.reset(new ceres::TrivialLoss());
        if (config_.sampson_loss_function.get() == nullptr)
            config_.sampson_loss_function.reset(new ceres::TrivialLoss());
    }

    void SetUp() {
        problem_.reset(new ceres::Problem(config_.problem_options));

        ceres::LossFunction *proj_loss_func = config_.reproj_loss_function.get();
        ceres::LossFunction *sampson_loss_func = config_.sampson_loss_function.get();

        if (config_.use_reprojection) {
            for (auto &i : indices_reproj_0_) {
                ceres::CostFunction *reproj_cost_0 =
                    LiftProjectionFunctor0::Create(K0_inv_ * x0_.col(i), x1_.col(i), d0_(i), K1_);
                problem_->AddResidualBlock(reproj_cost_0, proj_loss_func, &offset0_, qvec_.data(), tvec_.data());
            }
            for (auto &i : indices_reproj_1_) {
                ceres::CostFunction *reproj_cost_1 =
                    LiftProjectionFunctor1::Create(K1_inv_ * x1_.col(i), x0_.col(i), d1_(i), K0_);
                problem_->AddResidualBlock(reproj_cost_1, proj_loss_func, &scale_, &offset1_, qvec_.data(),
                                           tvec_.data());
            }
        }

        if (config_.use_sampson) {
            double weight =
                std::sqrt(config_.weight_sampson) / (1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1)));
            for (auto &i : indices_sampson_) {
                Eigen::Vector3d x0 = K0_inv_ * x0_.col(i);
                Eigen::Vector3d x1 = K1_inv_ * x1_.col(i);
                ceres::CostFunction *sampson_cost = SampsonErrorFunctor::Create(x0, x1, weight);
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
            } else {
#ifdef CERES_PARAMETERIZATION_ENABLED
                ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                problem_->SetParameterization(qvec_.data(), quaternion_parameterization);
#else
                ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold;
                problem_->SetManifold(qvec_.data(), quaternion_manifold);
#endif
            }
        }
    }

    bool Solve() {
        if (problem_->NumResiduals() == 0)
            return false;
        ceres::Solver::Options solver_options = config_.solver_options;

        solver_options.linear_solver_type = ceres::DENSE_QR;
        solver_options.num_threads = 1;

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

class HybridSharedFocalPoseOptimizer {
  protected:
    const Eigen::MatrixXd &x0_, &x1_;
    const Eigen::VectorXd &d0_, &d1_;
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;
    double focal_;
    double scale_, offset0_, offset1_;
    Eigen::Vector2d min_depth_;
    SharedFocalOptimizerConfig config_;

    const std::vector<int> &indices_reproj_0_, &indices_reproj_1_;
    const std::vector<int> &indices_sampson_;

    // ceres
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;

  public:
    HybridSharedFocalPoseOptimizer(const Eigen::MatrixXd &x0, const Eigen::MatrixXd &x1, const Eigen::VectorXd &depth0,
                                   const Eigen::VectorXd &depth1, const std::vector<int> &indices_reproj_0,
                                   const std::vector<int> &indices_reproj_1, const std::vector<int> &indices_sampson,
                                   const Eigen::Vector2d &min_depth, const PoseScaleOffsetSharedFocal &pose,
                                   const SharedFocalOptimizerConfig &config = SharedFocalOptimizerConfig())
        : x0_(x0), x1_(x1), d0_(depth0), d1_(depth1), indices_reproj_0_(indices_reproj_0),
          indices_reproj_1_(indices_reproj_1), indices_sampson_(indices_sampson), min_depth_(min_depth),
          config_(config) {
        qvec_ = RotationMatrixToQuaternion<double>(pose.R());
        tvec_ = pose.t();
        offset0_ = pose.offset0;
        offset1_ = pose.offset1;
        scale_ = pose.scale;
        focal_ = pose.focal;

        if (config_.reproj_loss_function.get() == nullptr)
            config_.reproj_loss_function.reset(new ceres::TrivialLoss());
        if (config_.sampson_loss_function.get() == nullptr)
            config_.sampson_loss_function.reset(new ceres::TrivialLoss());
    }

    void SetUp() {
        problem_.reset(new ceres::Problem(config_.problem_options));

        ceres::LossFunction *proj_loss_func = config_.reproj_loss_function.get();
        ceres::LossFunction *sampson_loss_func = config_.sampson_loss_function.get();

        if (config_.use_reprojection) {
            for (auto &i : indices_reproj_0_) {
                ceres::CostFunction *reproj_cost_0 =
                    LiftProjectionSharedFocalFunctor0::Create(x0_.col(i), x1_.col(i), d0_(i));
                problem_->AddResidualBlock(reproj_cost_0, proj_loss_func, &offset0_, qvec_.data(), tvec_.data(),
                                           &focal_);
            }
            for (auto &i : indices_reproj_1_) {
                ceres::CostFunction *reproj_cost_1 =
                    LiftProjectionSharedFocalFunctor1::Create(x1_.col(i), x0_.col(i), d1_(i));
                problem_->AddResidualBlock(reproj_cost_1, proj_loss_func, &scale_, &offset1_, qvec_.data(),
                                           tvec_.data(), &focal_);
            }
        }

        if (config_.use_sampson) {
            double weight = std::sqrt(config_.weight_sampson);
        for (auto &i : indices_sampson_) {
                ceres::CostFunction *sampson_cost =
                    SampsonErrorSharedFocalFunctor::Create(x0_.col(i), x1_.col(i), weight);
                problem_->AddResidualBlock(sampson_cost, sampson_loss_func, qvec_.data(), tvec_.data(), &focal_);
            }
        }

        if (problem_->HasParameterBlock(&scale_)) {
            problem_->SetParameterLowerBound(&scale_, 0, 1e-2); // scale >= 0
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset0_)) {
            problem_->SetParameterLowerBound(&offset0_, 0,
                                             -min_depth_(0) + 1e-2); // offset0 >= -min_depth_(0)
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset1_)) {
            problem_->SetParameterLowerBound(&offset1_, 0,
                                             -min_depth_(1) + 1e-2); // offset1 >= -min_depth_(1)
        }
        if (!config_.use_shift) {
            if (problem_->HasParameterBlock(&offset0_))
                problem_->SetParameterBlockConstant(&offset0_);
            if (problem_->HasParameterBlock(&offset1_))
                problem_->SetParameterBlockConstant(&offset1_);
        }

        if (problem_->HasParameterBlock(qvec_.data())) {
            if (config_.constant_pose) {
                problem_->SetParameterBlockConstant(qvec_.data());
                problem_->SetParameterBlockConstant(tvec_.data());
            } else {
#ifdef CERES_PARAMETERIZATION_ENABLED
                ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                problem_->SetParameterization(qvec_.data(), quaternion_parameterization);
#else
                ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold;
                problem_->SetManifold(qvec_.data(), quaternion_manifold);
#endif
            }
        }
    }

    bool Solve() {
        if (problem_->NumResiduals() == 0)
            return false;
        ceres::Solver::Options solver_options = config_.solver_options;

        solver_options.linear_solver_type = ceres::DENSE_QR;
        solver_options.num_threads = 1;

        std::string solver_error;
        CHECK(solver_options.IsValid(&solver_error)) << solver_error;

        ceres::Solve(solver_options, problem_.get(), &summary_);
        return true;
    }

    PoseScaleOffsetSharedFocal GetSolution() {
        Eigen::Matrix3d R = QuaternionToRotationMatrix<double>(qvec_);
        return PoseScaleOffsetSharedFocal(R, tvec_, scale_, offset0_, offset1_, focal_);
    }
};

class HybridTwoFocalPoseOptimizer {
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
    HybridTwoFocalPoseOptimizer(const Eigen::MatrixXd &x0, const Eigen::MatrixXd &x1, const Eigen::VectorXd &depth0,
                                const Eigen::VectorXd &depth1, const std::vector<int> &indices_reproj_0,
                                const std::vector<int> &indices_reproj_1, const std::vector<int> &indices_sampson,
                                const Eigen::Vector2d &min_depth, const PoseScaleOffsetTwoFocal &pose,
                                const TwoFocalOptimizerConfig &config = TwoFocalOptimizerConfig())
        : x0_(x0), x1_(x1), d0_(depth0), d1_(depth1), indices_reproj_0_(indices_reproj_0),
          indices_reproj_1_(indices_reproj_1), indices_sampson_(indices_sampson), min_depth_(min_depth),
          config_(config) {
        qvec_ = RotationMatrixToQuaternion<double>(pose.R());
        tvec_ = pose.t();
        offset0_ = pose.offset0;
        offset1_ = pose.offset1;
        scale_ = pose.scale;
        focal0_ = pose.focal0;
        focal1_ = pose.focal1;

        if (config_.reproj_loss_function.get() == nullptr)
            config_.reproj_loss_function.reset(new ceres::TrivialLoss());
        if (config_.sampson_loss_function.get() == nullptr)
            config_.sampson_loss_function.reset(new ceres::TrivialLoss());
    }

    void SetUp() {
        problem_.reset(new ceres::Problem(config_.problem_options));

        ceres::LossFunction *proj_loss_func = config_.reproj_loss_function.get();
        ceres::LossFunction *sampson_loss_func = config_.sampson_loss_function.get();

        if (config_.use_reprojection) {
            for (auto &i : indices_reproj_0_) {
                ceres::CostFunction *reproj_cost_0 =
                    LiftProjectionTwoFocalFunctor0::Create(x0_.col(i), x1_.col(i), d0_(i));
                problem_->AddResidualBlock(reproj_cost_0, proj_loss_func, &offset0_, qvec_.data(), tvec_.data(),
                                           &focal0_, &focal1_);
            }
            for (auto &i : indices_reproj_1_) {
                ceres::CostFunction *reproj_cost_1 =
                    LiftProjectionTwoFocalFunctor1::Create(x1_.col(i), x0_.col(i), d1_(i));
                problem_->AddResidualBlock(reproj_cost_1, proj_loss_func, &scale_, &offset1_, qvec_.data(),
                                           tvec_.data(), &focal0_, &focal1_);
            }
        }

        if (config_.use_sampson) {
            double weight = std::sqrt(config_.weight_sampson);
        for (auto &i : indices_sampson_) {
                ceres::CostFunction *sampson_cost = SampsonErrorTwoFocalFunctor::Create(x0_.col(i), x1_.col(i), weight);
                problem_->AddResidualBlock(sampson_cost, sampson_loss_func, qvec_.data(), tvec_.data(), &focal0_,
                                           &focal1_);
            }
        }

        if (problem_->HasParameterBlock(&scale_)) {
            problem_->SetParameterLowerBound(&scale_, 0, 1e-2); // scale >= 0
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset0_)) {
            problem_->SetParameterLowerBound(&offset0_, 0,
                                             -min_depth_(0) + 1e-2); // offset0 >= -min_depth_(0)
        }
        if (config_.min_depth_constraint && problem_->HasParameterBlock(&offset1_)) {
            problem_->SetParameterLowerBound(&offset1_, 0,
                                             -min_depth_(1) + 1e-2); // offset1 >= -min_depth_(1)
        }
        if (!config_.use_shift) {
            if (problem_->HasParameterBlock(&offset0_))
                problem_->SetParameterBlockConstant(&offset0_);
            if (problem_->HasParameterBlock(&offset1_))
                problem_->SetParameterBlockConstant(&offset1_);
        }

        if (problem_->HasParameterBlock(&focal0_)) {
            problem_->SetParameterLowerBound(&focal0_, 0, 1e-6); // focal0 >= 0
            problem_->SetParameterLowerBound(&focal1_, 0, 1e-6); // focal1 >= 0
        }

        if (problem_->HasParameterBlock(qvec_.data())) {
            if (config_.constant_pose) {
                problem_->SetParameterBlockConstant(qvec_.data());
                problem_->SetParameterBlockConstant(tvec_.data());
            } else {
#ifdef CERES_PARAMETERIZATION_ENABLED
                ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                problem_->SetParameterization(qvec_.data(), quaternion_parameterization);
#else
                ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold;
                problem_->SetManifold(qvec_.data(), quaternion_manifold);
#endif
            }
        }
    }

    bool Solve() {
        if (problem_->NumResiduals() == 0)
            return false;
        ceres::Solver::Options solver_options = config_.solver_options;

        solver_options.linear_solver_type = ceres::DENSE_QR;
        solver_options.num_threads = 1;

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

} // namespace madpose
