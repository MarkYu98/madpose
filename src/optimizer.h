#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "utils.h"
#include "pose.h"

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
    bool use_geometric = false;
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


struct SampsonErrorFunctor {
public:
    SampsonErrorFunctor(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, 
                        const double &sq_weight = 1.0) : 
        x0_(x0), x1_(x1), K0_(K0), K1_(K1), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), weight_(std::sqrt(sq_weight)) {}
    
    static ceres::CostFunction* Create(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, 
                                       const double &sq_weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<SampsonErrorFunctor, 1, 4, 3>(new SampsonErrorFunctor(x0, x1, K0, K1, sq_weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Matrix<T, 3, 3> E;
        E << T(0.0), -t(2), t(1), t(2), T(0.0), -t(0), -t(1), t(0), T(0.0);
        E = E * R;
        // E = K1_inv_.transpose() * E * K0_inv_;

        Eigen::Matrix<T, 3, 1> x1 = x0_.cast<T>();
        Eigen::Matrix<T, 3, 1> x2 = x1_.cast<T>();
        
        const T Ex1_0 = E(0, 0) * x1(0) + E(0, 1) * x1(1) + E(0, 2);
        const T Ex1_1 = E(1, 0) * x1(0) + E(1, 1) * x1(1) + E(1, 2);
        const T Ex1_2 = E(2, 0) * x1(0) + E(2, 1) * x1(1) + E(2, 2);

        const T Ex2_0 = E(0, 0) * x2(0) + E(1, 0) * x2(1) + E(2, 0);
        const T Ex2_1 = E(0, 1) * x2(0) + E(1, 1) * x2(1) + E(2, 1);
        // const T Ex2_2 = E(0, 2) * x2(0) + E(1, 2) * x2(1) + E(2, 2);

        // const T C = x2.dot(E * x1);
        // Eigen::Matrix<T, 2, 3> E1 = E.template block<2, 3>(0, 0);
        // Eigen::Matrix<T, 2, 3> E2 = E.template block<3, 2>(0, 0).transpose();
        // const T nJc_sq = (E1 * x1).squaredNorm() + (E2 * x2).squaredNorm();
        // residuals[0] = C / ceres::sqrt(nJc_sq);

        // const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
        const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
        // const T Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        // const T Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
        const T r = C / Eigen::Vector<T, 4>(Ex1_0, Ex1_1, Ex2_0, Ex2_1).norm();

        residuals[0] = r / (1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1)));
        residuals[0] *= weight_;
        return true;
    }

private:
    const Eigen::Vector3d x0_, x1_;
    const Eigen::Matrix3d K0_, K1_;
    const Eigen::Matrix3d K0_inv_, K1_inv_;
    const double weight_;
};

class SharedFocalOptimizerConfig : public OptimizerConfig {
public:
    SharedFocalOptimizerConfig() : OptimizerConfig() {}
    bool constant_focal = false;
};

typedef SharedFocalOptimizerConfig TwoFocalOptimizerConfig;

struct LiftProjectionFunctor {
public:
    LiftProjectionFunctor(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, 
                          const double& x0_depth, const double &x1_depth,
                          const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, bool squared_cost = false) : 
        x0_(x0), x1_(x1), K0_(K0), K1_(K1), x0_depth_(x0_depth), x1_depth_(x1_depth), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), squared_(squared_cost) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, 
                                       const double &x0_depth, const double &x1_depth,
                                       const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, const bool squared_cost = false) {
        return (new ceres::AutoDiffCostFunction<LiftProjectionFunctor, 4, 1, 1, 1, 4, 3>(new LiftProjectionFunctor(x0, x1, x0_depth, x1_depth, K0, K1, squared_cost)));
    }

    template <typename T>
    bool operator()(const T* const scale, const T* const o0, const T* const o1, const T* const qvec, const T* const tvec, T* residuals) const {
        Eigen::Vector<T, 3> x3d = (K0_inv_ * x0_.cast<T>()) * (x0_depth_ + o0[0]);
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_1 = R * x3d + t;
        Eigen::Vector<T, 3> x1_hat = K1_ * x3d_1;

        T r0, r1;
        x1_hat /= x1_hat[2];
        if (squared_) {
            r0 = (x1_hat - x1_.cast<T>()).squaredNorm();
        } else {
            r0 = (x1_hat - x1_.cast<T>()).norm();
        }

        x3d = K1_inv_ * x1_.cast<T>() * (x1_depth_ + o1[0]) * scale[0];
        Eigen::Vector<T, 3> x3d_0 = R.transpose() * x3d - R.transpose() * t;
        Eigen::Vector<T, 3> x0_hat = K0_ * x3d_0;
        x0_hat /= x0_hat[2];
        if (squared_) {
            r1 = (x0_hat - x0_.cast<T>()).squaredNorm();
        } else {
            r1 = (x0_hat - x0_.cast<T>()).norm();
        }
        
        // residuals[0] = ceres::sqrt(r0 * r1);
        // if (r0 < r1) {
        //     residuals[0] = x1_hat[0] - x1_[0];
        //     residuals[1] = x1_hat[1] - x1_[1];
        // }
        // else {
        //     residuals[0] = x0_hat[0] - x0_[0];
        //     residuals[1] = x0_hat[1] - x0_[1];
        // }
        residuals[0] = x1_hat[0] - x1_[0];
        residuals[1] = x1_hat[1] - x1_[1];
        residuals[2] = x0_hat[0] - x0_[0];
        residuals[3] = x0_hat[1] - x0_[1];

        // residuals[0] = residuals[0] < T(8.0) ? residuals[0] : T(8.0);
        // residuals[0] = ceres::sqrt(residuals[0]);
        return true;
    }
private:
    const Eigen::Vector3d x0_, x1_;
    const Eigen::Matrix3d K0_, K1_;
    const Eigen::Matrix3d K0_inv_, K1_inv_; 
    const double x0_depth_, x1_depth_;
    const bool squared_;
};

struct LiftProjectionFunctor0 {
public:
    LiftProjectionFunctor0(const Eigen::Vector3d &x0_calib, const Eigen::Vector3d &x1, 
                           const double& x0_depth, const Eigen::Matrix3d &K1) : 
        x0_calib_(x0_calib), x1_(x1), K1_(K1), x0_depth_(x0_depth) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x0_calib, const Eigen::Vector3d &x1, 
                                       const double &x0_depth, const Eigen::Matrix3d &K1) {
        return (new ceres::AutoDiffCostFunction<LiftProjectionFunctor0, 2, 1, 4, 3>(new LiftProjectionFunctor0(x0_calib, x1, x0_depth, K1)));
    }

    template <typename T>
    bool operator()(const T* const o0, const T* const qvec, const T* const tvec, T* residuals) const {
        Eigen::Vector<T, 3> x3d = x0_calib_.cast<T>() * (x0_depth_ + o0[0]);
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_1 = R * x3d + t;
        Eigen::Vector<T, 3> x1_hat = K1_ * x3d_1;
        x1_hat /= x1_hat[2];
        residuals[0] = x1_hat[0] - x1_[0];
        residuals[1] = x1_hat[1] - x1_[1];

        return true;
    }
private:
    const Eigen::Vector3d x0_calib_, x1_;
    const Eigen::Matrix3d K1_;
    const double x0_depth_;
};

struct LiftProjectionFunctor1 {
public:
    LiftProjectionFunctor1(const Eigen::Vector3d &x1_calib, const Eigen::Vector3d &x0, 
                           const double& x1_depth, const Eigen::Matrix3d &K0) : 
        x1_calib_(x1_calib), x0_(x0), K0_(K0), x1_depth_(x1_depth) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x1_calib, const Eigen::Vector3d &x0, 
                                       const double &x1_depth, const Eigen::Matrix3d &K0) {
        return (new ceres::AutoDiffCostFunction<LiftProjectionFunctor1, 2, 1, 1, 4, 3>(new LiftProjectionFunctor1(x1_calib, x0, x1_depth, K0)));
    }

    template <typename T>
    bool operator()(const T* const scale, const T* const o1, const T* const qvec, const T* const tvec, T* residuals) const {
        Eigen::Vector<T, 3> x3d = x1_calib_.cast<T>() * (x1_depth_ + o1[0]) * scale[0];
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_0 = R.transpose() * x3d - R.transpose() * t;
        Eigen::Vector<T, 3> x0_hat = K0_ * x3d_0;
        x0_hat /= x0_hat[2];
        residuals[0] = x0_hat[0] - x0_[0];
        residuals[1] = x0_hat[1] - x0_[1];

        return true;
    }
private:
    const Eigen::Vector3d x1_calib_, x0_;
    const Eigen::Matrix3d K0_;
    const double x1_depth_;
};

struct LiftGeometryFunctor {
public:
    LiftGeometryFunctor(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, 
                          const double& x0_depth, const double &x1_depth,
                          const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, const double& weight = 1.0) : 
        x0_(x0), x1_(x1), K0_(K0), K1_(K1), x0_depth_(x0_depth), x1_depth_(x1_depth), weight_(weight) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, 
                                       const double &x0_depth, const double &x1_depth,
                                       const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, const double &weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<LiftGeometryFunctor, 3, 1, 1, 1, 4, 3>(new LiftGeometryFunctor(x0, x1, x0_depth, x1_depth, K0, K1, weight)));
    }
    template <typename T>
    bool operator()(const T* const scale, const T* const o0, const T* const o1, const T* const qvec, const T* const tvec, T* residuals) const {
        Eigen::Vector<T, 3> x3d = x0_.cast<T>() * (x0_depth_ + o0[0]);
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_0 = R * x3d + t;
        Eigen::Vector<T, 3> x3d_1 = x1_.cast<T>() * (x1_depth_ + o1[0]) * scale[0];
        if (x3d_0[2] <= T(0.0)) {
            residuals[0] = T(1e6);
            residuals[1] = T(1e6);
            residuals[2] = T(1e6);
        }

        residuals[0] = x3d_0[0] - x3d_1[0];
        residuals[1] = x3d_0[1] - x3d_1[1];
        residuals[2] = x3d_0[2] - x3d_1[2];
        residuals[0] *= weight_;
        residuals[1] *= weight_;
        residuals[2] *= weight_;
        return true;
    }
private:
    const Eigen::Vector3d x0_, x1_;
    const Eigen::Matrix3d K0_, K1_;
    const double x0_depth_, x1_depth_;
    const double weight_;
};


struct LiftProjectionTwoFocalFunctor0 {
public:
    LiftProjectionTwoFocalFunctor0(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const double& x0_depth) : 
        x0_(x0), x1_(x1), x0_depth_(x0_depth) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const double &x0_depth) {
        return (new ceres::AutoDiffCostFunction<LiftProjectionTwoFocalFunctor0, 2, 1, 4, 3, 1, 1>(new LiftProjectionTwoFocalFunctor0(x0, x1, x0_depth)));
    }

    template <typename T>
    bool operator()(const T* const o0, const T* const qvec, const T* const tvec, 
                    const T* const focal0,  const T* const focal1, T* residuals) const {
        Eigen::Matrix<T, 3, 3> K0_inv, K1;
        K0_inv << T(1.0) / focal0[0], T(0.0), T(0.0), T(0.0), T(1.0) / focal0[0], T(0.0), T(0.0), T(0.0), T(1.0);
        K1 << focal1[0], T(0.0), T(0.0), T(0.0), focal1[0], T(0.0), T(0.0), T(0.0), T(1.0);
        Eigen::Vector<T, 3> x3d = (K0_inv * x0_.cast<T>()) * (x0_depth_ + o0[0]);
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_1 = R * x3d + t;
        Eigen::Vector<T, 3> x1_hat = K1 * x3d_1;
        x1_hat /= x1_hat[2];
        residuals[0] = x1_hat[0] - x1_[0];
        residuals[1] = x1_hat[1] - x1_[1];

        return true;
    }
private:
    const Eigen::Vector3d x0_, x1_;
    const double x0_depth_;
};

struct LiftProjectionTwoFocalFunctor1 {
public:
    LiftProjectionTwoFocalFunctor1(const Eigen::Vector3d &x1, const Eigen::Vector3d &x0, const double& x1_depth) : 
        x1_(x1), x0_(x0), x1_depth_(x1_depth) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x1, const Eigen::Vector3d &x0, const double &x1_depth) {
        return (new ceres::AutoDiffCostFunction<LiftProjectionTwoFocalFunctor1, 2, 1, 1, 4, 3, 1, 1>(new LiftProjectionTwoFocalFunctor1(x1, x0, x1_depth)));
    }

    template <typename T>
    bool operator()(const T* const scale, const T* const o1, const T* const qvec, const T* const tvec, 
                    const T* const focal0, const T* const focal1, T* residuals) const {
        Eigen::Matrix<T, 3, 3> K0, K1_inv;
        K0 << focal0[0], T(0.0), T(0.0), T(0.0), focal0[0], T(0.0), T(0.0), T(0.0), T(1.0);
        K1_inv << T(1.0) / focal1[0], T(0.0), T(0.0), T(0.0), T(1.0) / focal1[0], T(0.0), T(0.0), T(0.0), T(1.0);
        Eigen::Vector<T, 3> x3d = (K1_inv * x1_.cast<T>()) * (x1_depth_ + o1[0]) * scale[0];
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_0 = R.transpose() * x3d - R.transpose() * t;
        Eigen::Vector<T, 3> x0_hat = K0 * x3d_0;
        x0_hat /= x0_hat[2];
        residuals[0] = x0_hat[0] - x0_[0];
        residuals[1] = x0_hat[1] - x0_[1];

        return true;
    }
private:
    const Eigen::Vector3d x1_, x0_;
    const double x1_depth_;
};

struct SampsonErrorTwoFocalFunctor {
public:
    SampsonErrorTwoFocalFunctor(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const double &sq_weight = 1.0) : 
        x0_(x0), x1_(x1), weight_(std::sqrt(sq_weight)) {}
    
    static ceres::CostFunction* Create(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const double &sq_weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<SampsonErrorTwoFocalFunctor, 1, 4, 3, 1, 1>(new SampsonErrorTwoFocalFunctor(x0, x1, sq_weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const focal0, const T* const focal1, T* residuals) const {
        Eigen::Matrix<T, 3, 3> K0_inv, K1_inv;
        K0_inv << T(1.0) / focal0[0], T(0.0), T(0.0), T(0.0), T(1.0) / focal0[0], T(0.0), T(0.0), T(0.0), T(1.0);
        K1_inv << T(1.0) / focal1[0], T(0.0), T(0.0), T(0.0), T(1.0) / focal1[0], T(0.0), T(0.0), T(0.0), T(1.0);
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Matrix<T, 3, 3> E;
        E << T(0.0), -t(2), t(1), t(2), T(0.0), -t(0), -t(1), t(0), T(0.0);
        E = E * R;

        // Actually it's F here
        E = K1_inv.transpose() * E * K0_inv;

        Eigen::Matrix<T, 3, 1> x1 = x0_.cast<T>();
        Eigen::Matrix<T, 3, 1> x2 = x1_.cast<T>();
        
        const T Ex1_0 = E(0, 0) * x1(0) + E(0, 1) * x1(1) + E(0, 2);
        const T Ex1_1 = E(1, 0) * x1(0) + E(1, 1) * x1(1) + E(1, 2);
        const T Ex1_2 = E(2, 0) * x1(0) + E(2, 1) * x1(1) + E(2, 2);

        const T Ex2_0 = E(0, 0) * x2(0) + E(1, 0) * x2(1) + E(2, 0);
        const T Ex2_1 = E(0, 1) * x2(0) + E(1, 1) * x2(1) + E(2, 1);
        // const T Ex2_2 = E(0, 2) * x2(0) + E(1, 2) * x2(1) + E(2, 2);

        // const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
        const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
        // const T Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        // const T Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
        const T r = C / Eigen::Vector<T, 4>(Ex1_0, Ex1_1, Ex2_0, Ex2_1).norm();

        residuals[0] = r * weight_;
        return true;
    }

private:
    const Eigen::Vector3d x0_, x1_;
    const double weight_;
};


struct LiftProjectionSharedFocalFunctor0 {
public:
    LiftProjectionSharedFocalFunctor0(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const double& x0_depth) : 
        x0_(x0), x1_(x1), x0_depth_(x0_depth) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const double &x0_depth) {
        return (new ceres::AutoDiffCostFunction<LiftProjectionSharedFocalFunctor0, 2, 1, 4, 3, 1>(new LiftProjectionSharedFocalFunctor0(x0, x1, x0_depth)));
    }

    template <typename T>
    bool operator()(const T* const o0, const T* const qvec, const T* const tvec, const T* const focal, T* residuals) const {
        Eigen::Matrix<T, 3, 3> K, K_inv;
        K << focal[0], T(0.0), T(0.0), T(0.0), focal[0], T(0.0), T(0.0), T(0.0), T(1.0);
        K_inv << T(1.0) / focal[0], T(0.0), T(0.0), T(0.0), T(1.0) / focal[0], T(0.0), T(0.0), T(0.0), T(1.0);
        Eigen::Vector<T, 3> x3d = (K_inv * x0_.cast<T>()) * (x0_depth_ + o0[0]);
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_1 = R * x3d + t;
        Eigen::Vector<T, 3> x1_hat = K * x3d_1;
        x1_hat /= x1_hat[2];
        residuals[0] = x1_hat[0] - x1_[0];
        residuals[1] = x1_hat[1] - x1_[1];

        return true;
    }
private:
    const Eigen::Vector3d x0_, x1_;
    const double x0_depth_;
};

struct LiftProjectionSharedFocalFunctor1 {
public:
    LiftProjectionSharedFocalFunctor1(const Eigen::Vector3d &x1, const Eigen::Vector3d &x0, const double& x1_depth) : 
        x1_(x1), x0_(x0), x1_depth_(x1_depth) {}
    static ceres::CostFunction* Create(const Eigen::Vector3d &x1, const Eigen::Vector3d &x0, const double &x1_depth) {
        return (new ceres::AutoDiffCostFunction<LiftProjectionSharedFocalFunctor1, 2, 1, 1, 4, 3, 1>(new LiftProjectionSharedFocalFunctor1(x1, x0, x1_depth)));
    }

    template <typename T>
    bool operator()(const T* const scale, const T* const o1, const T* const qvec, const T* const tvec, const T* const focal, T* residuals) const {
        Eigen::Matrix<T, 3, 3> K, K_inv;
        K << focal[0], T(0.0), T(0.0), T(0.0), focal[0], T(0.0), T(0.0), T(0.0), T(1.0);
        K_inv << T(1.0) / focal[0], T(0.0), T(0.0), T(0.0), T(1.0) / focal[0], T(0.0), T(0.0), T(0.0), T(1.0);
        Eigen::Vector<T, 3> x3d = (K_inv * x1_.cast<T>()) * (x1_depth_ + o1[0]) * scale[0];
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Vector<T, 3> x3d_0 = R.transpose() * x3d - R.transpose() * t;
        Eigen::Vector<T, 3> x0_hat = K * x3d_0;
        x0_hat /= x0_hat[2];
        residuals[0] = x0_hat[0] - x0_[0];
        residuals[1] = x0_hat[1] - x0_[1];

        return true;
    }
private:
    const Eigen::Vector3d x1_, x0_;
    const double x1_depth_;
};

struct SampsonErrorSharedFocalFunctor {
public:
    SampsonErrorSharedFocalFunctor(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const double &sq_weight = 1.0) : 
        x0_(x0), x1_(x1), weight_(std::sqrt(sq_weight)) {}
    
    static ceres::CostFunction* Create(const Eigen::Vector3d &x0, const Eigen::Vector3d &x1, const double &sq_weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<SampsonErrorSharedFocalFunctor, 1, 4, 3, 1>(new SampsonErrorSharedFocalFunctor(x0, x1, sq_weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const focal, T* residuals) const {
        Eigen::Matrix<T, 3, 3> K_inv;
        K_inv << T(1.0) / focal[0], T(0.0), T(0.0), T(0.0), T(1.0) / focal[0], T(0.0), T(0.0), T(0.0), T(1.0);
        Eigen::Matrix<T, 3, 3> R = QuaternionToRotationMatrix<T>(Eigen::Map<const Eigen::Vector<T, 4>>(qvec));
        Eigen::Vector<T, 3> t = Eigen::Map<const Eigen::Vector<T, 3>>(tvec);

        Eigen::Matrix<T, 3, 3> E;
        E << T(0.0), -t(2), t(1), t(2), T(0.0), -t(0), -t(1), t(0), T(0.0);
        E = E * R;
        E = K_inv.transpose() * E * K_inv;

        Eigen::Matrix<T, 3, 1> x1 = x0_.cast<T>();
        Eigen::Matrix<T, 3, 1> x2 = x1_.cast<T>();
        
        const T Ex1_0 = E(0, 0) * x1(0) + E(0, 1) * x1(1) + E(0, 2);
        const T Ex1_1 = E(1, 0) * x1(0) + E(1, 1) * x1(1) + E(1, 2);
        const T Ex1_2 = E(2, 0) * x1(0) + E(2, 1) * x1(1) + E(2, 2);

        const T Ex2_0 = E(0, 0) * x2(0) + E(1, 0) * x2(1) + E(2, 0);
        const T Ex2_1 = E(0, 1) * x2(0) + E(1, 1) * x2(1) + E(2, 1);
        // const T Ex2_2 = E(0, 2) * x2(0) + E(1, 2) * x2(1) + E(2, 2);

        // const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
        const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
        // const T Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        // const T Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
        const T r = C / Eigen::Vector<T, 4>(Ex1_0, Ex1_1, Ex2_0, Ex2_1).norm();

        residuals[0] = r * weight_;
        return true;
    }

private:
    const Eigen::Vector3d x0_, x1_;
    const double weight_;
};

} // namespace acmpose

#endif // OPTIMIZER_H

