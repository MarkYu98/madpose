#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "utils.h"
#include "pose.h"

namespace madpose {

// *******************************************************************
//
// -------------- Cost Functors for calibrated cases -----------------
//
// *******************************************************************

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

        Eigen::Matrix<T, 3, 1> x1 = x0_.cast<T>();
        Eigen::Matrix<T, 3, 1> x2 = x1_.cast<T>();
        
        const T Ex1_0 = E(0, 0) * x1(0) + E(0, 1) * x1(1) + E(0, 2);
        const T Ex1_1 = E(1, 0) * x1(0) + E(1, 1) * x1(1) + E(1, 2);
        const T Ex1_2 = E(2, 0) * x1(0) + E(2, 1) * x1(1) + E(2, 2);

        const T Ex2_0 = E(0, 0) * x2(0) + E(1, 0) * x2(1) + E(2, 0);
        const T Ex2_1 = E(0, 1) * x2(0) + E(1, 1) * x2(1) + E(2, 1);
        // const T Ex2_2 = E(0, 2) * x2(0) + E(1, 2) * x2(1) + E(2, 2);

        const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
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

// *********************************************************************
//
// -------------- Cost Functors for shared-focal cases -----------------
//
// *********************************************************************

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

        const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
        const T r = C / Eigen::Vector<T, 4>(Ex1_0, Ex1_1, Ex2_0, Ex2_1).norm();

        residuals[0] = r * weight_;
        return true;
    }

private:
    const Eigen::Vector3d x0_, x1_;
    const double weight_;
};

// ******************************************************************
//
// -------------- Cost Functors for two-focal cases -----------------
//
// ******************************************************************

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

        const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
        const T r = C / Eigen::Vector<T, 4>(Ex1_0, Ex1_1, Ex2_0, Ex2_1).norm();

        residuals[0] = r * weight_;
        return true;
    }

private:
    const Eigen::Vector3d x0_, x1_;
    const double weight_;
};

} // namespace madpose

#endif // COST_FUNCTIONS_H