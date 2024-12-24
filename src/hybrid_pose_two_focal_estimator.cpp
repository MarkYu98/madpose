#include <colmap/geometry/triangulation.h>
#include <PoseLib/poselib.h>
#include <PoseLib/solvers/relpose_7pt.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include "solver.h"
#include "hybrid_pose_two_focal_estimator.h"

namespace madpose {

static std::pair<double, double> bougnoux_focals(const Eigen::Matrix3d &F) {
    Eigen::Vector3d p1 = Eigen::Vector3d(0.0, 0.0, 1.0);
    Eigen::Vector3d p2 = Eigen::Vector3d(0.0, 0.0, 1.0);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d e1 = svd.matrixV().col(2);
    Eigen::Vector3d e2 = svd.matrixU().col(2);

    e1 = e1 / e1[2];
    e2 = e2 / e2[2];

    Eigen::Matrix3d s_e1, s_e2;
    s_e2 << 0, -e2[2], e2[1],
            e2[2], 0, -e2[0],
            -e2[1], e2[0], 0;
    s_e1 << 0, -e1[2], e1[1],
            e1[2], 0, -e1[0],
            -e1[1], e1[0], 0;

    Eigen::Matrix3d II = Eigen::Vector3d(1.0, 1.0, 0.0).asDiagonal();
    double f1 = (-p2.transpose() * s_e2 * II * F * (p1 * p1.transpose()) * F.transpose() * p2)(0, 0) 
                / (p2.transpose() * s_e2 * II * F * II * F.transpose() * p2)(0, 0);
    double f2 = (-p1.transpose() * s_e1 * II * F.transpose() * (p2 * p2.transpose()) * F * p1)(0, 0) 
                / (p1.transpose() * s_e1 * II * F.transpose() * II * F * p1)(0, 0);

    return std::make_pair(f1, f2);
}

std::pair<PoseScaleOffsetTwoFocal, ransac_lib::HybridRansacStatistics> 
HybridEstimatePoseScaleOffsetTwoFocal(
    const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
    const std::vector<double> &depth0, const std::vector<double> &depth1, 
    const Eigen::Vector2d &min_depth, 
    const Eigen::Vector2d &pp0, const Eigen::Vector2d &pp1,
    const ExtendedHybridLORansacOptions& options, 
    const EstimatorConfig &est_config) {
    ExtendedHybridLORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = 0;

    std::vector<Eigen::Vector2d> x0_norm = x0;
    std::vector<Eigen::Vector2d> x1_norm = x1;
    for (int i = 0; i < x0.size(); i++) {
        x0_norm[i] -= pp0;
        x1_norm[i] -= pp1;
    }

    Eigen::Matrix3d T1, T2;
    double norm_scale = poselib::normalize_points(x0_norm, x1_norm, T1, T2, true, false, true);

    ransac_options.squared_inlier_thresholds_[0] /= norm_scale * norm_scale;
    ransac_options.squared_inlier_thresholds_[1] /= norm_scale * norm_scale;

    // Change to "three data types"
    ransac_options.data_type_weights_[1] *= 2 * ransac_options.squared_inlier_thresholds_[0] / ransac_options.squared_inlier_thresholds_[1];
    double sampson_squared_weight = ransac_options.data_type_weights_[1];
    ransac_options.data_type_weights_.push_back(ransac_options.data_type_weights_[1]);
    ransac_options.squared_inlier_thresholds_.push_back(ransac_options.squared_inlier_thresholds_[1]);
    ransac_options.data_type_weights_[1] = ransac_options.data_type_weights_[0];
    ransac_options.squared_inlier_thresholds_[1] = ransac_options.squared_inlier_thresholds_[0];

    HybridTwoFocalPoseEstimator solver(x0_norm, x1_norm, depth0, depth1, min_depth, 
        norm_scale, sampson_squared_weight, ransac_options.squared_inlier_thresholds_, est_config);

    PoseScaleOffsetTwoFocal best_solution;
    ransac_lib::HybridRansacStatistics ransac_stats;

    HybridLOMSAC<PoseScaleOffsetTwoFocal, std::vector<PoseScaleOffsetTwoFocal>, HybridTwoFocalPoseEstimator> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    best_solution.focal0 *= norm_scale;
    best_solution.focal1 *= norm_scale;
    return std::make_pair(best_solution, ransac_stats);
}

int HybridTwoFocalPoseEstimator::MinimalSolver(
    const std::vector<std::vector<int>>& sample,
    const int solver_idx, std::vector<PoseScaleOffsetTwoFocal>* models) const {
    models->clear();

    if (solver_idx == 0) {
        Eigen::Matrix3x4d x0 = x0_norm_(Eigen::all, sample[0]);
        Eigen::Matrix3x4d x1 = x1_norm_(Eigen::all, sample[0]);

        std::vector<PoseScaleOffsetTwoFocal> sols;
        int num_sols = estimate_scale_shift_pose_two_focal(x0, x1, d0_(sample[0]), d1_(sample[0]), &sols, false);
        for (int i = 0; i < num_sols; i++) {
            if (!est_config_.min_depth_constraint || (sols[i].offset0 > -min_depth_(0) && sols[i].offset1 > -min_depth_(1) * sols[i].scale)) {
                PoseScaleOffsetTwoFocal sol = sols[i];
                sol.offset1 /= sol.scale;
                models->push_back(sol);
            }
        }
    }
    else if (solver_idx == 1) {
        Eigen::MatrixXd x0 = x0_norm_(Eigen::all, sample[2]);
        Eigen::MatrixXd x1 = x1_norm_(Eigen::all, sample[2]);

        std::vector<Eigen::Vector3d> x0_vec(sample[2].size()), x1_vec(sample[2].size());
        std::vector<Eigen::Vector2d> x0_2dvec(sample[2].size()), x1_2dvec(sample[2].size());
        for (int i = 0; i < sample[2].size(); i++) {
            x0_vec[i] = x0.col(i).normalized();
            x1_vec[i] = x1.col(i).normalized();
            x0_2dvec[i] = x0.col(i).head<2>();
            x1_2dvec[i] = x1.col(i).head<2>();
        }
        std::vector<Eigen::Matrix3d> fund_matrices;
        poselib::relpose_7pt(x0_vec, x1_vec, &fund_matrices);

        for (auto &F : fund_matrices) {
            double f0, f1;
            std::tie(f0, f1) = bougnoux_focals(F);
            f0 = std::sqrt(std::abs(f0));
            f1 = std::sqrt(std::abs(f1));

            Eigen::Matrix3d K0, K1;
            K0 << f0, 0.0, 0.0, 0.0, f0, 0.0, 0.0, 0.0, 1.0;
            K1 << f1, 0.0, 0.0, 0.0, f1, 0.0, 0.0, 0.0, 1.0;

            Eigen::Matrix3d E = K1.transpose() * F * K0;
            Eigen::Matrix3d R;
            Eigen::Vector3d t;

            cv::Mat cv_E, cv_R, cv_tr;
            cv::eigen2cv(E, cv_E);

            cv::Mat cv_x0_2dvec(sample[2].size(), 2, CV_64F);
            cv::Mat cv_x1_2dvec(sample[2].size(), 2, CV_64F);
            for (int i = 0; i < sample[2].size(); i++) {
                cv_x0_2dvec.at<double>(i, 0) = x0_2dvec[i](0);
                cv_x0_2dvec.at<double>(i, 1) = x0_2dvec[i](1);
                cv_x1_2dvec.at<double>(i, 0) = x1_2dvec[i](0);
                cv_x1_2dvec.at<double>(i, 1) = x1_2dvec[i](1);
            }
            cv::recoverPose(cv_E, cv_x0_2dvec, cv_x1_2dvec, cv::Mat_<float>::eye(3, 3), cv_R, cv_tr, 1e9);

            cv::cv2eigen(cv_R, R);
            cv::cv2eigen(cv_tr, t);
            PoseScaleOffsetTwoFocal sol(R, t, 1.0, 0.0, 0.0, f0, f1);
            
            Eigen::Matrix3x4d proj_matrix0 = K0 * Eigen::Matrix3x4d::Identity();
            Eigen::Matrix3x4d proj_matrix1 = K1 * sol.pose;
            std::vector<Eigen::Vector3d> p3d_vec = colmap::TriangulatePoints(proj_matrix0, proj_matrix1, x0_2dvec, x1_2dvec);
            Eigen::MatrixXd p3d(3, p3d_vec.size());
            for (int i = 0; i < p3d_vec.size(); i++) {
                p3d.col(i) = p3d_vec[i];
            }  

            // Estimate scale and shift by least-squares
            Eigen::MatrixXd A(sample[2].size(), 2);
            A.col(0) = d0_(sample[2]);
            A.col(1) = Eigen::VectorXd::Ones(sample[2].size());
            Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * p3d.row(2).transpose());
            double s0 = x(0);
            double offset0 = x(1) / s0;
            if (est_config_.min_depth_constraint && offset0 < -min_depth_(0)) continue;

            p3d = p3d / s0;
            sol.pose.block<3, 1>(0, 3) = sol.t() / s0;
            p3d = (sol.R() * p3d).colwise() + sol.t();
            A.col(0) = d1_(sample[2]);
            A.col(1) = Eigen::VectorXd::Ones(sample[2].size());
            x = (A.transpose() * A).ldlt().solve(A.transpose() * p3d.row(2).transpose());
            double scale = x(0);
            double offset1 = x(1) / scale;
            if (est_config_.min_depth_constraint && offset1 < -min_depth_(1)) continue;

            sol.scale = scale;
            sol.offset0 = offset0;
            sol.offset1 = offset1;
            models->push_back(sol);
        }
    }
    return models->size();
}

int HybridTwoFocalPoseEstimator::NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetTwoFocal* solution) const {
    if ((sample[0].size() < 4 && sample[1].size() < 4) || sample[2].size() < 7) {
        return 0;
    } 

    TwoFocalOptimizerConfig config;
    config.solver_options.max_num_iterations = 25;
    config.use_sampson = true;
    config.use_reprojection = true;
    if (est_config_.LO_type == EstimatorOption::MD_ONLY)
        config.use_sampson = false;
    if (est_config_.LO_type == EstimatorOption::EPI_ONLY)
        config.use_reprojection = false;
    config.weight_sampson = sampson_squared_weight_;
    config.min_depth_constraint = est_config_.min_depth_constraint;
    HybridTwoFocalPoseOptimizer optim(x0_norm_, x1_norm_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_, 
        *solution, config);
    optim.SetUp();
    if (!optim.Solve()) 
        return 0;
    *solution = optim.GetSolution();
    return 1;
}

double HybridTwoFocalPoseEstimator::EvaluateModelOnPoint(const PoseScaleOffsetTwoFocal& model, int t, int i, bool is_for_inlier) const {
    if (!is_for_inlier && est_config_.score_type == EstimatorOption::EPI_ONLY && t != 2) {
        return std::numeric_limits<double>::max();
    }
    if (!is_for_inlier && est_config_.score_type == EstimatorOption::MD_ONLY && t == 2) {
        return std::numeric_limits<double>::max();
    }
    
    Eigen::Matrix3d K0, K1, K0_inv, K1_inv;
    K0 << model.focal0, 0.0, 0.0, 0.0, model.focal0, 0.0, 0.0, 0.0, 1.0;
    K1 << model.focal1, 0.0, 0.0, 0.0, model.focal1, 0.0, 0.0, 0.0, 1.0;
    K0_inv << 1.0 / model.focal0, 0.0, 0.0, 0.0, 1.0 / model.focal0, 0.0, 0.0, 0.0, 1.0;
    K1_inv << 1.0 / model.focal1, 0.0, 0.0, 0.0, 1.0 / model.focal1, 0.0, 0.0, 0.0, 1.0;

    if (t == 0) {
        Eigen::Vector3d p3d0 = (K0_inv * x0_norm_.col(i)) * (d0_(i) + model.offset0);
        Eigen::Vector3d q = model.R() * p3d0 + model.t();
        Eigen::Vector3d p2d_project = K1 * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2) 
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x1 = x1_norm_.col(i).head<2>();
        double reproj_error = (p2d - x1).squaredNorm();
        return reproj_error;
    }
    else if (t == 1) {   
        Eigen::Vector3d p3d1 = (K1_inv * x1_norm_.col(i)) * (d1_(i) + model.offset1) * model.scale;
        Eigen::Vector3d q = model.R().transpose() * p3d1 - model.R().transpose() * model.t();
        Eigen::Vector3d p2d_project = K0 * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2) 
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x0 = x0_norm_.col(i).head<2>();
        double reproj_error = (p2d - x0).squaredNorm();
        return reproj_error;
    }
    else if (t == 2) {
        Eigen::Matrix3d E = to_essential_matrix(model.R(), model.t());
        Eigen::Matrix3d F = K1_inv.transpose() * E * K0_inv;

        double sampson_error = compute_sampson_error(x0_norm_.col(i).head<2>(), x1_norm_.col(i).head<2>(), F);
        return sampson_error;
    }
}

// Linear least squares solver. 
void HybridTwoFocalPoseEstimator::LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffsetTwoFocal* model) const {
    if ((sample[0].size() < 4 && sample[1].size() < 4) || sample[2].size() < 7) {
        return;
    }

    TwoFocalOptimizerConfig config;
    config.use_sampson = true;
    config.use_reprojection = true;

    if (est_config_.LO_type == EstimatorOption::MD_ONLY)
        config.use_sampson = false;
    if (est_config_.LO_type == EstimatorOption::EPI_ONLY)
        config.use_reprojection = false;
    config.weight_sampson = sampson_squared_weight_;
    config.min_depth_constraint = est_config_.min_depth_constraint;
    config.solver_options.max_num_iterations = 25;
    HybridTwoFocalPoseOptimizer optim(x0_norm_, x1_norm_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_, 
        *model, config);
    optim.SetUp();
    if (!optim.Solve()) 
        return;
    *model = optim.GetSolution();
}

} // namespace madpose