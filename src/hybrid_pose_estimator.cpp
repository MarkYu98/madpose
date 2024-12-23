#include <colmap/geometry/triangulation.h>
#include <PoseLib/misc/essential.h>
#include <PoseLib/solvers/relpose_5pt.h>
#include "solver.h"
#include "hybrid_pose_estimator.h"

namespace acmpose {

std::pair<PoseScaleOffset, ransac_lib::HybridRansacStatistics> 
HybridEstimatePoseScaleOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                              const std::vector<double> &depth0, const std::vector<double> &depth1, 
                              const Eigen::Vector2d &min_depth, 
                              const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                              const ExtendedHybridLORansacOptions& options,
                              const EstimatorConfig &est_config) {
    ExtendedHybridLORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = 0;

    // Change to "three data types"
    ransac_options.data_type_weights_[1] *= 2 * ransac_options.squared_inlier_thresholds_[0] / ransac_options.squared_inlier_thresholds_[1];
    double sampson_squared_weight = ransac_options.data_type_weights_[1];
    
    ransac_options.data_type_weights_.push_back(ransac_options.data_type_weights_[1]);
    ransac_options.squared_inlier_thresholds_.push_back(ransac_options.squared_inlier_thresholds_[1]);
    ransac_options.data_type_weights_[1] = ransac_options.data_type_weights_[0];
    ransac_options.squared_inlier_thresholds_[1] = ransac_options.squared_inlier_thresholds_[0];

    HybridPoseEstimator solver(x0, x1, depth0, depth1, min_depth, K0, K1, sampson_squared_weight, ransac_options.squared_inlier_thresholds_, est_config);

    PoseScaleOffset best_solution;
    ransac_lib::HybridRansacStatistics ransac_stats;

    HybridUncertaintyLOMSAC<PoseScaleOffset, std::vector<PoseScaleOffset>, HybridPoseEstimator> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    return std::make_pair(best_solution, ransac_stats);
}

std::pair<PoseAndScale, ransac_lib::HybridRansacStatistics> 
HybridEstimatePoseAndScale(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                            const std::vector<double> &depth0, const std::vector<double> &depth1, 
                            const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                            const ExtendedHybridLORansacOptions& options,
                            const EstimatorConfig &est_config) {
    ExtendedHybridLORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = 0;

    // Change to "three data types"
    ransac_options.data_type_weights_[1] *= 2 * ransac_options.squared_inlier_thresholds_[0] / ransac_options.squared_inlier_thresholds_[1];
    double sampson_squared_weight = ransac_options.data_type_weights_[1];
    
    ransac_options.data_type_weights_.push_back(ransac_options.data_type_weights_[1]);
    ransac_options.squared_inlier_thresholds_.push_back(ransac_options.squared_inlier_thresholds_[1]);
    ransac_options.data_type_weights_[1] = ransac_options.data_type_weights_[0];
    ransac_options.squared_inlier_thresholds_[1] = ransac_options.squared_inlier_thresholds_[0];

    HybridPoseEstimatorScaleOnly solver(x0, x1, depth0, depth1, K0, K1, sampson_squared_weight, ransac_options.squared_inlier_thresholds_, est_config);

    PoseAndScale best_solution;
    ransac_lib::HybridRansacStatistics ransac_stats;

    HybridUncertaintyLOMSAC<PoseAndScale, std::vector<PoseAndScale>, HybridPoseEstimatorScaleOnly> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    return std::make_pair(best_solution, ransac_stats);
}

int HybridPoseEstimator::MinimalSolver(const std::vector<std::vector<int>>& sample,
                                       const int solver_idx, std::vector<PoseScaleOffset>* models) const {
    models->clear();
    if (solver_idx == 0) {
        Eigen::Matrix3d x0 = K0_inv_ * x0_(Eigen::all, sample[0]);
        Eigen::Matrix3d x1 = K1_inv_ * x1_(Eigen::all, sample[0]);

        std::vector<PoseScaleOffset> sols;
        if (est_config_.use_shift) {
            int num_sols = estimate_scale_shift_pose(x0, x1, d0_(sample[0]), d1_(sample[0]), &sols, false);
            for (int i = 0; i < num_sols; i++) {
                if (!est_config_.min_depth_constraint || (sols[i].offset0 > -min_depth_(0) && sols[i].offset1 > -min_depth_(1) * sols[i].scale)) {
                    PoseScaleOffset sol = sols[i];
                    sol.offset1 /= sol.scale;
                    models->push_back(sol);
                }
            }
        }
        else {
            Eigen::Matrix3d p0 = x0.array().rowwise() * d0_(sample[0]).transpose().array();
            Eigen::Matrix3d p1 = x1.array().rowwise() * d1_(sample[0]).transpose().array();
            Eigen::Vector3d v0, v1;
            v0 << (p0.col(0) - p0.col(1)).norm(), (p0.col(0) - p0.col(2)).norm(), (p0.col(1) - p0.col(2)).norm();
            v1 << (p1.col(0) - p1.col(1)).norm(), (p1.col(0) - p1.col(2)).norm(), (p1.col(1) - p1.col(2)).norm();
            // Find the least square scale s so that s * v1 = v0
            double scale = v1.dot(v0) / v1.squaredNorm();

            Eigen::Vector3d d0 = d0_(sample[0]);
            Eigen::Vector3d d1 = d1_(sample[0]) * scale;
            Eigen::Matrix3d X = x0.array().rowwise() * d0.transpose().array();
            Eigen::Matrix3d Y = x1.array().rowwise() * d1.transpose().array();

            Eigen::Vector3d centroid_X = X.rowwise().mean();
            Eigen::Vector3d centroid_Y = Y.rowwise().mean();

            Eigen::MatrixXd X_centered = X.colwise() - centroid_X;
            Eigen::MatrixXd Y_centered = Y.colwise() - centroid_Y;

            Eigen::Matrix3d S = Y_centered * X_centered.transpose();

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            if (U.determinant() * V.determinant() < 0) {
                U.col(2) *= -1;
            }
            Eigen::Matrix3d R = U * V.transpose();
            Eigen::Vector3d t = centroid_Y - R * centroid_X;

            models->push_back(PoseScaleOffset(R, t, scale, 0.0, 0.0));
        }
    }
    else if (solver_idx == 1) {
        Eigen::MatrixXd x0 = K0_inv_ * x0_(Eigen::all, sample[2]);
        Eigen::MatrixXd x1 = K1_inv_ * x1_(Eigen::all, sample[2]);

        std::vector<Eigen::Vector3d> x0_vec(sample[2].size()), x1_vec(sample[2].size());
        std::vector<Eigen::Vector2d> x0_2dvec(sample[2].size()), x1_2dvec(sample[2].size());
        for (int i = 0; i < sample[2].size(); i++) {
            x0_vec[i] = x0.col(i).normalized();
            x1_vec[i] = x1.col(i).normalized();
            x0_2dvec[i] = x0.col(i).head<2>();
            x1_2dvec[i] = x1.col(i).head<2>();
        }
        std::vector<poselib::CameraPose> poses;
        poselib::relpose_5pt(x0_vec, x1_vec, &poses);

        for (auto &pose : poses) {
            Eigen::Matrix3x4d proj_matrix0 = Eigen::Matrix3x4d::Identity();
            Eigen::Matrix3x4d proj_matrix1 = pose.Rt();
            
            std::vector<Eigen::Vector3d> p3d_vec = colmap::TriangulatePoints(proj_matrix0, proj_matrix1, x0_2dvec, x1_2dvec);
            Eigen::MatrixXd p3d(3, p3d_vec.size());
            for (int i = 0; i < p3d_vec.size(); i++) {
                p3d.col(i) = p3d_vec[i];
            }  

            if (!est_config_.use_shift) {
                // Estimate scale by least-squares
                double s0 = d0_(sample[2]).dot(p3d.row(2)) / d0_(sample[2]).squaredNorm();

                p3d = p3d / s0;
                pose.t = pose.t / s0;
                p3d = (pose.R() * p3d).colwise() + pose.t;

                double scale = d1_(sample[2]).dot(p3d.row(2)) / d1_(sample[2]).squaredNorm();

                PoseScaleOffset sol(pose.R(), pose.t, scale, 0.0, 0.0);
                models->push_back(sol);
            }
            else {
                // Estimate scale and shift by least-squares
                Eigen::MatrixXd A(sample[2].size(), 2);
                A.col(0) = d0_(sample[2]);
                A.col(1) = Eigen::VectorXd::Ones(sample[2].size());
                Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * p3d.row(2).transpose());
                double s0 = x(0);
                double offset0 = x(1) / s0;
                if (est_config_.min_depth_constraint && offset0 < -min_depth_(0)) continue;

                p3d = p3d / s0;
                pose.t = pose.t / s0;
                p3d = (pose.R() * p3d).colwise() + pose.t;
                A.col(0) = d1_(sample[2]);
                A.col(1) = Eigen::VectorXd::Ones(sample[2].size());
                x = (A.transpose() * A).ldlt().solve(A.transpose() * p3d.row(2).transpose());
                double scale = x(0);
                double offset1 = x(1) / scale;
                if (est_config_.min_depth_constraint && offset1 < -min_depth_(1)) continue;

                PoseScaleOffset sol(pose.R(), pose.t, scale, offset0, offset1);

                models->push_back(sol);
            }
        }
    }
    return models->size();
}

int HybridPoseEstimator::NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* solution) const {
    if ((sample[0].size() < 3 && sample[1].size() < 3) || sample[2].size() < 5) {
        return 0;
    }

    OptimizerConfig config;
    config.solver_options.max_num_iterations = 25;

    config.use_sampson = true;
    config.use_reprojection = true;
    if (est_config_.LO_type == EstimatorOption::MD_ONLY)
        config.use_sampson = false;
    if (est_config_.LO_type == EstimatorOption::EPI_ONLY)
        config.use_reprojection = false;
    config.weight_sampson = sampson_squared_weight_;
    config.min_depth_constraint = est_config_.min_depth_constraint;
    config.use_shift = est_config_.use_shift;
    PoseScaleOffsetOptimizer3 optim(x0_, x1_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_, 
        *solution, K0_, K1_, config);
    optim.SetUp();
    if (!optim.Solve()) 
        return 0;
    *solution = optim.GetSolution();
    return 1;
}

double HybridPoseEstimator::EvaluateModelOnPoint(const PoseScaleOffset& model, int t, int i, bool is_for_inlier) const {
    if (!is_for_inlier && est_config_.score_type == EstimatorOption::EPI_ONLY && t != 2) {
        return std::numeric_limits<double>::max();
    }
    if (!is_for_inlier && est_config_.score_type == EstimatorOption::MD_ONLY && t == 2) {
        return std::numeric_limits<double>::max();
    }
    if (t == 0) {
        Eigen::Vector3d p3d0 = (K0_inv_ * x0_.col(i)) * (d0_(i) + model.offset0);
        Eigen::Vector3d q = model.R() * p3d0 + model.t();
        Eigen::Vector3d p2d_project = K1_ * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2) 
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x1 = x1_.col(i).head<2>();
        double reproj_error = (p2d - x1).squaredNorm();

        return reproj_error;
    }
    else if (t == 1) {   
        Eigen::Vector3d p3d1 = (K1_inv_ * x1_.col(i)) * (d1_(i) + model.offset1) * model.scale;
        Eigen::Vector3d q = model.R().transpose() * p3d1 - model.R().transpose() * model.t();
        Eigen::Vector3d p2d_project = K0_ * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2) 
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x0 = x0_.col(i).head<2>();
        double reproj_error = (p2d - x0).squaredNorm();

        return reproj_error;
    }
    else if (t == 2) {
        poselib::CameraPose pose(model.R(), model.t());
        Eigen::Vector3d x0_calib = K0_inv_ * x0_.col(i);
        Eigen::Vector3d x1_calib = K1_inv_ * x1_.col(i);
        bool cheirality = poselib::check_cheirality(pose, x0_calib.normalized(), x1_calib.normalized(), 1e-2);
        if (!cheirality) {
            return std::numeric_limits<double>::max();
        }

        Eigen::Matrix3d E = to_essential_matrix(model.R(), model.t());

        double sampson_error = compute_sampson_error(x0_calib.head<2>(), x1_calib.head<2>(), E);
        double loss_scale = 1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1));
        return sampson_error / std::pow(loss_scale, 2);
    }
}

// Linear least squares solver. 
void HybridPoseEstimator::LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseScaleOffset* model) const {
    if ((sample[0].size() < 3 && sample[1].size() < 3) || sample[2].size() < 5) {
        return;
    }

    OptimizerConfig config;
    config.use_sampson = true;
    config.use_reprojection = true;
    if (est_config_.LO_type == EstimatorOption::MD_ONLY)
        config.use_sampson = false;
    if (est_config_.LO_type == EstimatorOption::EPI_ONLY)
        config.use_reprojection = false;
    config.weight_sampson = sampson_squared_weight_;
    config.min_depth_constraint = est_config_.min_depth_constraint;
    config.use_shift = est_config_.use_shift;

    PoseScaleOffsetOptimizer3 optim(x0_, x1_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_, 
        *model, K0_, K1_, config);
    optim.SetUp();
    if (!optim.Solve()) 
        return;
    *model = optim.GetSolution();
}

// **************************************************************
//
// ------------------ Scale only version below ------------------
//
// **************************************************************

int HybridPoseEstimatorScaleOnly::NonMinimalSolver(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseAndScale* solution) const {
    if ((sample[0].size() < 3 && sample[1].size() < 3) || sample[2].size() < 5) {
        return 0;
    }

    OptimizerConfig config;
    config.solver_options.max_num_iterations = 25;

    config.use_sampson = true;
    config.use_reprojection = true;
    if (est_config_.LO_type == EstimatorOption::MD_ONLY)
        config.use_sampson = false;
    if (est_config_.LO_type == EstimatorOption::EPI_ONLY)
        config.use_reprojection = false;
    config.weight_sampson = sampson_squared_weight_;
    PoseAndScaleOptimizer3 optim(x0_, x1_, d0_, d1_, sample[0], sample[1], sample[2], 
        *solution, K0_, K1_, config);
    optim.SetUp();
    if (!optim.Solve()) 
        return 0;
    *solution = optim.GetSolution();
    return 1;
}

int HybridPoseEstimatorScaleOnly::MinimalSolver(const std::vector<std::vector<int>>& sample,
                                                 const int solver_idx, std::vector<PoseAndScale>* models) const {
    models->clear();
    if (solver_idx == 0) {
        Eigen::Matrix3d x0 = K0_inv_ * x0_(Eigen::all, sample[0]);
        Eigen::Matrix3d x1 = K1_inv_ * x1_(Eigen::all, sample[0]);

        PoseAndScale sol = estimate_scale_and_pose(x0, x1, Eigen::VectorXd::Ones(3));        
        sol.scale = 1.0 / sol.scale; // scale now applies on the second camera
        models->push_back(sol);
    }
    else if (solver_idx == 1) {
        Eigen::MatrixXd x0 = K0_inv_ * x0_(Eigen::all, sample[2]);
        Eigen::MatrixXd x1 = K1_inv_ * x1_(Eigen::all, sample[2]);

        std::vector<Eigen::Vector3d> x0_vec(sample[2].size()), x1_vec(sample[2].size());
        std::vector<Eigen::Vector2d> x0_2dvec(sample[2].size()), x1_2dvec(sample[2].size());
        for (int i = 0; i < sample[2].size(); i++) {
            x0_vec[i] = x0.col(i).normalized();
            x1_vec[i] = x1.col(i).normalized();
            x0_2dvec[i] = x0.col(i).head<2>();
            x1_2dvec[i] = x1.col(i).head<2>();
        }
        std::vector<poselib::CameraPose> poses;
        poselib::relpose_5pt(x0_vec, x1_vec, &poses);

        for (auto &pose : poses) {
            Eigen::Matrix3x4d proj_matrix0 = Eigen::Matrix3x4d::Identity();
            Eigen::Matrix3x4d proj_matrix1 = pose.Rt();
            
            std::vector<Eigen::Vector3d> p3d_vec = colmap::TriangulatePoints(proj_matrix0, proj_matrix1, x0_2dvec, x1_2dvec);
            Eigen::MatrixXd p3d(3, p3d_vec.size());
            for (int i = 0; i < p3d_vec.size(); i++) {
                p3d.col(i) = p3d_vec[i];
            }  

            // Estimate scale by least-squares
            double s0 = d0_(sample[2]).dot(p3d.row(2)) / d0_(sample[2]).squaredNorm();

            p3d = p3d / s0;
            pose.t = pose.t / s0;
            p3d = (pose.R() * p3d).colwise() + pose.t;
            double scale = d1_(sample[2]).dot(p3d.row(2)) / d1_(sample[2]).squaredNorm();

            PoseAndScale sol(pose.R(), pose.t, scale);
            models->push_back(sol);
        }
    }
    return models->size();
}

double HybridPoseEstimatorScaleOnly::EvaluateModelOnPoint(const PoseAndScale& model, int t, int i, bool is_for_inlier) const {
    if (!is_for_inlier && est_config_.score_type == EstimatorOption::EPI_ONLY && t != 2) {
        return std::numeric_limits<double>::max();
    }
    if (!is_for_inlier && est_config_.score_type == EstimatorOption::MD_ONLY && t == 2) {
        return std::numeric_limits<double>::max();
    }
    if (t == 0) {
        Eigen::Vector3d p3d0 = (K0_inv_ * x0_.col(i)) * d0_(i);
        Eigen::Vector3d q = model.R() * p3d0 + model.t();
        Eigen::Vector3d p2d_project = K1_ * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2 || d0_(i) < 1e-2)
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x1 = x1_.col(i).head<2>();
        double reproj_error = (p2d - x1).squaredNorm();

        return reproj_error;
    }
    else if (t == 1) {   
        Eigen::Vector3d p3d1 = (K1_inv_ * x1_.col(i)) * d1_(i) * model.scale;
        Eigen::Vector3d q = model.R().transpose() * p3d1 - model.R().transpose() * model.t();
        Eigen::Vector3d p2d_project = K0_ * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2 || d1_(i) < 1e-2)
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x0 = x0_.col(i).head<2>();
        double reproj_error = (p2d - x0).squaredNorm();

        return reproj_error;
    }
    else if (t == 2) {
        poselib::CameraPose pose(model.R(), model.t());
        Eigen::Vector3d x0_calib = K0_inv_ * x0_.col(i);
        Eigen::Vector3d x1_calib = K1_inv_ * x1_.col(i);
        bool cheirality = poselib::check_cheirality(pose, x0_calib.normalized(), x1_calib.normalized(), 1e-2);
        if (!cheirality) {
            return std::numeric_limits<double>::max();
        }

        Eigen::Matrix3d E = to_essential_matrix(model.R(), model.t());

        double sampson_error = compute_sampson_error(x0_calib.head<2>(), x1_calib.head<2>(), E);
        double loss_scale = 1.0 / (K0_(0, 0) + K0_(1, 1)) + 1.0 / (K1_(0, 0) + K1_(1, 1));
        return sampson_error / std::pow(loss_scale, 2);
    }
}

// Linear least squares solver. 
void HybridPoseEstimatorScaleOnly::LeastSquares(const std::vector<std::vector<int>>& sample, const int solver_idx, PoseAndScale* model) const {
    if ((sample[0].size() < 3 && sample[1].size() < 3) || sample[2].size() < 5) {
        return;
    }

    OptimizerConfig config;
    config.use_sampson = true;
    config.use_reprojection = true;
    config.solver_options.max_num_iterations = 25;

    if (est_config_.LO_type == EstimatorOption::MD_ONLY)
        config.use_sampson = false;
    if (est_config_.LO_type == EstimatorOption::EPI_ONLY)
        config.use_reprojection = false;
    config.weight_sampson = sampson_squared_weight_;
    PoseAndScaleOptimizer3 optim(x0_, x1_, d0_, d1_, sample[0], sample[1], sample[2], 
        *model, K0_, K1_, config);
    optim.SetUp();
    if (!optim.Solve()) 
        return;
    *model = optim.GetSolution();
}
} // namespace acmpose