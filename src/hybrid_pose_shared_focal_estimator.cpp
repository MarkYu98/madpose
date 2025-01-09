#include "hybrid_pose_shared_focal_estimator.h"

#include <PoseLib/poselib.h>
#include <PoseLib/solvers/relpose_6pt_focal.h>

namespace madpose {

std::pair<PoseScaleOffsetSharedFocal, ransac_lib::HybridRansacStatistics> HybridEstimatePoseScaleOffsetSharedFocal(
    const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1, const std::vector<double> &depth0,
    const std::vector<double> &depth1, const Eigen::Vector2d &min_depth, const Eigen::Vector2d &pp0,
    const Eigen::Vector2d &pp1, const ExtendedHybridLORansacOptions &options, const EstimatorConfig &est_config) {
    ExtendedHybridLORansacOptions ransac_options(options);

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
    ransac_options.data_type_weights_[1] *=
        2 * ransac_options.squared_inlier_thresholds_[0] / ransac_options.squared_inlier_thresholds_[1];
    double sampson_squared_weight = ransac_options.data_type_weights_[1];
    ransac_options.data_type_weights_.push_back(ransac_options.data_type_weights_[1]);
    ransac_options.squared_inlier_thresholds_.push_back(ransac_options.squared_inlier_thresholds_[1]);
    ransac_options.data_type_weights_[1] = ransac_options.data_type_weights_[0];
    ransac_options.squared_inlier_thresholds_[1] = ransac_options.squared_inlier_thresholds_[0];

    HybridSharedFocalPoseEstimator solver(x0_norm, x1_norm, depth0, depth1, min_depth, norm_scale,
                                          sampson_squared_weight, ransac_options.squared_inlier_thresholds_,
                                          est_config);

    PoseScaleOffsetSharedFocal best_solution;
    ransac_lib::HybridRansacStatistics ransac_stats;

    // HybridLOMSAC<PoseScaleOffsetSharedFocal, std::vector<PoseScaleOffsetSharedFocal>,
    // HybridSharedFocalPoseEstimator> lomsac;
    HybridLOMSAC<PoseScaleOffsetSharedFocal, std::vector<PoseScaleOffsetSharedFocal>, HybridSharedFocalPoseEstimator>
        lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);

    best_solution.focal *= norm_scale;
    return std::make_pair(best_solution, ransac_stats);
}

int HybridSharedFocalPoseEstimator::MinimalSolver(const std::vector<std::vector<int>> &sample, const int solver_idx,
                                                  std::vector<PoseScaleOffsetSharedFocal> *models) const {
    models->clear();
    if (solver_idx == 0) {
        Eigen::Matrix3x4d x0 = x0_norm_(Eigen::all, sample[0]);
        Eigen::Matrix3x4d x1 = x1_norm_(Eigen::all, sample[0]);

        std::vector<PoseScaleOffsetSharedFocal> sols;
        int num_sols = solve_scale_shift_pose_shared_focal(x0, x1, d0_(sample[0]), d1_(sample[0]), &sols, false);
        for (int i = 0; i < num_sols; i++) {
            if (!est_config_.min_depth_constraint ||
                (sols[i].offset0 > -min_depth_(0) && sols[i].offset1 > -min_depth_(1) * sols[i].scale)) {
                PoseScaleOffsetSharedFocal sol = sols[i];
                sol.offset1 /= sol.scale;
                models->push_back(sol);
            }
        }
    } else if (solver_idx == 1) {
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
        std::vector<poselib::ImagePair> image_pairs;
        poselib::relpose_6pt_shared_focal(x0_vec, x1_vec, &image_pairs);

        for (auto &ip : image_pairs) {
            poselib::CameraPose pose = ip.pose;
            double f = ip.camera1.focal();

            Eigen::Matrix3d K;
            K << f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0;
            Eigen::Matrix3x4d proj_matrix0 = K * Eigen::Matrix3x4d::Identity();
            Eigen::Matrix3x4d proj_matrix1 = K * pose.Rt();

            std::vector<Eigen::Vector3d> p3d_vec = TriangulatePoints(proj_matrix0, proj_matrix1, x0_2dvec, x1_2dvec);
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
            if (est_config_.min_depth_constraint && offset0 < -min_depth_(0))
                continue;

            p3d = p3d / s0;
            pose.t = pose.t / s0;
            p3d = (pose.R() * p3d).colwise() + pose.t;
            A.col(0) = d1_(sample[2]);
            A.col(1) = Eigen::VectorXd::Ones(sample[2].size());
            x = (A.transpose() * A).ldlt().solve(A.transpose() * p3d.row(2).transpose());
            double scale = x(0);
            double offset1 = x(1) / scale;
            if (est_config_.min_depth_constraint && offset1 < -min_depth_(1))
                continue;

            PoseScaleOffsetSharedFocal sol(pose.R(), pose.t, scale, offset0, offset1, f);
            models->push_back(sol);
        }
    }
    return models->size();
}

int HybridSharedFocalPoseEstimator::NonMinimalSolver(const std::vector<std::vector<int>> &sample, const int solver_idx,
                                                     PoseScaleOffsetSharedFocal *solution) const {
    if ((sample[0].size() < 4 && sample[1].size() < 4) || sample[2].size() < 6) {
        return 0;
    }
    SharedFocalOptimizerConfig config;
    config.solver_options.max_num_iterations = 25;
    config.use_sampson = true;
    config.use_reprojection = true;
    if (est_config_.LO_type == EstimatorOption::MD_ONLY)
        config.use_sampson = false;
    if (est_config_.LO_type == EstimatorOption::EPI_ONLY)
        config.use_reprojection = false;
    config.weight_sampson = sampson_squared_weight_;
    config.min_depth_constraint = est_config_.min_depth_constraint;
    HybridSharedFocalPoseOptimizer optim(x0_norm_, x1_norm_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_,
                                         *solution, config);
    optim.SetUp();
    if (!optim.Solve())
        return 0;
    *solution = optim.GetSolution();

    return 1;
}

double HybridSharedFocalPoseEstimator::EvaluateModelOnPoint(const PoseScaleOffsetSharedFocal &model, int t, int i,
                                                            bool is_for_inlier) const {
    if (!is_for_inlier && est_config_.score_type == EstimatorOption::EPI_ONLY && t != 2) {
        return std::numeric_limits<double>::max();
    }
    if (!is_for_inlier && est_config_.score_type == EstimatorOption::MD_ONLY && t == 2) {
        return std::numeric_limits<double>::max();
    }

    Eigen::Matrix3d K_inv, K;
    K << model.focal, 0.0, 0.0, 0.0, model.focal, 0.0, 0.0, 0.0, 1.0;
    K_inv << 1.0 / model.focal, 0.0, 0.0, 0.0, 1.0 / model.focal, 0.0, 0.0, 0.0, 1.0;

    if (t == 0) {
        Eigen::Vector3d p3d0 = (K_inv * x0_norm_.col(i)) * (d0_(i) + model.offset0);
        Eigen::Vector3d q = model.R() * p3d0 + model.t();
        Eigen::Vector3d p2d_project = K * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2)
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x1 = x1_norm_.col(i).head<2>();
        double reproj_error = (p2d - x1).squaredNorm();
        return reproj_error;
    } else if (t == 1) {
        Eigen::Vector3d p3d1 = (K_inv * x1_norm_.col(i)) * (d1_(i) + model.offset1) * model.scale;
        Eigen::Vector3d q = model.R().transpose() * p3d1 - model.R().transpose() * model.t();
        Eigen::Vector3d p2d_project = K * q;
        Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
        double z = p2d_project(2);
        if (z < 1e-2)
            return std::numeric_limits<double>::max();
        Eigen::Vector2d x0 = x0_norm_.col(i).head<2>();
        double reproj_error = (p2d - x0).squaredNorm();
        return reproj_error;
    } else if (t == 2) {
        Eigen::Matrix3d E = to_essential_matrix(model.R(), model.t());
        Eigen::Matrix3d F = K_inv.transpose() * E * K_inv;

        double sampson_error = compute_sampson_error(x0_norm_.col(i).head<2>(), x1_norm_.col(i).head<2>(), F);
        return sampson_error;
    }
}

// Linear least squares solver.
void HybridSharedFocalPoseEstimator::LeastSquares(const std::vector<std::vector<int>> &sample, const int solver_idx,
                                                  PoseScaleOffsetSharedFocal *model) const {
    if ((sample[0].size() < 4 && sample[1].size() < 4) || sample[2].size() < 6) {
        return;
    }

    SharedFocalOptimizerConfig config;
    config.use_sampson = true;
    config.use_reprojection = true;
    config.solver_options.max_num_iterations = 25;

    if (est_config_.LO_type == EstimatorOption::MD_ONLY)
        config.use_sampson = false;
    if (est_config_.LO_type == EstimatorOption::EPI_ONLY)
        config.use_reprojection = false;
    config.weight_sampson = sampson_squared_weight_;
    config.min_depth_constraint = est_config_.min_depth_constraint;
    HybridSharedFocalPoseOptimizer optim(x0_norm_, x1_norm_, d0_, d1_, sample[0], sample[1], sample[2], min_depth_,
                                         *model, config);
    optim.SetUp();
    if (!optim.Solve())
        return;
    *model = optim.GetSolution();
}

} // namespace madpose