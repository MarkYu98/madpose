#include "pose_scale_shift_estimator.h"
#include "ransac.h"
#include "solver.h"

namespace acmpose {

std::pair<PoseScaleOffset, ransac_lib::RansacStatistics> 
EstimatePoseScaleOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                               const std::vector<double> &depth0, const std::vector<double> &depth1, 
                               const Eigen::Vector2d &min_depth,
                               const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1, 
                               const ransac_lib::LORansacOptions& options) {
    ransac_lib::LORansacOptions ransac_options(options);
    
    std::random_device rand_dev;
    ransac_options.random_seed_ = 0;

    PoseScaleOffsetEstimator solver(x0, x1, depth0, depth1, min_depth,
        K0, K1);
    solver.squared_inlier_thres = ransac_options.squared_inlier_threshold_;

    PoseScaleOffset best_solution;
    ransac_lib::RansacStatistics ransac_stats;

    LocallyOptimizedMSAC<PoseScaleOffset, std::vector<PoseScaleOffset>, PoseScaleOffsetEstimator> lomsac;
    lomsac.EstimateModel(ransac_options, solver, &best_solution, &ransac_stats);
    // poselib::RansacOptions opt;
    // opt.min_iterations = ransac_options.min_num_iterations_;
    // opt.max_iterations = ransac_options.max_num_iterations_;

    // poselib::RansacStats stats = poselib::ransac<PoseScaleOffsetEstimator>(solver, opt, &best_solution);

    return std::make_pair(best_solution, ransac_stats);
}

int PoseScaleOffsetEstimator::MinimalSolver(const std::vector<int>& sample, std::vector<PoseScaleOffset>* solutions) const {
    Eigen::Matrix3d x0 = K0_inv_ * x0_(Eigen::all, sample);
    Eigen::Matrix3d x1 = K1_inv_ * x1_(Eigen::all, sample);

    // Or do we estimate scale and shift and then p3p here?
    // note: scale on y
    solutions->clear();
    std::vector<PoseScaleOffset> sols;
    int num_sols = estimate_scale_shift_pose(x0, x1, d0_(sample), d1_(sample), &sols, false);
    int valid_sols = 0; 
    for (int i = 0; i < num_sols; i++) {
        if (sols[i].offset0 > -min_depth_(0) && sols[i].offset1 > -min_depth_(1) * sols[i].scale) {
            PoseScaleOffset sol = sols[i];
            sol.offset1 = sol.offset1 / sol.scale;
            solutions->push_back(sol);
            valid_sols++;
        }
    }
    return valid_sols;
}

int PoseScaleOffsetEstimator::NonMinimalSolver(const std::vector<int>& sample, PoseScaleOffset* solution) const {
    OptimizerConfig config;
    config.use_geometric = false;
    PoseScaleOffsetOptimizer optim(x0_, x1_, d0_, d1_, sample, {}, min_depth_(0), min_depth_(1), 
        *solution, K0_, K1_, config);
    optim.SetUp();
    if (!optim.Solve()) return 0;
    *solution = optim.GetSolution();
    return 1;
}

double PoseScaleOffsetEstimator::EvaluateModelOnPoint(const PoseScaleOffset& solution, int i) const {
    Eigen::Vector3d p3d0 = (K0_inv_ * x0_.col(i)) * (d0_(i) + solution.offset0);
    p3d0 = solution.R() * p3d0 + solution.t();
    Eigen::Vector3d p2d_project = K1_ * p3d0;
    if (p2d_project(2) < 1e-2) 
        return std::numeric_limits<double>::max();
    Eigen::Vector2d p2d = p2d_project.head<2>() / p2d_project(2);
    double r0 = (p2d - x1_.col(i).head<2>()).squaredNorm();
    
    Eigen::Vector3d p3d1 = (K1_inv_ * x1_.col(i)) * (d1_(i) + solution.offset1) * solution.scale;
    p2d_project = K0_ * (solution.R().transpose() * p3d1 - solution.R().transpose() * solution.t());
    if (p2d_project(2) < 1e-2) 
        return std::numeric_limits<double>::max();
    p2d = p2d_project.head<2>() / p2d_project(2);
    double r1 = (p2d - x0_.col(i).head<2>()).squaredNorm();
    return std::min(r0, r1);
}

// Linear least squares solver. 
void PoseScaleOffsetEstimator::LeastSquares(const std::vector<int>& sample, PoseScaleOffset* solution) const {
    OptimizerConfig config;
    config.use_geometric = false;
    config.weight_geometric = 1.0;
    PoseScaleOffsetOptimizer optim(x0_, x1_, d0_, d1_, sample, {}, min_depth_(0), min_depth_(1), 
        *solution, K0_, K1_, config);
    optim.SetUp();
    if (!optim.Solve()) return;
    *solution = optim.GetSolution();
}

} // namespace acmpose