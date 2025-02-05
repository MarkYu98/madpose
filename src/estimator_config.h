#pragma once

namespace madpose {

enum class EstimatorOption { HYBRID = 0, EPI_ONLY = 1, MD_ONLY = 2 };

class EstimatorConfig {
  public:
    EstimatorConfig() {
        solver_type = EstimatorOption::HYBRID;
        score_type = EstimatorOption::HYBRID;
        LO_type = EstimatorOption::HYBRID;
    }
    EstimatorConfig(int solver, int score, int LO) {
        solver_type = static_cast<EstimatorOption>(solver);
        score_type = static_cast<EstimatorOption>(score);
        LO_type = static_cast<EstimatorOption>(LO);
    }

    EstimatorOption solver_type;
    EstimatorOption score_type;
    EstimatorOption LO_type;

    bool min_depth_constraint = true;
    bool use_shift = true;

    double ceres_function_tolerance = 1e-6;
    double ceres_gradient_tolerance = 1e-8;
    double ceres_parameter_tolerance = 1e-6;
    double ceres_max_num_iterations = 25;
    bool ceres_use_nonmonotonic_steps = true;
    int ceres_num_threads = 1;
};

} // namespace madpose
