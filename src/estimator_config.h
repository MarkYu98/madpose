#ifndef ESTIMATOR_CONFIG_H
#define ESTIMATOR_CONFIG_H

namespace madpose {

enum class EstimatorOption {
    HYBRID = 0,
    EPI_ONLY = 1,
    MD_ONLY = 2
};

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
};

} // namespace madpose

#endif // ESTIMATOR_CONFIG_H