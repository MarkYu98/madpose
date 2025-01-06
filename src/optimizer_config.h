#ifndef OPTIMIZER_CONFIG_H
#define OPTIMIZER_CONFIG_H

#include "utils.h"

namespace madpose {

class OptimizerConfig {
public:
    OptimizerConfig() {
        solver_options.function_tolerance = 1e-6;
        solver_options.gradient_tolerance = 1e-8;
        solver_options.parameter_tolerance = 1e-6;
        solver_options.minimizer_progress_to_stdout = true;
        solver_options.max_num_iterations = 100;
        solver_options.use_nonmonotonic_steps = true;
        solver_options.num_threads = -1;
        solver_options.logging_type = ceres::SILENT;
    #if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads = -1;
    #endif  // CERES_VERSION_MAJOR
        problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
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

    double weight_sampson = 1.0;
    std::shared_ptr<ceres::LossFunction> reproj_loss_function;
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

} // namespace madpose

#endif // OPTIMIZER_CONFIG_H