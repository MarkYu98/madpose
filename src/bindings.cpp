#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>

#include <RansacLib/ransac.h>
#include "solver.h"
#include "hybrid_pose_estimator.h"
#include "hybrid_pose_shared_focal_estimator.h"
#include "hybrid_pose_two_focal_estimator.h"

namespace py = pybind11;
using namespace py::literals;

namespace madpose {

void bind_estimator(py::module &);
void bind_ba(py::module &);
void bind_ransaclib(py::module& m);

PYBIND11_MODULE(madpose, m) {
    m.doc() = "Solvers and estimators for relative pose estimation through affine correction of monocular depth priors.";

    py::add_ostream_redirect(m, "ostream_redirect");

    // bind modules
    bind_ransaclib(m);
    bind_estimator(m);
}

void bind_ransaclib(py::module& m) {
    py::class_<ransac_lib::RansacStatistics>(m, "RansacStats")
        .def(py::init<>())
        .def_readwrite("num_iterations", &ransac_lib::RansacStatistics::num_iterations)
        .def_readwrite("best_num_inliers", &ransac_lib::RansacStatistics::best_num_inliers)
        .def_readwrite("best_model_score", &ransac_lib::RansacStatistics::best_model_score)
        .def_readwrite("inlier_ratio", &ransac_lib::RansacStatistics::inlier_ratio)
        .def_readwrite("inlier_indices", &ransac_lib::RansacStatistics::inlier_indices)
        .def_readwrite("number_lo_iterations", &ransac_lib::RansacStatistics::number_lo_iterations);
    
    py::class_<ransac_lib::RansacOptions>(m, "RansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ransac_lib::RansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ransac_lib::RansacOptions::max_num_iterations_)
        .def_readwrite("success_probability_", &ransac_lib::RansacOptions::success_probability_)
        .def_readwrite("squared_inlier_threshold_", &ransac_lib::RansacOptions::squared_inlier_threshold_)
        .def_readwrite("random_seed_", &ransac_lib::RansacOptions::random_seed_);

    py::class_<ransac_lib::LORansacOptions>(m, "LORansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ransac_lib::LORansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ransac_lib::LORansacOptions::max_num_iterations_)
        .def_readwrite("success_probability_", &ransac_lib::LORansacOptions::success_probability_)
        .def_readwrite("squared_inlier_threshold_", &ransac_lib::LORansacOptions::squared_inlier_threshold_)
        .def_readwrite("random_seed_", &ransac_lib::LORansacOptions::random_seed_)
        .def_readwrite("num_lo_steps_", &ransac_lib::LORansacOptions::num_lo_steps_)
        .def_readwrite("threshold_multiplier_", &ransac_lib::LORansacOptions::threshold_multiplier_)
        .def_readwrite("num_lsq_iterations_", &ransac_lib::LORansacOptions::num_lsq_iterations_)
        .def_readwrite("min_sample_multiplicator_", &ransac_lib::LORansacOptions::min_sample_multiplicator_)
        .def_readwrite("non_min_sample_multiplier_", &ransac_lib::LORansacOptions::non_min_sample_multiplier_)
        .def_readwrite("lo_starting_iterations_", &ransac_lib::LORansacOptions::lo_starting_iterations_)
        .def_readwrite("final_least_squares_", &ransac_lib::LORansacOptions::final_least_squares_);
    
    // hybrid ransac
    py::class_<ransac_lib::HybridRansacStatistics>(m, "HybridRansacStatistics")
        .def(py::init<>())
        .def_readwrite("num_iterations_total", &ransac_lib::HybridRansacStatistics::num_iterations_total)
        .def_readwrite("num_iterations_per_solver", &ransac_lib::HybridRansacStatistics::num_iterations_per_solver)
        .def_readwrite("best_num_inliers", &ransac_lib::HybridRansacStatistics::best_num_inliers)
        .def_readwrite("best_solver_type", &ransac_lib::HybridRansacStatistics::best_solver_type)
        .def_readwrite("best_model_score", &ransac_lib::HybridRansacStatistics::best_model_score)
        .def_readwrite("inlier_ratios", &ransac_lib::HybridRansacStatistics::inlier_ratios)
        .def_readwrite("inlier_indices", &ransac_lib::HybridRansacStatistics::inlier_indices)
        .def_readwrite("number_lo_iterations", &ransac_lib::HybridRansacStatistics::number_lo_iterations);
    
    py::class_<ExtendedHybridLORansacOptions>(m, "HybridLORansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ExtendedHybridLORansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ExtendedHybridLORansacOptions::max_num_iterations_)
        .def_readwrite("max_num_iterations_per_solver_", &ExtendedHybridLORansacOptions::max_num_iterations_per_solver_)
        .def_readwrite("success_probability_", &ExtendedHybridLORansacOptions::success_probability_)
        .def_readwrite("squared_inlier_thresholds_", &ExtendedHybridLORansacOptions::squared_inlier_thresholds_)
        .def_readwrite("data_type_weights_", &ExtendedHybridLORansacOptions::data_type_weights_)
        .def_readwrite("random_seed_", &ExtendedHybridLORansacOptions::random_seed_)
        .def_readwrite("num_lo_steps_", &ExtendedHybridLORansacOptions::num_lo_steps_)
        .def_readwrite("threshold_multiplier_", &ExtendedHybridLORansacOptions::threshold_multiplier_)
        .def_readwrite("num_lsq_iterations_", &ExtendedHybridLORansacOptions::num_lsq_iterations_)
        .def_readwrite("min_sample_multiplicator_", &ExtendedHybridLORansacOptions::min_sample_multiplicator_)
        .def_readwrite("non_min_sample_multiplier_", &ExtendedHybridLORansacOptions::non_min_sample_multiplier_)
        .def_readwrite("lo_starting_iterations_", &ExtendedHybridLORansacOptions::lo_starting_iterations_)
        .def_readwrite("final_least_squares_", &ExtendedHybridLORansacOptions::final_least_squares_);
}

void bind_estimator(py::module& m) {
    py::class_<OptimizerConfig>(m, "OptimizerConfig")
    .def(py::init<>())
    .def_readwrite("constant_pose", &OptimizerConfig::constant_pose)
    .def_readwrite("constant_scale", &OptimizerConfig::constant_scale)
    .def_readwrite("constant_offset", &OptimizerConfig::constant_offset)
    .def_readwrite("solver_options", &OptimizerConfig::solver_options)
    .def_readwrite("min_depth_constraint", &OptimizerConfig::min_depth_constraint)
    .def_readwrite("use_shift", &OptimizerConfig::use_shift);

    py::class_<EstimatorConfig>(m, "EstimatorConfig")
    .def(py::init<>())
    .def(py::init<int, int, int>(), "solver"_a = 0, "score"_a = 0, "LO"_a = 0)
    .def_readwrite("min_depth_constraint", &EstimatorConfig::min_depth_constraint)
    .def_readwrite("use_shift", &EstimatorConfig::use_shift);

    py::class_<PoseAndScale>(m, "PoseAndScale")
    .def(py::init<>())
    .def(py::init<const Eigen::Matrix<double, 3, 4>&, double>())
    .def(py::init<const Eigen::Matrix3d&, const Eigen::Vector3d&, double>())
    .def_readwrite("pose", &PoseAndScale::pose)
    .def_readwrite("scale", &PoseAndScale::scale)
    .def("R", &PoseAndScale::R)
    .def("t", &PoseAndScale::t);

    py::class_<PoseScaleOffset>(m, "PoseScaleOffset")
    .def(py::init<>())
    .def(py::init<const Eigen::Matrix<double, 3, 4>&, double, double, double>())
    .def(py::init<const Eigen::Matrix3d&, const Eigen::Vector3d&, double, double, double>())
    .def_readwrite("pose", &PoseScaleOffset::pose)
    .def_readwrite("scale", &PoseScaleOffset::scale)
    .def_readwrite("offset0", &PoseScaleOffset::offset0)
    .def_readwrite("offset1", &PoseScaleOffset::offset1)
    .def("R", &PoseScaleOffset::R)
    .def("t", &PoseScaleOffset::t);

    py::class_<PoseScaleOffsetSharedFocal>(m, "PoseScaleOffsetSharedFocal")
    .def(py::init<>())
    .def(py::init<const Eigen::Matrix<double, 3, 4>&, double, double, double, double>())
    .def(py::init<const Eigen::Matrix3d&, const Eigen::Vector3d&, double, double, double, double>())
    .def_readwrite("pose", &PoseScaleOffsetSharedFocal::pose)
    .def_readwrite("scale", &PoseScaleOffsetSharedFocal::scale)
    .def_readwrite("offset0", &PoseScaleOffsetSharedFocal::offset0)
    .def_readwrite("offset1", &PoseScaleOffsetSharedFocal::offset1)
    .def_readwrite("focal", &PoseScaleOffsetSharedFocal::focal)
    .def("R", &PoseScaleOffsetSharedFocal::R)
    .def("t", &PoseScaleOffsetSharedFocal::t);

    py::class_<PoseScaleOffsetTwoFocal>(m, "PoseScaleOffsetTwoFocal")
    .def(py::init<>())
    .def(py::init<const Eigen::Matrix<double, 3, 4>&, double, double, double, double, double>())
    .def(py::init<const Eigen::Matrix3d&, const Eigen::Vector3d&, double, double, double, double, double>())
    .def_readwrite("pose", &PoseScaleOffsetTwoFocal::pose)
    .def_readwrite("scale", &PoseScaleOffsetTwoFocal::scale)
    .def_readwrite("offset0", &PoseScaleOffsetTwoFocal::offset0)
    .def_readwrite("offset1", &PoseScaleOffsetTwoFocal::offset1)
    .def_readwrite("focal0", &PoseScaleOffsetTwoFocal::focal0)
    .def_readwrite("focal1", &PoseScaleOffsetTwoFocal::focal1)
    .def("R", &PoseScaleOffsetTwoFocal::R)
    .def("t", &PoseScaleOffsetTwoFocal::t);

    m.def("estimate_scale_and_pose", &estimate_scale_and_pose, "X"_a, "Y"_a, "W"_a);
    m.def("solve_scale_and_shift", &solve_scale_and_shift, "x_homo"_a, "y_homo"_a, "depth_x"_a, "depth_y"_a);
    m.def("solve_scale_and_shift_shared_focal", &solve_scale_and_shift_shared_focal, "x_homo"_a, "y_homo"_a, "depth_x"_a, "depth_y"_a);
    m.def("solve_scale_and_shift_two_focal", &solve_scale_and_shift_two_focal, "x_homo"_a, "y_homo"_a, "depth_x"_a, "depth_y"_a);
    m.def("estimate_scale_shift_pose", &estimate_scale_shift_pose_wrapper, "x_homo"_a, "y_homo"_a, "depth_x"_a, "depth_y"_a);
    m.def("estimate_scale_shift_pose_shared_focal", &estimate_scale_shift_pose_shared_focal_wrapper,
        "x_homo"_a, "y_homo"_a, "depth_x"_a, "depth_y"_a);
    m.def("estimate_scale_shift_pose_two_focal", &estimate_scale_shift_pose_two_focal_wrapper,
        "x_homo"_a, "y_homo"_a, "depth_x"_a, "depth_y"_a);
    m.def("HybridEstimatePoseAndScale", &HybridEstimatePoseAndScale, "x0"_a, "x1"_a, "depth0"_a, "depth1"_a,  
        "K0"_a, "K1"_a, "options"_a, "est_config"_a = EstimatorConfig());
    m.def("HybridEstimatePoseScaleOffset", &HybridEstimatePoseScaleOffset, "x0"_a, "x1"_a, "depth0"_a, "depth1"_a, "min_depth"_a, 
        "K0"_a, "K1"_a, "options"_a, "est_config"_a = EstimatorConfig());
    m.def("HybridEstimatePoseScaleOffsetSharedFocal", &HybridEstimatePoseScaleOffsetSharedFocal, "x0"_a, "x1"_a, "depth0"_a, "depth1"_a, "min_depth"_a, 
        "pp0"_a, "pp1"_a, "options"_a, "est_config"_a = EstimatorConfig());
    m.def("HybridEstimatePoseScaleOffsetTwoFocal", &HybridEstimatePoseScaleOffsetTwoFocal, "x0"_a, "x1"_a, "depth0"_a, "depth1"_a, "min_depth"_a, 
        "pp0"_a, "pp1"_a, "options"_a, "est_config"_a = EstimatorConfig());
}

} // namespace madpose

