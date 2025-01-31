#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <pybind11/pybind11.h>

#define ASSIGN_PYDICT_ITEM(dict, key, type)                                                                            \
    if (dict.contains(#key))                                                                                           \
        key = dict[#key].cast<type>();

#define ASSIGN_PYDICT_ITEM_TO_MEMBER(obj, dict, key, type)                                                             \
    if (dict.contains(#key))                                                                                           \
        obj.key = dict[#key].cast<type>();

namespace Eigen {
using Matrix3x4f = Matrix<float, 3, 4>;
using Matrix3x4d = Matrix<double, 3, 4>;
} // namespace Eigen

namespace py = pybind11;
namespace madpose {

// -------- Triangulation functions from COLMAP 3.9 --------
template <typename T>
inline Eigen::Vector<T, 3> TriangulatePoint(const Eigen::Matrix<T, 3, 4> &cam1_from_world,
                                            const Eigen::Matrix<T, 3, 4> &cam2_from_world,
                                            const Eigen::Vector<T, 2> &point1, const Eigen::Vector<T, 2> &point2) {
    Eigen::Matrix<T, 4, 4> A;

    A.row(0) = point1(0) * cam1_from_world.row(2) - cam1_from_world.row(0);
    A.row(1) = point1(1) * cam1_from_world.row(2) - cam1_from_world.row(1);
    A.row(2) = point2(0) * cam2_from_world.row(2) - cam2_from_world.row(0);
    A.row(3) = point2(1) * cam2_from_world.row(2) - cam2_from_world.row(1);

    Eigen::JacobiSVD<Eigen::Matrix<T, 4, 4>> svd(A, Eigen::ComputeFullV);

    return svd.matrixV().col(3).hnormalized();
}

template <typename T>
inline std::vector<Eigen::Vector<T, 3>>
TriangulatePoints(const Eigen::Matrix<T, 3, 4> &cam1_from_world, const Eigen::Matrix<T, 3, 4> &cam2_from_world,
                  const std::vector<Eigen::Vector<T, 2>> &points1, const std::vector<Eigen::Vector<T, 2>> &points2) {
    CHECK_EQ(points1.size(), points2.size());

    std::vector<Eigen::Vector<T, 3>> points3D(points1.size());

    for (size_t i = 0; i < points3D.size(); ++i) {
        points3D[i] = TriangulatePoint(cam1_from_world, cam2_from_world, points1[i], points2[i]);
    }

    return points3D;
}
// ---------------------------------------------------------

template <typename T>
inline Eigen::Matrix<T, 3, 3> to_essential_matrix(Eigen::Matrix<T, 3, 3> R, Eigen::Vector<T, 3> t) {
    Eigen::Matrix<T, 3, 3> E;
    E << 0.0, -t(2), t(1), t(2), 0.0, -t(0), -t(1), t(0), 0.0;
    E = E * R;
    return E;
}

template <typename T>
inline T compute_sampson_error(const Eigen::Vector<T, 2> &x1, const Eigen::Vector<T, 2> &x2,
                               const Eigen::Matrix<T, 3, 3> &E) {
    // For some reason this is a lot faster than just using nice Eigen
    // expressions...
    const T Ex1_0 = E(0, 0) * x1(0) + E(0, 1) * x1(1) + E(0, 2);
    const T Ex1_1 = E(1, 0) * x1(0) + E(1, 1) * x1(1) + E(1, 2);
    const T Ex1_2 = E(2, 0) * x1(0) + E(2, 1) * x1(1) + E(2, 2);

    const T Ex2_0 = E(0, 0) * x2(0) + E(1, 0) * x2(1) + E(2, 0);
    const T Ex2_1 = E(0, 1) * x2(0) + E(1, 1) * x2(1) + E(2, 1);
    // const T Ex2_2 = E(0, 2) * x2(0) + E(1, 2) * x2(1) + E(2, 2);

    const T C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
    const T Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
    const T Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
    const T r2 = C * C / (Cx + Cy);

    return r2;
}

template <typename T> Eigen::Vector<T, 4> NormalizeQuaternion(const Eigen::Vector<T, 4> &qvec) {
    const T norm = qvec.norm();
    if (norm == T(0.0)) {
        // We do not just use (1, 0, 0, 0) because that is a constant and when
        // used for automatic differentiation that would lead to a zero
        // derivative.
        return Eigen::Vector<T, 4>(T(1.0), qvec(1), qvec(2), qvec(3));
    } else {
        return qvec / norm;
    }
}

template <typename T> Eigen::Matrix<T, 3, 3> QuaternionToRotationMatrix(const Eigen::Vector<T, 4> &qvec) {
    const Eigen::Vector<T, 4> normalized_qvec = NormalizeQuaternion(qvec);
    const Eigen::Quaternion<T> quat(normalized_qvec(0), normalized_qvec(1), normalized_qvec(2), normalized_qvec(3));
    return quat.toRotationMatrix();
}

template <typename T>
Eigen::Matrix<T, 4, 4> ComposeTransformationMatrix(const Eigen::Vector<T, 4> &qvec, const Eigen::Vector<T, 3> &tvec) {
    Eigen::Matrix<T, 4, 4> trans = Eigen::Matrix<T, 4, 4>::Identity();
    trans.template block<3, 3>(0, 0) = QuaternionToRotationMatrix(qvec);
    trans.template block<3, 1>(0, 3) = tvec;
    return trans;
}

template <typename T> Eigen::Vector<T, 4> RotationMatrixToQuaternion(const Eigen::Matrix<T, 3, 3> &R) {
    Eigen::Quaternion<T> q(R);
    return Eigen::Vector<T, 4>(q.w(), q.x(), q.y(), q.z());
}

inline void AssignSolverOptionsFromDict(ceres::Solver::Options &solver_options, py::dict dict) {
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, function_tolerance, double)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, gradient_tolerance, double)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, parameter_tolerance, double)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, minimizer_progress_to_stdout, bool)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, max_linear_solver_iterations, int)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, max_num_iterations, int)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, max_num_consecutive_invalid_steps, int)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, max_consecutive_nonmonotonic_steps, int)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, use_inner_iterations, bool)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options, dict, inner_iteration_tolerance, double)
}

} // namespace madpose
