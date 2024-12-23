#ifndef UTILS_H
#define UTILS_H

#include <pybind11/pybind11.h>
#include <colmap/util/threading.h>
#include <ceres/ceres.h>
#include <Eigen/Core>

#define ASSIGN_PYDICT_ITEM(dict,key,type) \
if (dict.contains(#key)) key = dict[#key].cast<type>();

#define ASSIGN_PYDICT_ITEM_TO_MEMBER(obj,dict,key,type) \
  if (dict.contains(#key)) obj.key = dict[#key].cast<type>();

namespace py = pybind11;
namespace acmpose {

inline Eigen::Matrix3d to_essential_matrix(Eigen::Matrix3d R, Eigen::Vector3d t) {
    Eigen::Matrix3d E;
    E << 0.0, -t(2), t(1), t(2), 0.0, -t(0), -t(1), t(0), 0.0;
    E = E * R;
    return E;    
}
 
inline double compute_sampson_error(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2, const Eigen::Matrix3d &E) {
    // For some reason this is a lot faster than just using nice Eigen expressions...
    const double Ex1_0 = E(0, 0) * x1(0) + E(0, 1) * x1(1) + E(0, 2);
    const double Ex1_1 = E(1, 0) * x1(0) + E(1, 1) * x1(1) + E(1, 2);
    const double Ex1_2 = E(2, 0) * x1(0) + E(2, 1) * x1(1) + E(2, 2);

    const double Ex2_0 = E(0, 0) * x2(0) + E(1, 0) * x2(1) + E(2, 0);
    const double Ex2_1 = E(0, 1) * x2(0) + E(1, 1) * x2(1) + E(2, 1);
    // const double Ex2_2 = E(0, 2) * x2(0) + E(1, 2) * x2(1) + E(2, 2);

    const double C = x2(0) * Ex1_0 + x2(1) * Ex1_1 + Ex1_2;
    const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
    const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
    const double r2 = C * C / (Cx + Cy);

    return r2;
}

template <typename T>
Eigen::Vector<T, 4> NormalizeQuaternion(const Eigen::Vector<T, 4>& qvec) {
    const T norm = qvec.norm();
    if (norm == T(0.0)) {
        // We do not just use (1, 0, 0, 0) because that is a constant and when used
        // for automatic differentiation that would lead to a zero derivative.
        return Eigen::Vector<T, 4>(T(1.0), qvec(1), qvec(2), qvec(3));
    } else {
        return qvec / norm;
    }
}

template <typename T>
Eigen::Matrix<T, 3, 3> QuaternionToRotationMatrix(const Eigen::Vector<T, 4>& qvec) {
    const Eigen::Vector<T, 4> normalized_qvec = NormalizeQuaternion(qvec);
    const Eigen::Quaternion<T> quat(normalized_qvec(0), normalized_qvec(1),
                                    normalized_qvec(2), normalized_qvec(3));
    return quat.toRotationMatrix();
}

template <typename T>
Eigen::Matrix<T, 4, 4> ComposeTransformationMatrix(const Eigen::Vector<T, 4>& qvec,
                                                   const Eigen::Vector<T, 3>& tvec) {
    Eigen::Matrix<T, 4, 4> trans = Eigen::Matrix<T, 4, 4>::Identity();
    trans.template block<3, 3>(0, 0) = QuaternionToRotationMatrix(qvec);
    trans.template block<3, 1>(0, 3) = tvec;
    return trans;
}

template <typename T>
Eigen::Vector<T, 4> RotationMatrixToQuaternion(const Eigen::Matrix<T, 3, 3>& R) {
    Eigen::Quaternion<T> q(R);
    return Eigen::Vector<T, 4>(q.w(), q.x(), q.y(), q.z());
}

inline void AssignSolverOptionsFromDict(ceres::Solver::Options& solver_options, py::dict dict) {
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,function_tolerance,double)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,gradient_tolerance,double)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,parameter_tolerance,double)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,minimizer_progress_to_stdout,bool)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,max_linear_solver_iterations,int)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,max_num_iterations,int)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,max_num_consecutive_invalid_steps,int)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,max_consecutive_nonmonotonic_steps,int)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,use_inner_iterations,bool)
    ASSIGN_PYDICT_ITEM_TO_MEMBER(solver_options,dict,inner_iteration_tolerance,double)
}

} // namespace acmpose

#endif // UTILS_H