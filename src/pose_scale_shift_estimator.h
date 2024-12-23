#ifndef POSE_SCALE_SHIFT_ESTIMATOR_H
#define POSE_SCALE_SHIFT_ESTIMATOR_H

#include <RansacLib/ransac.h>
#include "optimizer.h"

namespace acmpose {

class PoseScaleOffsetEstimator {
public:
    PoseScaleOffsetEstimator(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                            const std::vector<double> &depth0, const std::vector<double> &depth1, 
                            const Eigen::Vector2d &min_depth,
                            const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1) : 
                            K0_(K0), K1_(K1), K0_inv_(K0.inverse()), K1_inv_(K1.inverse()), min_depth_(min_depth) { 
                            assert(x0.size() == x1.size() && x0.size() == depth0.size() && x0.size() == depth1.size());
                            
                            d0_ = Eigen::Map<const Eigen::VectorXd>(depth0.data(), depth0.size());
                            d1_ = Eigen::Map<const Eigen::VectorXd>(depth1.data(), depth1.size());

                            x0_ = Eigen::MatrixXd(3, x0.size());
                            x1_ = Eigen::MatrixXd(3, x1.size());
                            for (int i = 0; i < x0.size(); i++) {
                                x0_.col(i) = Eigen::Vector3d(x0[i](0), x0[i](1), 1.0);
                                x1_.col(i) = Eigen::Vector3d(x1[i](0), x1[i](1), 1.0);
                            }
                         }  

    inline int min_sample_size() const { return 3; }

    inline int non_minimal_sample_size() const { return 10; }

    inline int num_data() const { return x0_.cols(); }

    int MinimalSolver(const std::vector<int>& sample, std::vector<PoseScaleOffset>* solutions) const;

    // Returns 0 if no model could be estimated and 1 otherwise.
    // Implemented by a simple linear least squares solver.
    int NonMinimalSolver(const std::vector<int>& sample, PoseScaleOffset* solution) const;

    // Evaluates the line on the i-th data point.
    double EvaluateModelOnPoint(const PoseScaleOffset& solution, int i) const;

    // Linear least squares solver. 
    void LeastSquares(const std::vector<int>& sample, PoseScaleOffset* solution) const;
    
    const size_t sample_sz = 3;
    double squared_inlier_thres;

private:
    Eigen::Matrix3d K0_, K1_;
    Eigen::Matrix3d K0_inv_, K1_inv_;
    Eigen::MatrixXd x0_, x1_;
    Eigen::VectorXd d0_, d1_;
    Eigen::Vector2d min_depth_;
};

std::pair<PoseScaleOffset, ransac_lib::RansacStatistics>
EstimatePoseScaleOffset(const std::vector<Eigen::Vector2d> &x0, const std::vector<Eigen::Vector2d> &x1,
                        const std::vector<double> &depth0, const std::vector<double> &depth1, const Eigen::Vector2d &min_depth,
                        const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
                        const ransac_lib::LORansacOptions& options);

} // namespace acmpose

#endif // POSE_SCALE_SHIFT_ESTIMATOR_H