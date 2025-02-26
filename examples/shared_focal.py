import json
import os

import cv2
import numpy as np

import madpose
from madpose.utils import compute_pose_error, get_depths

sample_path = "examples/image_pairs/1_eth3d"

# Thresholds for reprojection and epipolar errors
reproj_pix_thres = 8.0
epipolar_pix_thres = 2.0

# Weight for epipolar error
epipolar_weight = 1.0

# Ransac Options and Estimator Configs
options = madpose.HybridLORansacOptions()
options.min_num_iterations = 100
options.max_num_iterations = 1000
options.final_least_squares = True
options.threshold_multiplier = 5.0
options.num_lo_steps = 4
options.squared_inlier_thresholds = [reproj_pix_thres**2, epipolar_pix_thres**2]
options.data_type_weights = [1.0, epipolar_weight]
options.random_seed = 0

est_config = madpose.EstimatorConfig()
est_config.min_depth_constraint = True
est_config.use_shift = True
est_config.ceres_num_threads = 8

# Read the image pair
image0 = cv2.imread(os.path.join(sample_path, "image0.png"))
image1 = cv2.imread(os.path.join(sample_path, "image1.png"))

# Read info
with open(os.path.join(sample_path, "info.json")) as f:
    info = json.load(f)

# Load camera intrinsics
K0 = np.array(info["K0"])
K1 = np.array(info["K1"])

# Load pre-computed keypoints (you can also run keypoint detectors of your choice)
matches_0_file = os.path.join(sample_path, info["matches_0_file"])
matches_1_file = os.path.join(sample_path, info["matches_1_file"])
mkpts0 = np.load(matches_0_file)
mkpts1 = np.load(matches_1_file)

# Load pre-computed depth maps (you can also run Monocular Depth models of your choice)
depth_0_file = os.path.join(sample_path, info["depth_0_file"])
depth_1_file = os.path.join(sample_path, info["depth_1_file"])
depth_map0 = np.load(depth_0_file)
depth_map1 = np.load(depth_1_file)

# Query the depth priors of the keypoints
depth0 = get_depths(image0, depth_map0, mkpts0)
depth1 = get_depths(image1, depth_map1, mkpts1)

# Compute the principal points
pp0 = (np.array(image0.shape[:2][::-1]) - 1) / 2
pp1 = (np.array(image1.shape[:2][::-1]) - 1) / 2
pp = pp0

# Run hybrid estimation
pose, stats = madpose.HybridEstimatePoseScaleOffsetSharedFocal(
    mkpts0,
    mkpts1,
    depth0,
    depth1,
    [depth_map0.min(), depth_map1.min()],
    pp0,
    pp1,
    options,
    est_config,
)
# rotation and translation of the estimated pose
R_est, t_est = pose.R(), pose.t()
# scale and offsets of the affine corrected depth maps
s_est, o0_est, o1_est = pose.scale, pose.offset0, pose.offset1
# estimated shared focal length
f_est = pose.focal

# Load the GT Pose
T_0to1 = np.array(info["T_0to1"])

# Compute the pose error
err_t, err_R = compute_pose_error(T_0to1, R_est, t_est)

# Compute the focal error
f_mean = np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])
err_f = np.abs(f_est - f_mean) / f_mean

print("--- Hybrid Estimation Results ---")
print(f"Rotation Error: {err_R:.4f} degrees")
print(f"Translation Error: {err_t:.4f} degrees")
print(f"Focal Error: {(err_f * 100):.2f}%")
print(f"Estimated scale, offset0, offset1: {s_est:.4f}, {o0_est:.4f}, {o1_est:.4f}")

# Run point-based estimation using PoseLib
import poselib  # noqa: E402

ransac_opt_dict = {
    "max_epipolar_error": epipolar_pix_thres,
    "min_iterations": 1000,
    "max_iterations": 10000,
}
img_pair, stats = poselib.estimate_shared_focal_relative_pose(
    mkpts0, mkpts1, pp, ransac_opt_dict, {}
)
shared_focal_pose = img_pair.pose
R_ponly, t_ponly = shared_focal_pose.R, shared_focal_pose.t
f_ponly = img_pair.camera1.focal()

# Compute the pose error
err_t, err_R = compute_pose_error(T_0to1, R_ponly, t_ponly)

# Compute the focal error
err_f = np.abs(f_ponly - f_mean) / f_mean

print("--- Point-based Estimation Results ---")
print(f"Rotation Error: {err_R:.4f} degrees")
print(f"Translation Error: {err_t:.4f} degrees")
print(f"Focal Error: {(err_f * 100):.2f}%")
