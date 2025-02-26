import json
import os

import cv2
import numpy as np

import madpose
from madpose.utils import bougnoux_numpy, compute_pose_error, get_depths

sample_path = "examples/image_pairs/2_2d3ds"

# Thresholds for reprojection and epipolar errors
reproj_pix_thres = 16.0
epipolar_pix_thres = 1.0

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
pose, stats = madpose.HybridEstimatePoseScaleOffsetTwoFocal(
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
# the estimated two focal lengths
f0_est, f1_est = pose.focal0, pose.focal1

# Load the GT Pose
T_0to1 = np.array(info["T_0to1"])

# Compute the pose error
err_t, err_R = compute_pose_error(T_0to1, R_est, t_est)

# Compute the focal error
err_f = max(abs(f0_est - K0[0, 0]) / K0[0, 0], abs(f1_est - K1[0, 0]) / K1[0, 0])

print("--- Hybrid Estimation Results ---")
print(f"Rotation Error: {err_R:.4f} degrees")
print(f"Translation Error: {err_t:.4f} degrees")
print(f"Focal Error: {(err_f * 100):.2f}%")
print(f"Estimated scale, offset0, offset1: {s_est:.4f}, {o0_est:.4f}, {o1_est:.4f}")

# Run point-based estimation using PoseLib
import poselib  # noqa: E402

ransac_opt_dict = {
    "max_epipolar_error": epipolar_pix_thres,
    "progressive_sampling": True,
    "min_iterations": 1000,
    "max_iterations": 10000,
}

F, stats = poselib.estimate_fundamental(mkpts0, mkpts1, ransac_opt_dict, {})
f0, f1 = bougnoux_numpy(F, pp0, pp1)
f0_ponly, f1_ponly = np.sqrt(np.abs(f0)), np.sqrt(np.abs(f1))

K0_f = np.array([[f0_ponly, 0, pp0[0]], [0, f0_ponly, pp0[1]], [0, 0, 1]])
K1_f = np.array([[f1_ponly, 0, pp1[0]], [0, f1_ponly, pp1[1]], [0, 0, 1]])
E_f = K1_f.T @ F @ K0_f

kpts0_f = (mkpts0 - K0_f[[0, 1], [2, 2]][None]) / K0_f[[0, 1], [0, 1]][None]
kpts1_f = (mkpts1 - K1_f[[0, 1], [2, 2]][None]) / K1_f[[0, 1], [0, 1]][None]
n_f, R_f, t_f, _ = cv2.recoverPose(E_f, kpts0_f, kpts1_f, np.eye(3), 1e9)
t_f = t_f.flatten()
R_ponly, t_ponly = R_f, t_f

# Compute the pose error
err_t, err_R = compute_pose_error(T_0to1, R_ponly, t_ponly)

# Compute the focal error
err_f = max(abs(f0_ponly - K0[0, 0]) / K0[0, 0], abs(f1_ponly - K1[0, 0]) / K1[0, 0])

print("--- Point-based Estimation Results ---")
print(f"Rotation Error: {err_R:.4f} degrees")
print(f"Translation Error: {err_t:.4f} degrees")
print(f"Focal Error: {(err_f * 100):.2f}%")

# Since we used mast3r's matchings and depth maps,
# we also include here pose estimation results from mast3r on this image pair (2_2d3ds)
with open(os.path.join(sample_path, "mast3r_pose.json")) as f:
    mast3r_results = json.load(f)
RT_mast3r = np.array(mast3r_results["RT"])
R_mast3r = RT_mast3r[:3, :3]
t_mast3r = RT_mast3r[:3, 3]
f0_mast3r = mast3r_results["f0"]
f1_mast3r = mast3r_results["f1"]

# Compute the pose error
err_t, err_R = compute_pose_error(T_0to1, R_mast3r, t_mast3r)

# Compute the focal error
err_f = max(abs(f0_mast3r - K0[0, 0]) / K0[0, 0], abs(f1_mast3r - K1[0, 0]) / K1[0, 0])

print("--- MASt3R Estimation Results ---")
print(f"Rotation Error: {err_R:.4f} degrees")
print(f"Translation Error: {err_t:.4f} degrees")
print(f"Focal Error: {(err_f * 100):.2f}%")
