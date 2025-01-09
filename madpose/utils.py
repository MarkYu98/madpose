import numpy as np


def get_depths(image, depth_map, mkpts):
    # Calculate the scaling factor between the image and the depth map
    resize = np.array([depth_map.shape[1] / image.shape[1], depth_map.shape[0] / image.shape[0]])

    # Scale the keypoints to match the depth map's resolution and round to the nearest integer
    scaled_coords = np.round(mkpts * resize).astype(int)

    # Clip the coordinates to ensure they are within the bounds of the depth map
    scaled_coords[:, 0] = np.clip(scaled_coords[:, 0], 0, depth_map.shape[1] - 1)
    scaled_coords[:, 1] = np.clip(scaled_coords[:, 1], 0, depth_map.shape[0] - 1)

    # Perform nearest neighbor query
    depths = depth_map[scaled_coords[:, 1], scaled_coords[:, 0]]
    return depths


def bougnoux_numpy(F, p1, p2):
    # Convert p1 and p2 to homogeneous coordinates and reshape
    p1 = np.append(p1, 1.0).reshape(3, 1)
    p2 = np.append(p2, 1.0).reshape(3, 1)

    # Perform SVD on F
    e2, _, e1 = np.linalg.svd(F, full_matrices=True)
    e1 = e1[2, :]
    e2 = e2[:, 2]

    # Normalize e1 and e2
    e1 = e1 / e1[2]
    e2 = e2 / e2[2]

    # Skew-symmetric matrices for e1 and e2
    s_e2 = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])

    s_e1 = np.array([
        [0, -e1[2], e1[1]],
        [e1[2], 0, -e1[0]],
        [-e1[1], e1[0], 0]
    ])

    # Diagonal matrix II
    II = np.diag([1.0, 1.0, 0.0])

    # Compute f1 and f2
    numerator_f1 = -p2.T @ s_e2 @ II @ F @ (p1 @ p1.T) @ F.T @ p2
    denominator_f1 = p2.T @ s_e2 @ II @ F @ II @ F.T @ p2
    f1 = numerator_f1 / denominator_f1

    numerator_f2 = -p1.T @ s_e1 @ II @ F.T @ (p2 @ p2.T) @ F @ p1
    denominator_f2 = p1.T @ s_e1 @ II @ F.T @ II @ F @ p1
    f2 = numerator_f2 / denominator_f2

    return f1[0, 0], f2[0, 0]


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t, t_thres=None):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    if t_thres is not None and np.linalg.norm(t_gt) < t_thres:
        error_t = 0
    return error_t, error_R