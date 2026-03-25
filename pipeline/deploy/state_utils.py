"""State derivation utilities for deployment.

Vendored from ProtoMotions (Apache 2.0) deployment/state_utils.py.
Pure NumPy — no PyTorch required.

Quaternion convention: xyzw (ProtoMotions common format).
MuJoCo provides wxyz — convert at the read boundary with mujoco_wxyz_to_xyzw().
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mujoco_wxyz_to_xyzw",
    "compute_anchor_rot_np",
    "compute_root_local_ang_vel_np",
    "compute_yaw_offset_np",
    "apply_heading_offset_np",
]


# ---------------------------------------------------------------------------
# Quaternion conversion
# ---------------------------------------------------------------------------


def mujoco_wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (wxyz) to ProtoMotions xyzw convention."""
    return wxyz[..., [1, 2, 3, 0]]


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def compute_anchor_rot_np(
    rigid_body_rot: np.ndarray,
    anchor_body_index: int,
) -> np.ndarray:
    """Extract anchor body orientation from full body-rotation array.

    Args:
        rigid_body_rot: Shape [num_bodies, 4] (xyzw).
        anchor_body_index: 0-based index (e.g. 16 for torso_link on G1).

    Returns:
        Shape [4,] (xyzw).
    """
    return rigid_body_rot[anchor_body_index]


def compute_root_local_ang_vel_np(
    rigid_body_rot: np.ndarray,
    rigid_body_ang_vel: np.ndarray,
    root_body_index: int = 0,
) -> np.ndarray:
    """Convert root angular velocity from WORLD frame to LOCAL frame.

    USE THIS ONLY when source angular velocity is in WORLD frame
    (e.g. MuJoCo data.cvel). DO NOT use when source is already local
    (e.g. MuJoCo data.qvel[3:6] or real robot IMU gyro).

    Args:
        rigid_body_rot: Shape [num_bodies, 4] (xyzw).
        rigid_body_ang_vel: Shape [num_bodies, 3] (world frame).
        root_body_index: 0-based root body index (default 0 = pelvis).

    Returns:
        Shape [3,] angular velocity in local frame.
    """
    root_rot = rigid_body_rot[root_body_index]
    root_ang_vel = rigid_body_ang_vel[root_body_index]
    return _quat_rotate_inverse_np(root_rot, root_ang_vel)


def _quat_rotate_inverse_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by inverse of quaternion q (xyzw convention)."""
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


# ---------------------------------------------------------------------------
# Heading alignment
# ---------------------------------------------------------------------------


def _extract_yaw_quat_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from full orientation (xyzw)."""
    x, y, z, w = q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _quat_mul_np(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions."""
    ax, ay, az, aw = a_xyzw[..., 0], a_xyzw[..., 1], a_xyzw[..., 2], a_xyzw[..., 3]
    bx, by, bz, bw = b_xyzw[..., 0], b_xyzw[..., 1], b_xyzw[..., 2], b_xyzw[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    ).astype(np.float32)


def _quat_conjugate_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quats) of xyzw quaternion."""
    result = q_xyzw.copy()
    result[..., :3] *= -1.0
    return result


def compute_yaw_offset_np(
    robot_quat_xyzw: np.ndarray,
    motion_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Compute yaw-only heading offset between robot and motion frames.

    Returns R_offset such that R_offset * motion_body_rot is aligned
    with the robot's heading.
    """
    robot_yaw = _extract_yaw_quat_np(robot_quat_xyzw)
    motion_yaw = _extract_yaw_quat_np(motion_quat_xyzw)
    return _quat_mul_np(robot_yaw, _quat_conjugate_np(motion_yaw))


def apply_heading_offset_np(
    offset_quat_xyzw: np.ndarray,
    body_rots_xyzw: np.ndarray,
) -> np.ndarray:
    """Apply heading offset to array of body rotations.

    Args:
        offset_quat_xyzw: Shape [4,] (xyzw).
        body_rots_xyzw: Shape [..., 4] (xyzw).

    Returns:
        Aligned rotations, same shape as input.
    """
    original_shape = body_rots_xyzw.shape
    flat = body_rots_xyzw.reshape(-1, 4)
    offset_broadcast = np.broadcast_to(offset_quat_xyzw, flat.shape)
    aligned = _quat_mul_np(offset_broadcast, flat)
    return aligned.reshape(original_shape)
