"""Bridge between Kimodo qpos output and ProtoMotions MotionPlayer format.

Converts Kimodo G1 qpos [T, 36] into the cache dict that MotionPlayer expects:
    dof_pos [T, 29], dof_vel [T, 29],
    body_pos [T, num_bodies, 3], body_rot [T, num_bodies, 4] (xyzw),
    body_vel [T, num_bodies, 3], body_ang_vel [T, num_bodies, 3].

The conversion uses MuJoCo FK on the **ProtoMotions** G1 MJCF to produce
body states in the exact body ordering the ONNX tracker expects.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from pipeline.deploy.state_utils import mujoco_wxyz_to_xyzw

__all__ = ["qpos_to_motion_data", "kimodo_output_to_qpos"]


def _load_patched_mjcf(mjcf_path: Path):
    """Load MJCF with floor geom added (required for collision pairs)."""
    import os
    import tempfile
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(mjcf_path))
    root = tree.getroot()

    # Strip sensors
    for s in root.findall("sensor"):
        root.remove(s)

    # Add floor to worldbody
    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            "floor" in g.get("name", "").lower()
            or g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            g = ET.SubElement(worldbody, "geom")
            g.set("name", "floor")
            g.set("type", "plane")
            g.set("size", "0 0 0.05")
            g.set("rgba", "0.7 0.7 0.7 1")

    xml_str = ET.tostring(root, encoding="unicode")

    asset_dir = str(mjcf_path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=asset_dir, delete=False
    ) as f:
        f.write(xml_str)
        tmp_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)

    data = mujoco.MjData(model)
    return model, data

# Default path to ProtoMotions G1 MJCF (relative to this file)
_DEFAULT_PROTO_MJCF = (
    Path(__file__).resolve().parent / "assets" / "proto_g1" / "g1_holo_compat.xml"
)


def qpos_to_motion_data(
    qpos: np.ndarray,
    fps: float,
    proto_mjcf_path: str | Path | None = None,
    control_dt: float = 0.02,
) -> dict:
    """Convert Kimodo G1 qpos to MotionPlayer-compatible cache dict.

    Steps:
    1. Load ProtoMotions G1 MJCF into MuJoCo.
    2. For each frame: set qpos, run mj_forward, extract body states.
    3. Compute velocities via finite differences.
    4. Resample from source fps to control_dt if needed.

    Args:
        qpos: Shape [T, 36] — root_xyz(3) + root_quat_wxyz(4) + joints(29).
        fps: Source frame rate in Hz.
        proto_mjcf_path: Path to ProtoMotions G1 MJCF. Defaults to bundled asset.
        control_dt: Target control period in seconds (default 0.02s = 50Hz).

    Returns:
        Dict with keys: dof_pos, dof_vel, body_rot, body_pos, body_vel,
        body_ang_vel, control_dt, num_frames. All numpy float32.
    """
    if proto_mjcf_path is None:
        proto_mjcf_path = _DEFAULT_PROTO_MJCF
    proto_mjcf_path = Path(proto_mjcf_path)

    if not proto_mjcf_path.exists():
        raise FileNotFoundError(
            f"ProtoMotions G1 MJCF not found: {proto_mjcf_path}\n"
            f"Run 'python -m pipeline.setup_proto_assets' to set up assets."
        )

    qpos = np.asarray(qpos, dtype=np.float64)
    assert qpos.ndim == 2 and qpos.shape[1] == 36, (
        f"Expected qpos shape [T, 36], got {qpos.shape}"
    )
    T = qpos.shape[0]

    # Load MuJoCo model from ProtoMotions MJCF (with floor patch)
    model, data = _load_patched_mjcf(proto_mjcf_path)

    num_bodies = model.nbody - 1  # Skip world body at index 0
    num_dofs = model.nq - 7  # Skip free-joint (3 pos + 4 quat)

    print(f"[proto_bridge] MJCF: {num_bodies} bodies, {num_dofs} DOFs")
    print(f"[proto_bridge] Processing {T} frames at {fps} Hz...")

    # Extract body states via FK for each frame
    body_pos_all = np.zeros((T, num_bodies, 3), dtype=np.float32)
    body_rot_all = np.zeros((T, num_bodies, 4), dtype=np.float32)  # xyzw
    dof_pos_all = np.zeros((T, num_dofs), dtype=np.float32)

    for t in range(T):
        data.qpos[:] = qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # Body positions/rotations (skip world body at index 0)
        body_pos_all[t] = data.xpos[1:].copy()
        body_rot_wxyz = data.xquat[1:].copy()
        body_rot_all[t] = mujoco_wxyz_to_xyzw(body_rot_wxyz)

        # For root body, use canonical free-joint quaternion from qpos
        root_rot_wxyz = data.qpos[3:7].copy()
        body_rot_all[t, 0] = mujoco_wxyz_to_xyzw(root_rot_wxyz)

        # DOF positions (skip free-joint prefix)
        dof_pos_all[t] = data.qpos[7:].astype(np.float32)

    # Compute velocities via finite differences
    dt_src = 1.0 / fps

    dof_vel_all = np.zeros_like(dof_pos_all)
    body_vel_all = np.zeros_like(body_pos_all)
    body_ang_vel_all = np.zeros((T, num_bodies, 3), dtype=np.float32)

    if T > 1:
        # DOF velocities
        dof_vel_all[:-1] = (dof_pos_all[1:] - dof_pos_all[:-1]) / dt_src
        dof_vel_all[-1] = dof_vel_all[-2]

        # Body linear velocities
        body_vel_all[:-1] = (body_pos_all[1:] - body_pos_all[:-1]) / dt_src
        body_vel_all[-1] = body_vel_all[-2]

        # Body angular velocities (approximate from quaternion finite diff)
        body_ang_vel_all = _quat_finite_diff_ang_vel(body_rot_all, dt_src)

    # Resample to control rate if source fps differs from target
    target_fps = 1.0 / control_dt
    if abs(fps - target_fps) < 0.5:
        # Close enough, no resampling needed
        result = {
            "dof_pos": dof_pos_all,
            "dof_vel": dof_vel_all,
            "body_rot": body_rot_all,
            "body_pos": body_pos_all,
            "body_vel": body_vel_all,
            "body_ang_vel": body_ang_vel_all,
            "control_dt": control_dt,
            "num_frames": T,
        }
    else:
        # Resample using the MotionPlayer's SLERP interpolation
        from pipeline.deploy.motion_utils import MotionPlayer

        raw_cache = {
            "dof_pos": dof_pos_all,
            "dof_vel": dof_vel_all,
            "body_rot": body_rot_all,
            "body_pos": body_pos_all,
            "body_vel": body_vel_all,
            "body_ang_vel": body_ang_vel_all,
            "control_dt": dt_src,
            "num_frames": T,
        }
        # Load as cache at source dt, then resample
        player = MotionPlayer(raw_cache, control_dt=dt_src)

        # Re-create at target control_dt via SLERP
        motion_length = dt_src * (T - 1)
        num_ctrl_frames = max(1, int(round(motion_length / control_dt)) + 1)

        from pipeline.deploy.motion_utils import _slerp, _lerp, _calc_frame_blend

        r_dof_pos = []
        r_dof_vel = []
        r_body_rot = []
        r_body_pos = []
        r_body_vel = []
        r_body_ang_vel = []

        for i in range(num_ctrl_frames):
            t = i * control_dt
            f0, f1, blend = _calc_frame_blend(t, motion_length, T, dt_src)
            bl = np.float32(blend)

            r_dof_pos.append(_lerp(dof_pos_all[f0], dof_pos_all[f1], bl))
            r_dof_vel.append(_lerp(dof_vel_all[f0], dof_vel_all[f1], bl))
            r_body_rot.append(_slerp(body_rot_all[f0], body_rot_all[f1], bl))
            r_body_pos.append(_lerp(body_pos_all[f0], body_pos_all[f1], bl))
            r_body_vel.append(_lerp(body_vel_all[f0], body_vel_all[f1], bl))
            r_body_ang_vel.append(
                _lerp(body_ang_vel_all[f0], body_ang_vel_all[f1], bl)
            )

        result = {
            "dof_pos": np.stack(r_dof_pos).astype(np.float32),
            "dof_vel": np.stack(r_dof_vel).astype(np.float32),
            "body_rot": np.stack(r_body_rot).astype(np.float32),
            "body_pos": np.stack(r_body_pos).astype(np.float32),
            "body_vel": np.stack(r_body_vel).astype(np.float32),
            "body_ang_vel": np.stack(r_body_ang_vel).astype(np.float32),
            "control_dt": control_dt,
            "num_frames": num_ctrl_frames,
        }

    print(
        f"[proto_bridge] Output: {result['num_frames']} frames "
        f"@ {1.0 / control_dt:.0f} Hz, {num_bodies} bodies, {num_dofs} DOFs"
    )
    return result


def _quat_finite_diff_ang_vel(
    body_rot: np.ndarray, dt: float
) -> np.ndarray:
    """Approximate angular velocity from quaternion finite differences.

    Uses the formula: omega = 2 * (q[t+1] * q[t]^-1 - identity) / dt
    Simplified to: omega ≈ 2 * quat_to_axis_angle(q[t+1] * q[t]^-1) / dt

    Args:
        body_rot: Shape [T, num_bodies, 4] (xyzw).
        dt: Time step between frames.

    Returns:
        Angular velocity shape [T, num_bodies, 3].
    """
    T = body_rot.shape[0]
    ang_vel = np.zeros(body_rot.shape[:-1] + (3,), dtype=np.float32)

    if T <= 1:
        return ang_vel

    for t in range(T - 1):
        q0 = body_rot[t]  # [num_bodies, 4]
        q1 = body_rot[t + 1]

        # q_diff = q1 * q0^-1 (conjugate for unit quats)
        q0_conj = q0.copy()
        q0_conj[..., :3] *= -1.0

        # Hamilton product: q1 * q0_conj
        ax = q0_conj[..., 0]
        ay = q0_conj[..., 1]
        az = q0_conj[..., 2]
        aw = q0_conj[..., 3]
        bx = q1[..., 0]
        by = q1[..., 1]
        bz = q1[..., 2]
        bw = q1[..., 3]

        diff_x = aw * bx + ax * bw + ay * bz - az * by
        diff_y = aw * by - ax * bz + ay * bw + az * bx
        diff_z = aw * bz + ax * by - ay * bx + az * bw
        diff_w = aw * bw - ax * bx - ay * by - az * bz

        # Axis-angle: theta * axis = 2 * acos(w) * (xyz / |xyz|)
        # For small angles: omega ≈ 2 * xyz / dt
        ang_vel[t, :, 0] = 2.0 * diff_x / dt
        ang_vel[t, :, 1] = 2.0 * diff_y / dt
        ang_vel[t, :, 2] = 2.0 * diff_z / dt

    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def kimodo_output_to_qpos(
    output: dict,
    skeleton=None,
    device: str = "cuda:0",
) -> np.ndarray:
    """Convert Kimodo model output dict to MuJoCo qpos [T, 36].

    Wraps MujocoQposConverter.dict_to_qpos() from kimodo/exports/mujoco.py.

    Args:
        output: Kimodo model output dict (local_rot_mats, root_positions, etc.).
        skeleton: G1Skeleton34 instance. Auto-created if None.
        device: Torch device for conversion.

    Returns:
        qpos numpy array of shape [T, 36].
    """
    from kimodo.exports.mujoco import MujocoQposConverter

    if skeleton is None:
        from kimodo.skeleton import G1Skeleton34

        skeleton = G1Skeleton34()

    converter = MujocoQposConverter(skeleton)
    qpos = converter.dict_to_qpos(output, device)  # [B, T, 36] or [T, 36]

    if hasattr(qpos, "numpy"):
        qpos = qpos.numpy()
    qpos = np.asarray(qpos, dtype=np.float64)

    # Remove batch dim if present
    if qpos.ndim == 3:
        qpos = qpos[0]

    return qpos
