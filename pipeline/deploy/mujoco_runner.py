"""ONNX tracker control loop with MuJoCo physics and Viser rendering.

Adapted from ProtoMotions (Apache 2.0) deployment/test_tracker_mujoco.py.
Replaces the MuJoCo native viewer with Viser for web-based 3D visualization.

Pipeline per control step (50 Hz):
1. Read robot state from MuJoCo (qpos, qvel, xquat, cvel).
2. Derive anchor_rot (torso) and root_local_ang_vel (pelvis).
3. Query MotionPlayer for 25 future reference frames.
4. Run ONNX inference -> PD position targets.
5. Apply acceleration clamp + EMA filter.
6. Step MuJoCo physics (decimation substeps).
7. Update Viser scene.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from pipeline.deploy.motion_utils import MotionPlayer
from pipeline.deploy.state_utils import (
    compute_anchor_rot_np,
    compute_yaw_offset_np,
    apply_heading_offset_np,
    mujoco_wxyz_to_xyzw,
)

log = logging.getLogger(__name__)

__all__ = ["run_simulation", "load_mujoco_model"]


# ---------------------------------------------------------------------------
# MuJoCo model loading
# ---------------------------------------------------------------------------


def _resolve_mjcf_path(mjcf_path: str, onnx_dir: Path | None = None) -> Path:
    """Resolve MJCF path, searching in common locations."""
    p = Path(mjcf_path)
    if p.is_absolute() and p.exists():
        return p

    candidates = [p]
    if onnx_dir:
        candidates.append(onnx_dir / mjcf_path)
        candidates.append(onnx_dir.parent / mjcf_path)

    # Check relative to pipeline assets
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    candidates.append(assets_dir / "proto_g1" / Path(mjcf_path).name)
    # Also try matching YAML's "mjcf/<filename>" pattern
    candidates.append(assets_dir / "proto_g1" / mjcf_path.replace("mjcf/", ""))

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        f"Cannot find MJCF '{mjcf_path}'. Tried: {[str(c) for c in candidates]}"
    )


def _patch_mjcf_xml(xml_path: Path) -> str:
    """Patch MJCF: strip sensors, add ground plane + light."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            "floor" in g.get("name", "").lower()
            or "ground" in g.get("name", "").lower()
            or g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")
            ground.set("rgba", "0.7 0.7 0.7 1")

        if not worldbody.findall("light"):
            light = ET.SubElement(worldbody, "light")
            light.set("pos", "2 0 5.0")
            light.set("dir", "0 0 -1")
            light.set("diffuse", "0.4 0.4 0.4")
            light.set("specular", "0.1 0.1 0.1")
            light.set("directional", "true")

    return ET.tostring(root, encoding="unicode")


def load_mujoco_model(
    mjcf_path: str,
    stiffness: list,
    damping: list,
    physics_dt: float,
    onnx_dir: Path | None = None,
):
    """Load and configure MuJoCo model to match ProtoMotions training.

    Sets physics timestep, zeros passive forces, configures implicit PD actuators.
    """
    mjcf_file = _resolve_mjcf_path(mjcf_path, onnx_dir)
    log.info(f"Loading MuJoCo model: {mjcf_file}")

    patched_xml = _patch_mjcf_xml(mjcf_file)

    asset_dir = str(mjcf_file.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=asset_dir, delete=False
    ) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)

    data = mujoco.MjData(model)

    # Physics timestep
    model.opt.timestep = physics_dt
    log.info(f"  Physics: {1.0 / physics_dt:.0f} Hz")

    # Zero passive forces (PD handled by actuators)
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0

    # Configure implicit PD actuators
    num_actuators = model.nu
    assert num_actuators == len(stiffness) == len(damping), (
        f"Actuator mismatch: nu={num_actuators}, stiffness={len(stiffness)}, damping={len(damping)}"
    )
    for i in range(num_actuators):
        kp = stiffness[i]
        kd = damping[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1  # affine
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        model.actuator_ctrllimited[i] = 0

    log.info(f"  {num_actuators} PD actuators, {model.nbody} bodies, {model.nq} qpos")
    return model, data


# ---------------------------------------------------------------------------
# Robot state reading
# ---------------------------------------------------------------------------


def read_robot_state(data, anchor_body_index: int, root_body_index: int = 0):
    """Read robot state from MuJoCo data buffers.

    Returns dict with numpy arrays (no batch dimension).
    Quaternions are converted from MuJoCo wxyz to xyzw.
    """
    body_rot_wxyz = data.xquat[1:].copy()
    body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz)

    # Root body: use canonical free-joint quat from qpos
    root_rot_wxyz = data.qpos[3:7].copy()
    body_rot[root_body_index] = mujoco_wxyz_to_xyzw(root_rot_wxyz)

    # Root angular velocity: qvel[3:6] is already in body-local frame
    root_local_ang_vel = data.qvel[3:6].copy().astype(np.float32)

    return {
        "dof_pos": data.qpos[7:].copy().astype(np.float32),
        "dof_vel": data.qvel[6:].copy().astype(np.float32),
        "body_rot": body_rot.astype(np.float32),
        "root_local_ang_vel": root_local_ang_vel,
    }


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def build_onnx_inputs(
    robot_state: dict,
    future_refs: dict,
    onnx_name_to_key: dict,
    anchor_body_index: int,
    num_dofs: int,
    prev_actions: np.ndarray | None = None,
) -> dict:
    """Build ONNX input dict from robot state + future motion references."""
    dof_pos = robot_state["dof_pos"]
    dof_vel = robot_state["dof_vel"]
    body_rot = robot_state["body_rot"]
    root_local_ang_vel = robot_state["root_local_ang_vel"]

    anchor_rot = compute_anchor_rot_np(body_rot, anchor_body_index)

    if prev_actions is None:
        prev_actions = np.zeros(num_dofs, dtype=np.float32)

    future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]

    key_to_array = {
        "current.dof_pos": dof_pos[None],
        "current.dof_vel": dof_vel[None],
        "current.anchor_rot": anchor_rot[None],
        "current.root_local_ang_vel": root_local_ang_vel[None],
        "historical.processed_actions": prev_actions[None, None],
        "mimic.future_anchor_rot": future_anchor_rot[None],
        "mimic.future_rot": future_refs["body_rot"][None],
        "mimic.future_dof_pos": future_refs["dof_pos"][None],
        "mimic.future_dof_vel": future_refs["dof_vel"][None],
    }

    onnx_inputs = {}
    for onnx_name, sem_key in onnx_name_to_key.items():
        if sem_key in key_to_array:
            onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)
        else:
            log.warning(f"No value for ONNX input '{onnx_name}' (key='{sem_key}')")

    return onnx_inputs


# ---------------------------------------------------------------------------
# Initial pose
# ---------------------------------------------------------------------------


def set_initial_pose(model, data, motion_player: MotionPlayer) -> None:
    """Set robot to frame 0 of the motion."""
    frame0 = motion_player.get_state_at_frame(0)

    root_pos = frame0["body_pos"][0]
    root_quat = frame0["body_rot"][0]  # xyzw

    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat[[3, 0, 1, 2]]  # xyzw -> wxyz
    data.qpos[7:] = frame0["dof_pos"]
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Viser visualization
# ---------------------------------------------------------------------------


def _setup_viser(model, data, port: int = 8080):
    """Set up Viser 3D viewer for G1 visualization."""
    import viser
    import trimesh

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")

    # Add ground plane
    server.scene.add_grid("ground", width=10, height=10, cell_size=0.5)

    # Load robot meshes and create scene nodes
    mesh_handles = []
    for i in range(model.nmesh):
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        if mesh_name is None:
            continue

        # Extract mesh data from MuJoCo
        vert_start = model.mesh_vertadr[i]
        vert_count = model.mesh_vertnum[i]
        face_start = model.mesh_faceadr[i]
        face_count = model.mesh_facenum[i]

        verts = model.mesh_vert[vert_start : vert_start + vert_count].copy()
        faces = model.mesh_face[face_start : face_start + face_count].copy()

        if vert_count == 0 or face_count == 0:
            continue

        # Find which geom uses this mesh
        rgba = np.array([0.7, 0.7, 0.7, 1.0])
        for gi in range(model.ngeom):
            if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[gi] == i:
                rgba = model.geom_rgba[gi].copy()
                break

        color = (rgba[:3] * 255).astype(np.uint8)

        handle = server.scene.add_mesh(
            f"robot/{mesh_name}",
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            color=tuple(color),
        )
        mesh_handles.append((i, handle))

    return server, mesh_handles


def _update_viser(server, mesh_handles, model, data):
    """Update Viser mesh transforms from MuJoCo state."""
    import viser.transforms as tf

    for mesh_id, handle in mesh_handles:
        # Find the geom that uses this mesh
        for gi in range(model.ngeom):
            if (
                model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH
                and model.geom_dataid[gi] == mesh_id
            ):
                body_id = model.geom_bodyid[gi]
                # Body transform
                pos = data.xpos[body_id].copy()
                quat_wxyz = data.xquat[body_id].copy()
                # Geom offset relative to body
                geom_pos = model.geom_pos[gi].copy()
                geom_quat = model.geom_quat[gi].copy()

                # Combine: body_transform * geom_offset
                # For simplicity, use MuJoCo's computed geom positions
                # which already include the body transform
                full_pos = pos + _quat_rotate(quat_wxyz, geom_pos)
                full_quat = _quat_mul_wxyz(quat_wxyz, geom_quat)

                handle.position = full_pos.astype(np.float32)
                handle.wxyz = full_quat.astype(np.float32)
                break


def _quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz convention)."""
    w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product (wxyz convention)."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def run_simulation(
    onnx_path: str,
    motion_player: MotionPlayer,
    render: bool = True,
    realtime: bool = True,
    num_loops: int = 1,
    viser_port: int = 8080,
    use_native_viewer: bool = False,
) -> None:
    """Run the ONNX tracker in a MuJoCo simulation loop.

    Args:
        onnx_path: Path to unified_pipeline.onnx.
        motion_player: Loaded MotionPlayer with motion data.
        render: Enable visualization.
        realtime: Pace simulation to wall-clock time.
        num_loops: Number of times to loop the motion.
        viser_port: Port for Viser web viewer.
        use_native_viewer: Use MuJoCo native viewer instead of Viser.
    """
    onnx_path = str(onnx_path)
    yaml_path = onnx_path.replace(".onnx", ".yaml")

    # Load YAML metadata
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    robot_meta = meta["robot"]
    timing = meta["timing"]
    motion_meta = meta["motion"]
    control = meta["control"]
    runtime = meta["_runtime"]

    anchor_body_index = robot_meta["anchor_body_index"]
    root_body_index = robot_meta["root_body_index"]
    num_dofs = robot_meta["num_dofs"]
    mjcf_path = robot_meta["mjcf_path"]
    control_dt = timing["control_dt"]
    decimation = timing["decimation"]
    physics_dt = timing["physics_dt"]
    future_step_indices = motion_meta["future_step_indices"]
    stiffness = control["stiffness"]
    damping_vals = control["damping"]
    pd_target_max_accel = control.get("pd_target_max_accel")
    action_ema_alpha = control.get("action_ema_alpha", 1.0)
    onnx_name_to_key = runtime["onnx_name_to_in_key"]

    log.info(f"ONNX: {onnx_path}")
    log.info(f"Robot: {num_dofs} DOFs, anchor={anchor_body_index}, root={root_body_index}")
    log.info(f"Control: dt={control_dt}s, decimation={decimation}, physics={1.0 / physics_dt:.0f}Hz")

    # Load ONNX session (CPU is fast enough for single-env inference)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual_out_names = [out.name for out in session.get_outputs()]

    # Warm-up
    for inp in session.get_inputs():
        shape = [1 if (d is None or isinstance(d, str)) else d for d in inp.shape]
        dummy = {inp.name: np.zeros(shape, dtype=np.float32) for inp in session.get_inputs()}
    try:
        session.run(actual_out_names, dummy)
    except Exception:
        pass

    # Load MuJoCo model
    onnx_dir = Path(onnx_path).parent
    model, data = load_mujoco_model(mjcf_path, stiffness, damping_vals, physics_dt, onnx_dir)

    # Set up viewer
    viewer = None
    viser_server = None
    mesh_handles = None

    if render:
        if use_native_viewer:
            try:
                from mujoco import viewer as mj_viewer

                viewer = mj_viewer.launch_passive(
                    model, data, show_left_ui=False, show_right_ui=False
                )
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -10.0
                viewer.cam.azimuth = 180.0
                viewer.cam.trackbodyid = 1
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                log.info("MuJoCo native viewer launched")
            except Exception as e:
                log.warning(f"Could not launch native viewer: {e}")
        else:
            try:
                viser_server, mesh_handles = _setup_viser(model, data, viser_port)
                log.info(f"Viser viewer: http://localhost:{viser_port}")
            except Exception as e:
                log.warning(f"Could not launch Viser: {e}")
                # Fall back to native viewer
                try:
                    from mujoco import viewer as mj_viewer

                    viewer = mj_viewer.launch_passive(
                        model, data, show_left_ui=False, show_right_ui=False
                    )
                    log.info("Fell back to MuJoCo native viewer")
                except Exception:
                    log.warning("No viewer available, running headless")

    # EMA filter state
    use_ema = action_ema_alpha < 1.0
    if pd_target_max_accel is not None:
        log.info(f"PD accel clamp: {pd_target_max_accel}")
    if use_ema:
        log.info(f"Action EMA alpha: {action_ema_alpha}")

    # Simulation loop
    total_steps = 0
    loop_idx = 0

    while loop_idx < num_loops:
        log.info(f"\n--- Loop {loop_idx + 1}/{num_loops} ---")
        set_initial_pose(model, data, motion_player)

        # Reset per-loop state
        prev_pd = None
        prev_prev_pd = None
        ema_prev_targets = None
        prev_actions = None
        heading_offset = None
        loop_start = time.perf_counter()

        for frame_idx in range(motion_player.total_frames):
            step_start = time.perf_counter()

            # Read robot state
            robot_state = read_robot_state(data, anchor_body_index, root_body_index)

            # Heading offset on first step
            if heading_offset is None:
                robot_anchor = robot_state["body_rot"][anchor_body_index]
                motion_anchor = motion_player.get_state_at_frame(0)["body_rot"][anchor_body_index]
                heading_offset = compute_yaw_offset_np(robot_anchor, motion_anchor)

            # Future motion references
            future_refs = motion_player.get_future_references(frame_idx, future_step_indices)
            future_refs["body_rot"] = apply_heading_offset_np(
                heading_offset, future_refs["body_rot"]
            )

            # ONNX inference
            onnx_inputs = build_onnx_inputs(
                robot_state=robot_state,
                future_refs=future_refs,
                onnx_name_to_key=onnx_name_to_key,
                anchor_body_index=anchor_body_index,
                num_dofs=num_dofs,
                prev_actions=prev_actions,
            )

            ort_out = session.run(actual_out_names, onnx_inputs)
            pd_targets = ort_out[1].squeeze().copy()  # joint_pos_targets

            # PD target acceleration clamp
            if pd_target_max_accel is not None and prev_pd is not None and prev_prev_pd is not None:
                delta = pd_targets - prev_pd
                prev_delta = prev_pd - prev_prev_pd
                accel = delta - prev_delta
                clamped_accel = np.clip(accel, -pd_target_max_accel, pd_target_max_accel)
                pd_targets = prev_pd + prev_delta + clamped_accel

            prev_prev_pd = prev_pd
            prev_pd = pd_targets.copy()

            # EMA filter
            if use_ema:
                if ema_prev_targets is None:
                    ema_prev_targets = pd_targets.copy()
                pd_targets = (
                    action_ema_alpha * pd_targets
                    + (1.0 - action_ema_alpha) * ema_prev_targets
                )
                ema_prev_targets = pd_targets.copy()

            prev_actions = pd_targets.copy()

            # Write PD targets and step physics
            data.ctrl[:] = pd_targets
            for _ in range(decimation):
                mujoco.mj_step(model, data)

            # Update viewer
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            elif viser_server is not None and mesh_handles is not None:
                _update_viser(viser_server, mesh_handles, model, data)

            # Real-time pacing
            if realtime:
                elapsed = time.perf_counter() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            total_steps += 1

            if frame_idx % 100 == 0:
                root_h = float(data.qpos[2])
                wall_elapsed = time.perf_counter() - loop_start
                sim_elapsed = (frame_idx + 1) * control_dt
                speed = sim_elapsed / max(wall_elapsed, 1e-6)
                log.info(
                    f"  step={total_steps:5d}  frame={frame_idx:4d}  "
                    f"root_h={root_h:.3f}  speed={speed:.2f}x"
                )

        loop_idx += 1

        if viewer is not None and not viewer.is_running():
            break

    log.info(f"\nDone: {total_steps} steps over {loop_idx} loops")

    # Keep Viser running until user closes
    if viser_server is not None:
        log.info("Viser viewer still running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass
