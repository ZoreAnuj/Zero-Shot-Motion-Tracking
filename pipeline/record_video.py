"""Record video of zero-shot G1 tracking in MuJoCo (headless offscreen).

Generates or loads a motion, runs the ONNX tracker in MuJoCo with offscreen
rendering, and saves the result as an MP4 via imageio.

Usage:
    # From text prompt
    python -m pipeline.record_video --prompt "a person walking forward" --duration 4 --output samples/walking.mp4

    # From existing CSV
    python -m pipeline.record_video --csv output.csv --output samples/existing.mp4

    # From NPZ
    python -m pipeline.record_video --npz output.npz --output samples/from_npz.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)

# Default paths relative to this file
_PIPELINE_DIR = Path(__file__).resolve().parent
_DEFAULT_ONNX = _PIPELINE_DIR / "assets" / "proto_tracker" / "unified_pipeline.onnx"


# ---------------------------------------------------------------------------
# Text overlay (optional, requires PIL)
# ---------------------------------------------------------------------------

_HAS_PIL = False
try:
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except ImportError:
    pass


def _add_text_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a text label into the top-left corner of a frame.

    Falls back to a no-op if PIL is not available.
    """
    if not _HAS_PIL or not text:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font; fall back to default bitmap font
    font_size = max(16, frame.shape[0] // 40)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Semi-transparent background box
    margin = 8
    bbox = draw.textbbox((margin, margin), text, font=font)
    pad = 4
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(0, 0, 0, 180),
    )
    draw.text((margin, margin), text, fill=(255, 255, 255), font=font)

    return np.asarray(img)


# ---------------------------------------------------------------------------
# Core recording function
# ---------------------------------------------------------------------------


def record_simulation(
    motion_data: dict,
    output_path: str,
    onnx_path: str | None = None,
    video_fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
    camera_distance: float = 3.0,
    camera_elevation: float = -15.0,
    camera_azimuth: float = 180.0,
    overlay_text: str = "",
) -> str:
    """Run the ONNX tracker headless and record frames to MP4.

    Args:
        motion_data: Pre-converted motion dict (from proto_bridge).
        output_path: Destination .mp4 path.
        onnx_path: Path to unified_pipeline.onnx.
        video_fps: Output video frame rate.
        width: Render width in pixels.
        height: Render height in pixels.
        camera_distance: Camera distance from the tracked body.
        camera_elevation: Camera elevation angle in degrees.
        camera_azimuth: Camera azimuth angle in degrees.
        overlay_text: Optional text to burn into each frame.

    Returns:
        Absolute path to the written MP4 file.
    """
    import imageio.v3 as iio
    import onnxruntime as ort
    import yaml

    from pipeline.deploy.motion_utils import MotionPlayer
    from pipeline.deploy.mujoco_runner import (
        build_onnx_inputs,
        load_mujoco_model,
        read_robot_state,
        set_initial_pose,
    )
    from pipeline.deploy.state_utils import (
        apply_heading_offset_np,
        compute_yaw_offset_np,
    )

    # -- Resolve ONNX and YAML paths --
    onnx_file = onnx_path or str(_DEFAULT_ONNX)
    yaml_path = onnx_file.replace(".onnx", ".yaml")

    if not Path(onnx_file).exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_file}\n"
            "Run 'python -m pipeline.setup_proto_assets' to set up assets."
        )
    if not Path(yaml_path).exists():
        raise FileNotFoundError(f"YAML sidecar not found: {yaml_path}")

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

    control_hz = 1.0 / control_dt
    log.info(f"Control: dt={control_dt}s ({control_hz:.0f} Hz), decimation={decimation}")

    # -- Load ONNX session --
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    actual_out_names = [out.name for out in session.get_outputs()]

    # Warm-up run
    dummy = {
        inp.name: np.zeros(
            [1 if (d is None or isinstance(d, str)) else d for d in inp.shape],
            dtype=np.float32,
        )
        for inp in session.get_inputs()
    }
    try:
        session.run(actual_out_names, dummy)
    except Exception:
        pass

    # -- Load MuJoCo model --
    onnx_dir = Path(onnx_file).parent
    model, data = load_mujoco_model(mjcf_path, stiffness, damping_vals, physics_dt, onnx_dir)

    # -- Set up MotionPlayer --
    player = MotionPlayer(motion_data)
    log.info(
        f"Motion: {player.total_frames} frames @ "
        f"{1.0 / player.control_dt:.0f} Hz "
        f"({player.total_frames * player.control_dt:.1f}s)"
    )

    # -- Set up offscreen renderer and tracking camera --
    # Ensure offscreen framebuffer is large enough
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1  # pelvis
    cam.distance = camera_distance
    cam.elevation = camera_elevation
    cam.azimuth = camera_azimuth

    # -- Determine frame skipping --
    # Control loop runs at control_hz (e.g. 50 Hz), video at video_fps (e.g. 30 fps).
    # We render every control step but only keep frames at the video rate.
    frames_per_video_frame = max(1, int(round(control_hz / video_fps)))
    effective_video_fps = control_hz / frames_per_video_frame
    log.info(
        f"Recording: {width}x{height} @ {effective_video_fps:.1f} fps "
        f"(every {frames_per_video_frame} control steps)"
    )

    # -- Prepare output directory --
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # EMA filter state
    use_ema = action_ema_alpha < 1.0

    # -- Simulation + recording loop --
    set_initial_pose(model, data, player)

    prev_pd = None
    prev_prev_pd = None
    ema_prev_targets = None
    prev_actions = None
    heading_offset = None

    frames = []
    sim_start = time.perf_counter()

    for frame_idx in range(player.total_frames):
        # Read robot state
        robot_state = read_robot_state(data, anchor_body_index, root_body_index)

        # Heading offset on first step
        if heading_offset is None:
            robot_anchor = robot_state["body_rot"][anchor_body_index]
            motion_anchor = player.get_state_at_frame(0)["body_rot"][anchor_body_index]
            heading_offset = compute_yaw_offset_np(robot_anchor, motion_anchor)

        # Future motion references
        future_refs = player.get_future_references(frame_idx, future_step_indices)
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
        pd_targets = ort_out[1].squeeze().copy()

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

        # Capture frame at video rate
        if frame_idx % frames_per_video_frame == 0:
            renderer.update_scene(data, camera=cam)
            pixel_data = renderer.render()
            frame = np.array(pixel_data, copy=True)  # copy from renderer buffer

            if overlay_text:
                frame = _add_text_overlay(frame, overlay_text)

            frames.append(frame)

        # Progress logging
        if frame_idx % 100 == 0:
            root_h = float(data.qpos[2])
            wall_elapsed = time.perf_counter() - sim_start
            sim_elapsed = (frame_idx + 1) * control_dt
            speed = sim_elapsed / max(wall_elapsed, 1e-6)
            log.info(
                f"  step={frame_idx:4d}/{player.total_frames}  "
                f"root_h={root_h:.3f}  speed={speed:.1f}x  "
                f"frames_captured={len(frames)}"
            )

    sim_elapsed = time.perf_counter() - sim_start
    log.info(
        f"Simulation done: {player.total_frames} steps in {sim_elapsed:.1f}s "
        f"({player.total_frames * control_dt / sim_elapsed:.1f}x realtime)"
    )

    # -- Write video --
    renderer.close()

    if not frames:
        raise RuntimeError("No frames captured — motion may be empty.")

    log.info(f"Writing {len(frames)} frames to {out_path} at {effective_video_fps:.1f} fps ...")

    # Use mediapy (reliable on Windows) with fallback to imageio
    try:
        import mediapy as media
        media.write_video(str(out_path), np.stack(frames), fps=effective_video_fps)
    except Exception:
        iio.imwrite(
            str(out_path),
            np.stack(frames),
            fps=effective_video_fps,
            codec="h264",
            plugin="pyav",
        )

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    duration_s = len(frames) / effective_video_fps
    log.info(f"Saved: {out_path} ({file_size_mb:.1f} MB, {duration_s:.1f}s)")

    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Record zero-shot G1 tracking to MP4 (headless offscreen)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input (mutually exclusive)
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt", type=str, help="Text prompt for Kimodo motion generation"
    )
    input_group.add_argument(
        "--csv", type=str, help="Path to MuJoCo qpos CSV file"
    )
    input_group.add_argument(
        "--npz", type=str, help="Path to NPZ file (Kimodo output or pipeline format)"
    )

    # Generation options
    p.add_argument(
        "--duration", type=float, default=5.0,
        help="Motion duration in seconds (default: 5.0)"
    )

    # Output
    p.add_argument(
        "--output", "-o", type=str, default="recording.mp4",
        help="Output MP4 path (default: recording.mp4)"
    )

    # Model paths
    p.add_argument(
        "--onnx", type=str, default=None,
        help="Path to ONNX tracker model"
    )
    p.add_argument(
        "--proto-mjcf", type=str, default=None,
        help="Path to ProtoMotions G1 MJCF"
    )

    # Video options
    p.add_argument(
        "--width", type=int, default=1280, help="Render width (default: 1280)"
    )
    p.add_argument(
        "--height", type=int, default=720, help="Render height (default: 720)"
    )
    p.add_argument(
        "--video-fps", type=float, default=30.0,
        help="Output video FPS (default: 30)"
    )

    # Camera options
    p.add_argument(
        "--cam-distance", type=float, default=3.0,
        help="Camera distance (default: 3.0)"
    )
    p.add_argument(
        "--cam-elevation", type=float, default=-15.0,
        help="Camera elevation in degrees (default: -15)"
    )
    p.add_argument(
        "--cam-azimuth", type=float, default=180.0,
        help="Camera azimuth in degrees (default: 180)"
    )

    # Overlay
    p.add_argument(
        "--no-overlay", action="store_true",
        help="Disable text overlay on video"
    )

    args = p.parse_args()

    total_start = time.time()

    # ---- Step 1: Acquire motion ----
    if args.prompt:
        log.info("=== Step 1: Generating motion with Kimodo ===")
        from pipeline.run_g1_zeroshot import generate_motion
        qpos, fps = generate_motion(args.prompt, args.duration)
    elif args.csv:
        log.info("=== Step 1: Loading motion from CSV ===")
        from pipeline.run_g1_zeroshot import load_motion
        qpos, fps = load_motion(args.csv)
    else:
        log.info("=== Step 1: Loading motion from NPZ ===")
        from pipeline.run_g1_zeroshot import load_motion
        qpos, fps = load_motion(args.npz)

    # ---- Step 2: Convert to MotionPlayer format ----
    log.info("\n=== Step 2: Converting to ProtoMotions format ===")
    from pipeline.run_g1_zeroshot import convert_motion
    motion_data = convert_motion(qpos, fps, args.proto_mjcf)

    elapsed = time.time() - total_start
    log.info(f"Preparation took {elapsed:.1f}s")

    # ---- Step 3: Record ----
    log.info("\n=== Step 3: Recording simulation to video ===")

    # Build overlay text
    overlay_text = ""
    if not args.no_overlay:
        if args.prompt:
            overlay_text = args.prompt
        elif args.csv:
            overlay_text = f"CSV: {Path(args.csv).name}"
        elif args.npz:
            overlay_text = f"NPZ: {Path(args.npz).name}"

    out_file = record_simulation(
        motion_data=motion_data,
        output_path=args.output,
        onnx_path=args.onnx,
        video_fps=args.video_fps,
        width=args.width,
        height=args.height,
        camera_distance=args.cam_distance,
        camera_elevation=args.cam_elevation,
        camera_azimuth=args.cam_azimuth,
        overlay_text=overlay_text,
    )

    total_elapsed = time.time() - total_start
    log.info(f"\nTotal time: {total_elapsed:.1f}s")
    log.info(f"Output: {out_file}")


if __name__ == "__main__":
    main()
