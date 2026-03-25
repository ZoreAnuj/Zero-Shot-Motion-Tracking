"""Zero-shot text-to-physics pipeline for G1 in MuJoCo.

Uses Kimodo for motion generation and ProtoMotions' pretrained Generalist
Tracking Policy (GTP) for physics-based simulation. No training required.

Usage:
    # From text prompt (generates motion with Kimodo, then tracks in MuJoCo)
    python -m pipeline.run_g1_zeroshot --prompt "a person walking forward" --duration 5

    # From existing Kimodo CSV (MuJoCo qpos format)
    python -m pipeline.run_g1_zeroshot --csv output.csv

    # From existing NPZ (Kimodo output or pipeline format)
    python -m pipeline.run_g1_zeroshot --npz output.npz

    # Use MuJoCo native viewer instead of Viser
    python -m pipeline.run_g1_zeroshot --csv output.csv --native-viewer

    # Run headless (no visualization)
    python -m pipeline.run_g1_zeroshot --csv output.csv --no-render
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Default paths relative to this file
_PIPELINE_DIR = Path(__file__).resolve().parent
_DEFAULT_ONNX = _PIPELINE_DIR / "assets" / "proto_tracker" / "unified_pipeline.onnx"
_DEFAULT_MJCF = _PIPELINE_DIR / "assets" / "proto_g1" / "g1_bm_box_feet.xml"


# ---------------------------------------------------------------------------
# Step 1: Acquire motion (text prompt or existing file)
# ---------------------------------------------------------------------------


def generate_motion(prompt: str, duration: float, device: str = "cuda:0"):
    """Generate G1 motion with Kimodo and return qpos [T, 36] + fps.

    Reuses the pattern from pipeline/train.py:generate_reference().
    """
    import torch
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.model.load_model import load_model

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    log.info(f"Loading Kimodo G1 model on {device}...")
    model, _ = load_model("kimodo-g1-rp", device=device, return_resolved_name=True)

    fps = model.fps
    num_frames = [int(duration * fps)]
    texts = [prompt.strip().rstrip(".") + "."]

    log.info(f"Generating: '{texts[0]}' ({num_frames[0]} frames at {fps} fps)")
    output = model(
        texts,
        num_frames,
        num_denoising_steps=100,
        num_samples=1,
        multi_prompt=True,
        num_transition_frames=5,
        post_processing=False,
        return_numpy=True,
    )

    converter = MujocoQposConverter(model.skeleton)
    qpos = converter.dict_to_qpos(output, device)
    if hasattr(qpos, "numpy"):
        qpos = qpos.numpy()
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos[0]  # Remove batch dim

    log.info(f"Generated {qpos.shape[0]} frames at {fps} fps ({qpos.shape[0] / fps:.1f}s)")
    return qpos, fps


def load_motion(path: str):
    """Load motion from CSV or NPZ file. Returns (qpos [T, 36], fps).

    Reuses the pattern from pipeline/train.py:load_reference().
    """
    path = os.path.abspath(path)

    if path.endswith(".csv"):
        qpos = np.loadtxt(path, delimiter=",")
        log.info(f"Loaded CSV: {qpos.shape[0]} frames (assuming 30 fps)")
        return qpos, 30.0

    data = np.load(path, allow_pickle=True)

    if "qpos" in data:
        fps = float(data["fps"]) if "fps" in data else 30.0
        qpos = data["qpos"]
        log.info(f"Loaded NPZ (qpos): {qpos.shape[0]} frames at {fps} fps")
        return qpos, fps

    if "local_rot_mats" in data:
        log.info("Detected Kimodo NPZ format, converting to qpos...")
        from pipeline.proto_bridge import kimodo_output_to_qpos

        output = {k: data[k] for k in data.files}
        if output["local_rot_mats"].ndim == 4:
            output["local_rot_mats"] = output["local_rot_mats"][np.newaxis]
            output["root_positions"] = output["root_positions"][np.newaxis]

        qpos = kimodo_output_to_qpos(output)
        log.info(f"Converted: {qpos.shape[0]} frames")
        return qpos, 30.0

    raise ValueError(
        f"Cannot load motion from {path}: expected 'qpos' or 'local_rot_mats' key"
    )


# ---------------------------------------------------------------------------
# Step 2: Convert to MotionPlayer format
# ---------------------------------------------------------------------------


def convert_motion(qpos: np.ndarray, fps: float, proto_mjcf_path: str | None = None):
    """Convert qpos to MotionPlayer-compatible dict via proto_bridge."""
    from pipeline.proto_bridge import qpos_to_motion_data

    mjcf_path = proto_mjcf_path or str(_DEFAULT_MJCF)
    return qpos_to_motion_data(qpos, fps, mjcf_path)


# ---------------------------------------------------------------------------
# Step 3: Run simulation
# ---------------------------------------------------------------------------


def run_tracker(
    motion_data: dict,
    onnx_path: str | None = None,
    render: bool = True,
    realtime: bool = True,
    num_loops: int = 1,
    viser_port: int = 8080,
    use_native_viewer: bool = False,
):
    """Run the ONNX tracker with the converted motion data."""
    from pipeline.deploy.motion_utils import MotionPlayer
    from pipeline.deploy.mujoco_runner import run_simulation

    onnx = onnx_path or str(_DEFAULT_ONNX)

    if not Path(onnx).exists():
        print(
            f"\nError: ONNX model not found at {onnx}\n"
            f"Run 'python -m pipeline.setup_proto_assets' to set up assets.\n"
            f"Or provide --onnx /path/to/unified_pipeline.onnx"
        )
        sys.exit(1)

    yaml_path = onnx.replace(".onnx", ".yaml")
    if not Path(yaml_path).exists():
        print(f"\nError: YAML sidecar not found at {yaml_path}")
        sys.exit(1)

    player = MotionPlayer(motion_data)

    log.info(
        f"Motion: {player.total_frames} frames @ "
        f"{1.0 / player.control_dt:.0f} Hz "
        f"({player.total_frames * player.control_dt:.1f}s)"
    )

    run_simulation(
        onnx_path=onnx,
        motion_player=player,
        render=render,
        realtime=realtime,
        num_loops=num_loops,
        viser_port=viser_port,
        use_native_viewer=use_native_viewer,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Zero-shot G1 tracking: text/motion -> physics simulation",
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
        "--duration", type=float, default=5.0, help="Motion duration in seconds (default: 5.0)"
    )

    # Model paths
    p.add_argument(
        "--onnx", type=str, default=None, help="Path to ONNX tracker model"
    )
    p.add_argument(
        "--proto-mjcf", type=str, default=None, help="Path to ProtoMotions G1 MJCF"
    )

    # Viewer options
    p.add_argument(
        "--no-render", action="store_true", help="Disable visualization"
    )
    p.add_argument(
        "--native-viewer", action="store_true", help="Use MuJoCo native viewer instead of Viser"
    )
    p.add_argument(
        "--viser-port", type=int, default=8080, help="Viser web viewer port (default: 8080)"
    )

    # Playback options
    p.add_argument(
        "--loops", type=int, default=None,
        help="Number of loops (default: infinite with render, 1 without)"
    )
    p.add_argument(
        "--no-realtime", action="store_true", help="Run as fast as possible"
    )

    # Output
    p.add_argument(
        "--save-motion", type=str, default=None,
        help="Save converted motion cache to .pt file"
    )

    args = p.parse_args()

    total_start = time.time()

    # ---- Step 1: Acquire motion ----
    if args.prompt:
        log.info(f"=== Step 1: Generating motion with Kimodo ===")
        qpos, fps = generate_motion(args.prompt, args.duration)
    elif args.csv:
        log.info(f"=== Step 1: Loading motion from CSV ===")
        qpos, fps = load_motion(args.csv)
    else:
        log.info(f"=== Step 1: Loading motion from NPZ ===")
        qpos, fps = load_motion(args.npz)

    # ---- Step 2: Convert to MotionPlayer format ----
    log.info(f"\n=== Step 2: Converting to ProtoMotions format ===")
    motion_data = convert_motion(qpos, fps, args.proto_mjcf)

    # Optionally save motion cache
    if args.save_motion:
        import torch

        torch.save(motion_data, args.save_motion)
        log.info(f"Saved motion cache to {args.save_motion}")

    elapsed = time.time() - total_start
    log.info(f"\nPreparation took {elapsed:.1f}s")

    # ---- Step 3: Run tracker ----
    num_loops = args.loops
    if num_loops is None:
        num_loops = 10_000_000 if not args.no_render else 1

    log.info(f"\n=== Step 3: Running ONNX tracker in MuJoCo ===")
    if not args.no_render and not args.native_viewer:
        log.info(f"Open http://localhost:{args.viser_port} in your browser")

    run_tracker(
        motion_data=motion_data,
        onnx_path=args.onnx,
        render=not args.no_render,
        realtime=not args.no_realtime,
        num_loops=num_loops,
        viser_port=args.viser_port,
        use_native_viewer=args.native_viewer,
    )


if __name__ == "__main__":
    main()
