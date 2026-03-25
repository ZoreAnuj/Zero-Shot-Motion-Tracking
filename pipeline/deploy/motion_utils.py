"""Motion playback utilities for deployment.

Adapted from ProtoMotions (Apache 2.0) deployment/motion_utils.py.
Self-contained: SLERP interpolation is vendored so no protomotions import
is needed for resampling.

Two input modes:
1. Pre-resampled cache dict (from proto_bridge.qpos_to_motion_data).
2. Raw ProtoMotions .pt or .motion files (requires torch.load).

Cache format (dict with numpy arrays):
    dof_pos:      [num_frames, num_dofs]
    dof_vel:      [num_frames, num_dofs]
    body_rot:     [num_frames, num_bodies, 4]  (xyzw)
    body_pos:     [num_frames, num_bodies, 3]
    body_vel:     [num_frames, num_bodies, 3]
    body_ang_vel: [num_frames, num_bodies, 3]
    control_dt:   float
    num_frames:   int
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

__all__ = ["MotionPlayer"]

_STATE_KEYS = ("dof_pos", "dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel")


# ---------------------------------------------------------------------------
# Vendored SLERP interpolation (replaces protomotions dependency)
# ---------------------------------------------------------------------------


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) along last axis."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return q / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between quaternions.

    Args:
        q0: Shape [..., 4], start quaternions.
        q1: Shape [..., 4], end quaternions.
        t: Shape [...] or scalar, interpolation parameter in [0, 1].

    Returns:
        Interpolated quaternions, shape [..., 4].
    """
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)

    # Ensure shortest path
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    # Clamp for numerical stability
    dot = np.clip(dot, -1.0, 1.0)

    # Ensure t has right shape for broadcasting
    if np.ndim(t) < np.ndim(dot):
        t = t[..., np.newaxis]

    # Linear fallback for near-parallel quaternions
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    # Avoid division by zero
    use_linear = sin_theta.squeeze(-1) < 1e-6 if sin_theta.ndim > 0 else sin_theta < 1e-6
    if np.ndim(use_linear) == 0:
        use_linear = np.array([use_linear])

    s0 = np.sin((1.0 - t) * theta) / np.maximum(sin_theta, 1e-8)
    s1 = np.sin(t * theta) / np.maximum(sin_theta, 1e-8)

    result = s0 * q0 + s1 * q1

    # Use linear interpolation where sin(theta) ~ 0
    linear_result = (1.0 - t) * q0 + t * q1

    if use_linear.any():
        use_linear_expanded = np.broadcast_to(
            use_linear.reshape(use_linear.shape + (1,) * (result.ndim - use_linear.ndim)),
            result.shape,
        )
        result = np.where(use_linear_expanded, linear_result, result)

    return _normalize_quat(result)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation: a + t * (b - a)."""
    if np.ndim(t) < np.ndim(a):
        t = t[..., np.newaxis]
    return a + t * (b - a)


def _calc_frame_blend(
    time: float, motion_length: float, num_frames: int, src_dt: float
):
    """Calculate interpolation frame indices and blend factor.

    Args:
        time: Query time in seconds.
        motion_length: Total motion duration in seconds.
        num_frames: Number of source frames.
        src_dt: Source time between frames.

    Returns:
        (frame0, frame1, blend) where blend is in [0, 1).
    """
    phase = max(0.0, min(time / max(motion_length, 1e-8), 1.0))
    frame_f = phase * (num_frames - 1)
    f0 = int(frame_f)
    f1 = min(f0 + 1, num_frames - 1)
    blend = frame_f - f0
    return f0, f1, blend


# ---------------------------------------------------------------------------
# MotionPlayer
# ---------------------------------------------------------------------------


class MotionPlayer:
    """Lightweight player for a single motion clip at a fixed control rate.

    Accepts two input modes:
    1. Cache dict (from proto_bridge or cache_to_file) — fast, no dependencies.
    2. File path to a .pt file (raw or cached) — uses torch.load.

    Parameters:
        source: Either a dict (cache) or str path to a .pt file.
        control_dt: Control period in seconds (default 0.02s = 50Hz).
    """

    def __init__(
        self,
        source,
        control_dt: float = 0.02,
        motion_index: int = 0,
    ):
        if isinstance(source, dict):
            self._load_cache(source)
        elif isinstance(source, str):
            self._load_file(source, motion_index, control_dt)
        else:
            raise TypeError(f"Expected dict or str, got {type(source)}")

    @property
    def total_frames(self) -> int:
        return self._num_frames

    @property
    def num_bodies(self) -> int:
        return self._body_rot.shape[1]

    @property
    def num_dofs(self) -> int:
        return self._dof_pos.shape[1]

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def get_state_at_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Return motion state at frame_idx (clamped to valid range)."""
        idx = int(np.clip(frame_idx, 0, self._num_frames - 1))
        return {
            "dof_pos": self._dof_pos[idx],
            "dof_vel": self._dof_vel[idx],
            "body_rot": self._body_rot[idx],
            "body_pos": self._body_pos[idx],
            "body_vel": self._body_vel[idx],
            "body_ang_vel": self._body_ang_vel[idx],
        }

    def get_future_references(
        self,
        frame_idx: int,
        step_indices: List[int],
    ) -> Dict[str, np.ndarray]:
        """Return stacked future motion states.

        Args:
            frame_idx: Current frame (0-based).
            step_indices: List of positive offsets (e.g. [1, 2, ..., 25]).

        Returns:
            Dict with arrays of shape [len(step_indices), ...].
        """
        future_states = [
            self.get_state_at_frame(frame_idx + s) for s in step_indices
        ]
        return {
            key: np.stack([s[key] for s in future_states], axis=0)
            for key in _STATE_KEYS
        }

    def cache_to_file(self, output_path: str) -> None:
        """Write pre-resampled cache to a .pt file."""
        import torch

        cache = {
            "dof_pos": self._dof_pos,
            "dof_vel": self._dof_vel,
            "body_rot": self._body_rot,
            "body_pos": self._body_pos,
            "body_vel": self._body_vel,
            "body_ang_vel": self._body_ang_vel,
            "control_dt": self._control_dt,
            "num_frames": self._num_frames,
        }
        torch.save(cache, output_path)
        print(
            f"[MotionPlayer] Cached {self._num_frames} frames "
            f"@ {1.0 / self._control_dt:.0f} Hz -> {output_path}"
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_cache(self, data: dict) -> None:
        """Load from a pre-resampled cache dict."""
        self._dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        self._dof_vel = np.asarray(data["dof_vel"], dtype=np.float32)
        self._body_rot = np.asarray(data["body_rot"], dtype=np.float32)
        self._body_pos = np.asarray(data["body_pos"], dtype=np.float32)
        self._body_vel = np.asarray(data["body_vel"], dtype=np.float32)
        self._body_ang_vel = np.asarray(data["body_ang_vel"], dtype=np.float32)
        self._control_dt = float(data["control_dt"])
        self._num_frames = int(data["num_frames"])
        print(
            f"[MotionPlayer] Loaded cache: {self._num_frames} frames "
            f"@ {1.0 / self._control_dt:.0f} Hz"
        )

    def _load_file(self, path: str, motion_index: int, control_dt: float) -> None:
        """Load from a .pt file (cache or raw) and resample if needed."""
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)

        if "control_dt" in data and "body_rot" in data:
            # Already a cache file
            self._load_cache(data)
            return

        # Raw ProtoMotions format — resample with vendored SLERP
        self._control_dt = control_dt
        self._resample_raw(data, motion_index)

    def _resample_raw(self, data: dict, motion_index: int) -> None:
        """Resample raw ProtoMotions motion data to control rate using vendored SLERP."""
        if "length_starts" in data:
            # Packaged multi-motion library
            length_starts = data["length_starts"]
            motion_num_frames = data["motion_num_frames"]
            motion_dt_all = data["motion_dt"]

            start = int(length_starts[motion_index].item())
            nf = int(motion_num_frames[motion_index].item())
            end = start + nf
            src_dt = float(motion_dt_all[motion_index].item())

            gts = np.asarray(data["gts"][start:end], dtype=np.float32)
            grs = np.asarray(data["grs"][start:end], dtype=np.float32)
            gvs = np.asarray(data["gvs"][start:end], dtype=np.float32)
            gavs = np.asarray(data["gavs"][start:end], dtype=np.float32)
            dps = np.asarray(data["dps"][start:end], dtype=np.float32)
            dvs = np.asarray(data["dvs"][start:end], dtype=np.float32)

        elif "rigid_body_pos" in data:
            # Single-motion file
            fps = float(data["fps"])
            src_dt = 1.0 / fps

            gts = np.asarray(data["rigid_body_pos"], dtype=np.float32)
            grs = np.asarray(data["rigid_body_rot"], dtype=np.float32)
            gvs = np.asarray(data["rigid_body_vel"], dtype=np.float32)
            gavs = np.asarray(data["rigid_body_ang_vel"], dtype=np.float32)
            dps = np.asarray(data["dof_pos"], dtype=np.float32)
            dvs = np.asarray(data["dof_vel"], dtype=np.float32)
            nf = gts.shape[0]
        else:
            raise ValueError(
                "Unrecognised raw motion format. Expected 'length_starts' or 'rigid_body_pos'."
            )

        motion_length = src_dt * (nf - 1)
        num_ctrl_frames = max(1, int(round(motion_length / self._control_dt)) + 1)

        # Resample each control frame
        body_pos_list = []
        body_rot_list = []
        body_vel_list = []
        body_ang_vel_list = []
        dof_pos_list = []
        dof_vel_list = []

        for i in range(num_ctrl_frames):
            t = i * self._control_dt
            f0, f1, blend = _calc_frame_blend(t, motion_length, nf, src_dt)
            bl = np.float32(blend)

            body_pos_list.append(_lerp(gts[f0], gts[f1], bl))
            body_rot_list.append(_slerp(grs[f0], grs[f1], bl))
            body_vel_list.append(_lerp(gvs[f0], gvs[f1], bl))
            body_ang_vel_list.append(_lerp(gavs[f0], gavs[f1], bl))
            dof_pos_list.append(_lerp(dps[f0], dps[f1], bl))
            dof_vel_list.append(_lerp(dvs[f0], dvs[f1], bl))

        self._body_pos = np.stack(body_pos_list).astype(np.float32)
        self._body_rot = np.stack(body_rot_list).astype(np.float32)
        self._body_vel = np.stack(body_vel_list).astype(np.float32)
        self._body_ang_vel = np.stack(body_ang_vel_list).astype(np.float32)
        self._dof_pos = np.stack(dof_pos_list).astype(np.float32)
        self._dof_vel = np.stack(dof_vel_list).astype(np.float32)
        self._num_frames = num_ctrl_frames

        print(
            f"[MotionPlayer] Resampled: {nf} frames @ {1.0 / src_dt:.1f} Hz "
            f"-> {num_ctrl_frames} frames @ {1.0 / self._control_dt:.0f} Hz"
        )
