# Zero-Shot Motion Tracking for Unitree G1

> **Text-to-Physics in one command**: Type a motion description, get a physically-simulated humanoid robot performing it — no training required.

A streamlined pipeline that takes a **text prompt**, generates a motion with [Kimodo](https://github.com/nv-tlabs/kimodo), and runs a physics-simulated **Unitree G1 humanoid robot** tracking that motion in MuJoCo — all **zero-shot, with no training required**.

The pipeline uses NVIDIA's [ProtoMotions](https://github.com/NVlabs/ProtoMotions) pretrained **Generalist Tracking Policy (GTP)**, exported as an ONNX model, to drive PD joint controllers in a full MuJoCo physics simulation with gravity, contacts, and actuator dynamics.

### Highlights

- **Zero-shot**: No per-motion training. A single pretrained policy tracks any motion.
- **Text-to-physics**: Describe a motion in natural language, watch a robot perform it with real physics.
- **Full physics simulation**: Gravity, ground contacts, 29 PD-controlled actuators at 1kHz.
- **Fast**: Motion generation (~2s on GPU) + physics sim runs at 20x real-time.
- **Self-contained**: All assets (ONNX model, G1 MJCF, meshes) bundled. Just `pip install` and run.

## Demo Videos

All motions below were generated from text prompts by Kimodo, then physically simulated with the ProtoMotions GTP in MuJoCo (full physics: gravity, contacts, PD actuators).

### Walking Forward
> Prompt: `"a person walking forward"`

https://github.com/ZoreAnuj/Zero-Shot-Motion-Tracking/raw/main/samples/walking_forward.mp4

### Waving Hands
> Prompt: `"a person waving their hands"`

https://github.com/ZoreAnuj/Zero-Shot-Motion-Tracking/raw/main/samples/waving_hands.mp4

### Squats
> Prompt: `"a person doing squats"`

https://github.com/ZoreAnuj/Zero-Shot-Motion-Tracking/raw/main/samples/squats.mp4

## How It Works

```
Text Prompt ("a person walking forward")
        |
        v
   Kimodo G1 Model
   (text-to-motion diffusion)
        |
        v
   MuJoCo qpos [T, 36]
   (root xyz + quat + 29 joint angles)
        |
        v
   Proto Bridge (MuJoCo FK)
   body_pos [T, 33, 3], body_rot [T, 33, 4], dof [T, 29]
   resampled 30 Hz -> 50 Hz via SLERP
        |
        v
   ONNX Tracker Control Loop (50 Hz)
   +--> Read robot state from MuJoCo
   |    Query 4 future reference frames [+1, +2, +4, +8]
   |    ONNX inference -> PD joint targets
   |    Apply accel clamp + EMA filter
   |    Step MuJoCo physics (20 substeps @ 1kHz)
   +--- Update viewer
        |
        v
   G1 Robot in MuJoCo Viewer / Viser
   (full physics: gravity, contacts, PD actuators)
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU (for Kimodo motion generation)
- [Kimodo](https://github.com/nv-tlabs/kimodo) installed (`pip install -e .` in the kimodo repo)

### Install Dependencies

```bash
pip install onnxruntime pyyaml mujoco viser
```

### Run

```bash
# From a text prompt (requires Kimodo model)
python -m pipeline.run_g1_zeroshot --prompt "a person walking forward" --duration 5

# From an existing MuJoCo qpos CSV
python -m pipeline.run_g1_zeroshot --csv output.csv

# From an existing Kimodo NPZ output
python -m pipeline.run_g1_zeroshot --npz output.npz
```

The MuJoCo viewer will open showing the G1 robot physically tracking the motion.

### Viewer Options

```bash
# Viser web viewer (default) - open http://localhost:8080
python -m pipeline.run_g1_zeroshot --prompt "walking" --duration 3

# MuJoCo native viewer
python -m pipeline.run_g1_zeroshot --prompt "walking" --duration 3 --native-viewer

# Headless (no visualization)
python -m pipeline.run_g1_zeroshot --prompt "walking" --duration 3 --no-render

# Control loops and speed
python -m pipeline.run_g1_zeroshot --csv output.csv --loops 5 --no-realtime
```

### Record Video

```bash
python -m pipeline.record_video --prompt "a person doing squats" --duration 5 --output my_video.mp4
```

## Architecture

### File Structure

```
pipeline/
    run_g1_zeroshot.py        # Main entry point (text/CSV/NPZ -> physics sim)
    record_video.py           # Offscreen recording to MP4
    proto_bridge.py           # Kimodo qpos -> MotionPlayer format via MuJoCo FK
    setup_proto_assets.py     # Helper to copy assets from ProtoMotions clone
    deploy/                   # Vendored ProtoMotions deployment code
        state_utils.py        #   Pure-numpy quaternion/rotation utilities
        motion_utils.py       #   MotionPlayer with self-contained SLERP
        mujoco_runner.py      #   ONNX tracker control loop + viewer
    assets/
        proto_g1/             # ProtoMotions G1 MJCF + meshes (33 bodies, 29 DOFs)
        proto_tracker/        # Pretrained ONNX tracker + YAML sidecar
```

### Key Components

| Component | Source | Purpose |
|-----------|--------|---------|
| **Kimodo** | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) | Text-to-motion diffusion model for G1 |
| **ProtoMotions GTP** | [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) | Pretrained tracking policy (ONNX) |
| **MuJoCo** | [mujoco.org](https://mujoco.org) | Physics simulation (1kHz) |
| **Viser** | [nerfstudio-project/viser](https://github.com/nerfstudio-project/viser) | 3D web viewer |

### Physics Details

- **Simulation**: MuJoCo with full physics (gravity, contacts, friction)
- **Control rate**: 50 Hz (20ms per control step)
- **Physics rate**: 1000 Hz (1ms substeps, 20x decimation)
- **Actuators**: 29 implicit PD controllers with per-joint stiffness/damping
- **Robot**: Unitree G1 (33 bodies, 29 DOFs)
- **Policy**: BeyondMimic tracker with 4-step future lookahead [+1, +2, +4, +8 frames]

### Zero-Shot Tracking

The pretrained GTP was trained on a large diverse motion dataset. At inference time, it generalizes to **any new motion** without retraining:

1. Receives the current robot state (joint positions, velocities, torso orientation)
2. Receives 4 future reference frames from the target motion
3. Outputs PD joint position targets that physically track the motion while maintaining balance

This is in contrast to per-motion training approaches (like DeepMimic PPO) which require hours of training for each new motion clip.

## Setting Up Assets

The pipeline needs ProtoMotions assets (ONNX model + G1 MJCF). These are bundled in `pipeline/assets/`. To regenerate from source:

```bash
# Clone ProtoMotions
git clone https://github.com/NVlabs/ProtoMotions.git

# Copy assets
python -m pipeline.setup_proto_assets --proto-path ./ProtoMotions
```

## Acknowledgements

- [Kimodo](https://github.com/nv-tlabs/kimodo) - Kinematic Motion Diffusion Model (NVIDIA)
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) - Physics simulation and RL framework for humanoids (NVIDIA)
- [MuJoCo](https://mujoco.org) - Multi-Joint dynamics with Contact (Google DeepMind)

## License

This pipeline code is provided for research purposes. The bundled assets (ONNX model, MJCF) are from ProtoMotions (Apache 2.0). Kimodo is licensed under Apache 2.0 with NVIDIA Open Model License.
