"""Set up ProtoMotions assets for the zero-shot G1 pipeline.

This script copies the required files from a local ProtoMotions clone:
1. G1 MJCF model + meshes -> pipeline/assets/proto_g1/
2. Pretrained ONNX tracker  -> pipeline/assets/proto_tracker/

Usage:
    # Auto-detect ProtoMotions clone (looks in ../ProtoMotions, ./ProtoMotions)
    python -m pipeline.setup_proto_assets

    # Specify ProtoMotions path explicitly
    python -m pipeline.setup_proto_assets --proto-path /path/to/ProtoMotions

    # After cloning ProtoMotions and exporting ONNX:
    #   git clone https://github.com/NVlabs/ProtoMotions.git
    #   cd ProtoMotions && pip install -e . && cd ..
    #   python -m pipeline.setup_proto_assets
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _PIPELINE_DIR / "assets"
_PROTO_G1_DIR = _ASSETS_DIR / "proto_g1"
_PROTO_TRACKER_DIR = _ASSETS_DIR / "proto_tracker"


def find_protomotions(hint: str | None = None) -> Path | None:
    """Try to find a local ProtoMotions clone."""
    candidates = []
    if hint:
        candidates.append(Path(hint))

    repo_root = _PIPELINE_DIR.parent
    candidates.extend([
        repo_root / "ProtoMotions",
        repo_root.parent / "ProtoMotions",
        Path.home() / "ProtoMotions",
    ])

    for p in candidates:
        if (p / "protomotions").is_dir() or (p / "deployment").is_dir():
            return p.resolve()

    return None


def copy_g1_mjcf(proto_path: Path) -> bool:
    """Copy G1 MJCF model and mesh assets."""
    # Find the MJCF file
    mjcf_candidates = [
        proto_path / "protomotions" / "data" / "assets" / "mjcf" / "g1_bm_box_feet.xml",
        proto_path / "data" / "assets" / "mjcf" / "g1_bm_box_feet.xml",
    ]

    mjcf_src = None
    for c in mjcf_candidates:
        if c.exists():
            mjcf_src = c
            break

    if mjcf_src is None:
        # Try any g1*.xml
        for pattern_dir in [
            proto_path / "protomotions" / "data" / "assets" / "mjcf",
            proto_path / "data" / "assets" / "mjcf",
        ]:
            if pattern_dir.is_dir():
                g1_files = list(pattern_dir.glob("g1*.xml"))
                if g1_files:
                    mjcf_src = g1_files[0]
                    break

    if mjcf_src is None:
        print("  WARNING: Could not find G1 MJCF file")
        return False

    mjcf_dir = mjcf_src.parent
    _PROTO_G1_DIR.mkdir(parents=True, exist_ok=True)

    # Copy MJCF file
    shutil.copy2(mjcf_src, _PROTO_G1_DIR / mjcf_src.name)
    print(f"  Copied {mjcf_src.name}")

    # Copy mesh directory if referenced in XML
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(mjcf_src))
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
        if meshdir:
            mesh_src = mjcf_dir / meshdir
            if mesh_src.is_dir():
                mesh_dst = _PROTO_G1_DIR / meshdir
                if mesh_dst.exists():
                    shutil.rmtree(mesh_dst)
                shutil.copytree(mesh_src, mesh_dst)
                print(f"  Copied mesh directory: {meshdir}/ ({len(list(mesh_dst.rglob('*')))} files)")
            else:
                print(f"  WARNING: Mesh directory not found: {mesh_src}")

    # Also copy any other mesh dirs that might be referenced
    for mesh_elem in root.iter("mesh"):
        mesh_file = mesh_elem.get("file", "")
        if mesh_file:
            mesh_path = mjcf_dir / mesh_file
            if mesh_path.exists():
                dest = _PROTO_G1_DIR / mesh_file
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(mesh_path, dest)

    return True


def copy_onnx_tracker(proto_path: Path) -> bool:
    """Copy pre-exported ONNX tracker if available."""
    onnx_candidates = [
        proto_path / "data" / "pretrained_models" / "motion_tracker" / "g1-bones-deploy",
        proto_path / "deployment" / "exports",
    ]

    _PROTO_TRACKER_DIR.mkdir(parents=True, exist_ok=True)

    for d in onnx_candidates:
        if not d.is_dir():
            continue
        onnx_files = list(d.glob("unified_pipeline.onnx"))
        if onnx_files:
            for f in d.glob("unified_pipeline.*"):
                shutil.copy2(f, _PROTO_TRACKER_DIR / f.name)
                print(f"  Copied {f.name}")
            return True

    # Check if checkpoint exists for manual export
    ckpt_dirs = [
        proto_path / "data" / "pretrained_models" / "motion_tracker" / "g1-bones-deploy",
    ]
    for d in ckpt_dirs:
        if d.is_dir() and list(d.glob("*.pt")):
            print(f"  Found checkpoint at {d}")
            print(f"  To export ONNX, run:")
            print(f"    cd {proto_path}")
            print(f"    python deployment/export_bm_tracker_onnx.py --checkpoint {d}")
            print(f"  Then re-run this script.")
            return False

    print("  WARNING: No ONNX model or checkpoint found")
    print("  You may need to export it manually from ProtoMotions.")
    return False


def main():
    p = argparse.ArgumentParser(description="Set up ProtoMotions assets")
    p.add_argument(
        "--proto-path", type=str, default=None,
        help="Path to local ProtoMotions clone"
    )
    args = p.parse_args()

    print("Setting up ProtoMotions assets for zero-shot G1 pipeline\n")

    proto_path = find_protomotions(args.proto_path)
    if proto_path is None:
        print("ERROR: Cannot find ProtoMotions clone.")
        print("\nPlease clone it first:")
        print("  git clone https://github.com/NVlabs/ProtoMotions.git")
        print("\nThen run:")
        print(f"  python -m pipeline.setup_proto_assets --proto-path ./ProtoMotions")
        return

    print(f"Found ProtoMotions at: {proto_path}\n")

    print("[1/2] Copying G1 MJCF + meshes...")
    mjcf_ok = copy_g1_mjcf(proto_path)

    print(f"\n[2/2] Copying ONNX tracker...")
    onnx_ok = copy_onnx_tracker(proto_path)

    print(f"\n{'=' * 50}")
    if mjcf_ok and onnx_ok:
        print("All assets ready! You can now run:")
        print("  python -m pipeline.run_g1_zeroshot --prompt 'a person walking forward'")
    elif mjcf_ok:
        print("G1 MJCF ready. ONNX model still needs to be exported.")
    else:
        print("Some assets are missing. Check the warnings above.")


if __name__ == "__main__":
    main()
