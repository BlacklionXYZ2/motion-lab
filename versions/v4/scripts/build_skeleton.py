"""Build ``src/motionlab/assets/canonical_skeleton.json`` from her VRM (roadmap M0).

    .venv\\Scripts\\python scripts\\build_skeleton.py [--vrm PATH]

Default VRM: SORA's default avatar, Alice (``SORA/.local/avatars/Alice.vrm``, owner's choice
2026-10-11; kept outside git because its licence forbids redistribution). Seed-san (the built-in
fallback) is ``SEED_SAN``. The file is only read.
It checks the conventions the skeleton relies on before writing anything:
the VRM's humanoid hierarchy matches ``skeleton.JOINTS``; she faces +Z (toes in front of the
feet); her left is +X; the body is left/right symmetric within tolerance; feet are on the ground.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from motionlab import skeleton
from motionlab.vrm import read_skeleton

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
DEFAULT_VRM = LAB.parent / "SORA/.local/avatars/Alice.vrm"
SEED_SAN = LAB.parent / "SORA/clients/unity/Assets/StreamingAssets/Seed-san.vrm"
SYMMETRY_TOLERANCE_M = 0.005
# Where upperChest sits on the chest→neck segment when her VRM has none: measured on
# sister1.0.vrm (N.E.K.O's VRM 1.0 avatar, which has one): 0.436 along the segment, 7 mm off it.
UPPER_CHEST_FRACTION = 0.436
SYNTHESISED = {"upperChest": "placed 0.436 of the way from chest to neck, as measured on "
                             "sister1.0.vrm (N.E.K.O), because this VRM has no upperChest"}


def present_ancestor(name: str, present: set[str]) -> str | None:
    """The nearest canonical ancestor of ``name`` that the VRM has."""
    parent = dict(skeleton.JOINTS)[name]
    while parent is not None and parent not in present:
        parent = dict(skeleton.JOINTS)[parent]
    return parent


def build(vrm_path: Path) -> dict[str, object]:
    vrm = read_skeleton(vrm_path)
    present = {n for n in skeleton.NAMES if n in vrm.bone_nodes}
    missing = [n for n in skeleton.NAMES if n not in present]
    if set(missing) - set(SYNTHESISED):
        raise SystemExit(f"VRM lacks canonical bones that can't be placed: {missing}")
    for name in sorted(present):
        if vrm.parents[name] != present_ancestor(name, present):
            raise SystemExit(f"hierarchy mismatch: {name}'s humanoid parent in the VRM is "
                             f"{vrm.parents[name]}, expected {present_ancestor(name, present)}")
    pos = {n: vrm.world_positions[n] for n in present}
    if "upperChest" in missing:
        pos["upperChest"] = pos["chest"] + UPPER_CHEST_FRACTION * (pos["neck"] - pos["chest"])

    # Ground = the lowest foot/toe joint; positions become height above it.
    ground = min(pos[n][1] for n in skeleton.CONTACT_JOINTS)
    for n in pos:
        pos[n] = pos[n] - np.array([0.0, ground, 0.0])
    checks: dict[str, object] = {}
    if not pos["leftToes"][2] > pos["leftFoot"][2]:
        raise SystemExit("she does not face +Z (toes are not in front of the feet)")
    if not pos["leftUpperArm"][0] > 0 > pos["rightUpperArm"][0]:
        raise SystemExit("her left is not +X")
    asymmetry = max(
        float(np.abs(pos[n] * np.array([-1.0, 1.0, 1.0]) - pos[skeleton.NAMES[m]]).max())
        for n, m in zip(skeleton.NAMES, skeleton.MIRROR, strict=True))
    checks["max_left_right_asymmetry_m"] = round(asymmetry, 6)
    if asymmetry > SYMMETRY_TOLERANCE_M:
        raise SystemExit(f"left/right asymmetry {asymmetry:.4f} m exceeds tolerance")
    checks["hips_height_m"] = round(float(pos["hips"][1]), 4)
    checks["head_height_m"] = round(float(pos["head"][1]), 4)

    offsets = []
    for name, parent in skeleton.JOINTS:
        offsets.append((pos[name] if parent is None else pos[name] - pos[parent]).tolist())
    lengths = {name: round(float(np.linalg.norm(o)), 4)
               for (name, parent), o in zip(skeleton.JOINTS, offsets, strict=True) if parent}
    data = {
        "names": list(skeleton.NAMES),
        "parents": list(skeleton.PARENTS),
        "offsets": [[round(v, 6) for v in o] for o in offsets],
        "distance_from_parent_m": lengths,
        "max_angle_deg": [skeleton.placeholder_max_angle(n) for n in skeleton.NAMES],
        "omega_max": [skeleton.placeholder_omega_max(n) for n in skeleton.NAMES],
        "limits_status": "placeholder until M1 dataset percentiles",
        "checks": checks,
        "avatar_joints": [n for n in skeleton.NAMES if n in present],
        "synthesised": {n: SYNTHESISED[n] for n in missing},
        "source": {
            "vrm": vrm_path.name,
            "vrm_sha256": hashlib.sha256(vrm_path.read_bytes()).hexdigest(),
            "vrm_version": vrm.vrm_version,
            "turned_to_face_plus_z": vrm.turned,
            "vrm_meta_name": str(vrm.meta.get("name") or vrm.meta.get("title")),
            "vrm_license": str(vrm.meta.get("licenseUrl") or vrm.meta.get("otherLicenseUrl")),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--vrm", type=Path, default=DEFAULT_VRM)
    args = parser.parse_args()
    data = build(args.vrm)
    skeleton.ASSET.parent.mkdir(parents=True, exist_ok=True)
    skeleton.ASSET.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    summary = {k: data[k] for k in ("distance_from_parent_m", "checks", "source")}
    print(json.dumps(summary, indent=2))
    print("wrote", skeleton.ASSET)


if __name__ == "__main__":
    main()
