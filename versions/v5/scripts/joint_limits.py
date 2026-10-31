"""Measure each joint's range in the training data: v3's joint limits (roadmap M1: "limits from
dataset percentiles", replacing the M0 placeholders).

    ..\\..\\.venv\\Scripts\\python scripts\\joint_limits.py [--frames 1000000]

Reads the training split of every processed dataset in ``configs/combined.json``, after the
quality filter (``motionlab/datasets/quality.py``; its report goes to ``outputs/quality-v4.txt``),
and writes
``motionlab/assets/joint_limits.json``:

- per joint (all but the hips, whose rotation is her facing): a **box on the rotation vector**
  (``so3_log`` of the local rotation, radians, each axis between two percentiles) and the
  **angle from rest** (degrees, one high percentile). The box keeps one-sided joints one-sided:
  a knee or a toe bends one way only, which a single angle can't say;
- the **hips height** range (metres).

Left and right are measured together (each side's samples plus the other side's, mirrored), so
the limits are symmetric, as the mirrored training windows are.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import rotation, skeleton  # noqa: E402
from motionlab.datasets import m1_data  # noqa: E402

OUT = ROOT / "motionlab" / "assets" / "joint_limits.json"


def mirror_vector(w: torch.Tensor) -> torch.Tensor:
    """The rotation vector of ``M·R·M`` (``rotation.mirror_rotation``): ``(x, −y, −z)``."""
    return w * w.new_tensor([1.0, -1.0, -1.0])


def quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.double(), q))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "combined.json")
    parser.add_argument("--frames", type=int, default=1_000_000, help="frames sampled")
    parser.add_argument("--low", type=float, default=0.0005, help="box's lower percentile")
    parser.add_argument("--high", type=float, default=0.9995, help="box's upper percentile")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text("utf-8"))
    sources = config["data"] if isinstance(config["data"], list) else [config["data"]]
    # v4: with the config's support frame, the hips range is above what she stands on.
    clips, cleaning = m1_data.load_clean_clips([ROOT / folder for folder in sources], "train",
                                               support_frame=config.get("support_frame", False))
    lost = 1 - cleaning.frames_out / cleaning.frames_in
    summary = (f"Quality filter (all splits): {cleaning.clips_in:,} clips, "
               f"{cleaning.frames_in:,} frames in; {cleaning.clips_out:,} pieces, "
               f"{cleaning.frames_out:,} frames out ({lost:.2%} removed). "
               f"Broken frames {cleaning.broken_frames:,}; short pieces dropped "
               f"{cleaning.short_pieces_dropped}; duplicate takes {len(cleaning.duplicates)}.\n"
               f"Broken joint-frames by joint: {cleaning.broken_by_joint}\n"
               f"Duplicates dropped: {cleaning.duplicates}\n")
    print(summary)
    (ROOT / "outputs" / "quality-v4.txt").write_text(summary, encoding="utf-8")
    rot6d = torch.cat([c.rot6d for c in clips]).float()
    hips = torch.cat([c.root_position[:, 1] for c in clips]).float()
    generator = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(rot6d.shape[0], generator=generator)[:args.frames]
    r = rotation.rot6d_to_matrix(rot6d[pick])  # (N, J, 3, 3)
    w = torch.cat([rotation.so3_log(r[:, j:j + 1]) for j in range(skeleton.J)], 1)  # (N, J, 3)
    angle = torch.rad2deg(rotation.rotation_angle(r))  # (N, J)

    joints = {}
    for j, name in enumerate(skeleton.NAMES):
        if j == 0:
            continue
        partner = skeleton.MIRROR[j]
        both = torch.cat((w[:, j], mirror_vector(w[:, partner])))
        angles = torch.cat((angle[:, j], angle[:, partner]))
        joints[name] = {
            "box_min_rad": [round(quantile(both[:, a], args.low), 4) for a in range(3)],
            "box_max_rad": [round(quantile(both[:, a], args.high), 4) for a in range(3)],
            "angle_max_deg": round(quantile(angles, args.high), 2),
            "placeholder_angle_deg": skeleton.placeholder_max_angle(name),
        }
    report = {
        "what": "joint limits from the training data (scripts/joint_limits.py); the hips height "
                + ("above the support" if config.get("support_frame") else "above the floor"),
        "datasets": sources,
        "frames_sampled": int(pick.numel()),
        "frames_total": int(rot6d.shape[0]),
        "percentiles": [args.low, args.high],
        "hips_height_m": [round(quantile(hips, args.low), 4),
                          round(quantile(hips, args.high), 4)],
        "joints": joints,
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {pick.numel():,} of {rot6d.shape[0]:,} training frames; "
          f"hips {report['hips_height_m']} m")
    print(f"{'joint':15} {'angle':>6} {'placeholder':>11}   box (degrees, x / y / z)")
    for name, limits in joints.items():
        box = "  ".join(f"{math.degrees(lo):6.1f}..{math.degrees(hi):5.1f}"
                        for lo, hi in zip(limits["box_min_rad"], limits["box_max_rad"],
                                          strict=True))
        print(f"{name:15} {limits['angle_max_deg']:6.1f} {limits['placeholder_angle_deg']:11.0f}"
              f"   {box}")


if __name__ == "__main__":
    main()
