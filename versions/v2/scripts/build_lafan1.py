"""Build LAFAN1 for M1: calibrate the retarget, convert every take, check it.

    .venv\\Scripts\\python scripts\\build_lafan1.py

Reads ``data/lafan1/lafan1.zip`` and writes:

- ``src/motionlab/assets/lafan1_alignment.json``: the retarget calibration (22 rotations and the
  root scale; measurements, not motion data);
- ``data/processed/lafan1/<take>.pt``: each take as a 60 Hz canonical clip with foot contacts
  (git-ignored; LAFAN1's licence forbids sharing retargeted copies);
- ``outputs/m1/lafan1_build.json``: the checks per take.

Checks: every bone's world direction after retargeting against the BVH's (exact by construction
for joints with a child bone); her hips height while standing; the planted ankles' height and how
fast planted feet slide.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import fk, rotation, skeleton  # noqa: E402
from motionlab.datasets import bvh, contacts, lafan1  # noqa: E402
from motionlab.formats import DT, Clip  # noqa: E402

ARCHIVE = LAB / "data" / "lafan1" / "lafan1.zip"
CALIBRATION = ROOT / "motionlab" / "assets" / "lafan1_alignment.json"
OUT = ROOT / "data" / "processed" / "lafan1"
REPORT = ROOT / "outputs" / "lafan1_build_v2.json"


def bone_error_degrees(source: bvh.Bvh, calibration: lafan1.Calibration) -> float:
    """Worst angle between a retargeted bone and the BVH's, over every frame."""
    local, root = lafan1.to_canonical(source, calibration)
    _, positions = fk.forward_kinematics(local, root, skeleton.load(torch.float64))
    order = lafan1.canonical_order(source)
    _, original = bvh.forward(source, source.local_rotations(), source.root_position())
    original = original[:, order]
    worst = 0.0
    for name, child in lafan1.BONE_CHILD.items():
        if child is None:
            continue
        j, c = skeleton.INDEX[name], skeleton.INDEX[child]
        a, b = positions[:, c] - positions[:, j], original[:, c] - original[:, j]
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-9)
        worst = max(worst, float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).max()))
    return worst


def clip_checks(clip: Clip) -> dict[str, float]:
    _, positions = fk.forward_kinematics(rotation.rot6d_to_matrix(clip.rot6d.double()),
                                         clip.root_position.double(),
                                         skeleton.load(torch.float64))
    assert clip.contacts is not None
    feet = positions[:, [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]]
    planted = clip.contacts
    ankles = feet[:, [0, 2], 1][planted[:, [0, 2]]]
    step = (feet[1:] - feet[:-1])[..., [0, 2]].norm(dim=-1) / DT  # horizontal speed
    both = planted[1:] & planted[:-1]
    slide = step[both]
    return {
        "frames": clip.frames,
        "seconds": round(clip.frames * DT, 1),
        "hips_y_median": round(float(clip.root_position[:, 1].median()), 3),
        "planted_ankle_y_median": round(float(ankles.median()), 3) if ankles.numel() else -1,
        "planted_share": round(float(planted.float().mean()), 3),
        "planted_slide_m_s_p95": round(float(slide.quantile(0.95)), 3) if slide.numel() else -1,
    }


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith(".bvh"))
        texts = {n: archive.read(n).decode("utf-8") for n in names}
    sources = {n: bvh.parse(t) for n, t in texts.items()}
    calibration = lafan1.calibrate(sources.values())
    lafan1.save_calibration(calibration, CALIBRATION)
    print(f"Calibrated on {calibration.takes} takes; root scale {calibration.root_scale:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"root_scale": calibration.root_scale, "rest_ankle_y": round(float(
        skeleton.load().rest_positions[skeleton.INDEX["leftFoot"], 1]), 3), "takes": {}}
    for name in names:
        clip = contacts.with_contacts(lafan1.load_take(texts[name], name, calibration))
        torch.save({"rot6d": clip.rot6d, "root_position": clip.root_position,
                    "contacts": clip.contacts, "source": clip.source, "subject": clip.subject,
                    "notes": clip.notes}, OUT / f"{Path(name).stem}.pt")
        checks = clip_checks(clip)
        checks["bone_direction_error_deg"] = round(bone_error_degrees(sources[name], calibration),
                                                   4)
        checks["subject"], checks["action"] = clip.subject, clip.notes["action"]
        report["takes"][Path(name).stem] = checks
        print(f"{name:32} {checks}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
