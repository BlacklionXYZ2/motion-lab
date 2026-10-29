"""Build AMASS for M1: convert every SMPL-H sequence onto the canonical skeleton.

    .venv\\Scripts\\python scripts\\build_amass.py [data\\amass]

Expects AMASS's own layout, as unpacked from its downloads (the "SMPL+H G" body data):
``data/amass/<dataset>/<subject>/<sequence>_poses.npz`` (``shape.npz`` files are skipped).
Each sequence becomes ``data/processed/amass/<dataset>__<subject>__<sequence>.pt``: a 60 Hz
canonical clip with foot contacts, its performer ``amass/<dataset>/<subject>`` (which decides its
split, ``m1_data.split_of``). Writes ``outputs/m1/amass_build.json`` with each sequence's root
scale, length and planted-ankle height, and lists the files it couldn't read.

AMASS's licence comes with its download (owner's registration); like LAFAN1, the processed
copies stay on this PC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import fk, rotation, skeleton  # noqa: E402
from motionlab.datasets import amass, contacts  # noqa: E402
from motionlab.formats import DT  # noqa: E402

OUT = ROOT / "data" / "processed" / "amass"
REPORT = ROOT / "outputs" / "amass_build.json"
MIN_SECONDS = 1.0


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else LAB / "data" / "amass"
    files = sorted(p for p in source.rglob("*.npz")
                   if p.name != "shape.npz" and "sample" not in p.parts)
    if not files:
        raise SystemExit(f"No AMASS .npz files under {source}")
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"sequences": {}, "skipped": {}}
    seconds = 0.0
    for number, path in enumerate(files, 1):
        relative = path.relative_to(source)
        if len(relative.parts) < 3:
            report["skipped"][str(relative)] = "not in <dataset>/<subject>/"  # type: ignore[index]
            continue
        dataset, subject = relative.parts[0], relative.parts[1]
        try:
            clip = amass.load_clip(path, f"amass/{dataset}/{subject}")
        except (KeyError, ValueError, RuntimeError) as error:
            report["skipped"][str(relative)] = str(error)  # type: ignore[index]
            continue
        if clip.frames * DT < MIN_SECONDS:
            report["skipped"][str(relative)] = "shorter than 1 s"  # type: ignore[index]
            continue
        clip = contacts.with_contacts(clip)
        name = "__".join((dataset, subject, path.stem))
        torch.save({"rot6d": clip.rot6d, "root_position": clip.root_position,
                    "contacts": clip.contacts, "source": clip.source, "subject": clip.subject,
                    "notes": clip.notes}, OUT / f"{name}.pt")
        _, positions = fk.forward_kinematics(rotation.rot6d_to_matrix(clip.rot6d),
                                             clip.root_position, skeleton.load())
        assert clip.contacts is not None
        ankles = positions[:, [skeleton.INDEX["leftFoot"], skeleton.INDEX["rightFoot"]], 1]
        planted = ankles[clip.contacts[:, [0, 2]]]
        report["sequences"][name] = {  # type: ignore[index]
            "seconds": round(clip.frames * DT, 1), "root_scale": clip.notes["root_scale"],
            "source_fps": clip.notes["source_fps"],
            "planted_ankle_y_median": round(float(planted.median()), 3)
            if planted.numel() else None}
        seconds += clip.frames * DT
        if number % 200 == 0:
            print(f"{number}/{len(files)} files, {seconds / 3600:.1f} h so far")
    report["hours"] = round(seconds / 3600, 2)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{len(report['sequences'])} sequences, {report['hours']} h; "  # type: ignore[arg-type]
          f"{len(report['skipped'])} skipped. Report: {REPORT}")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
