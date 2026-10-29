"""Build 100STYLE and CMU for M1: retarget every take onto the canonical skeleton and check it.

    .venv\\Scripts\\python scripts\\build_datasets.py style100
    .venv\\Scripts\\python scripts\\build_datasets.py cmu

Reads ``data/100style/100STYLE.zip`` or ``data/cmu/allasfamc.zip`` (with BABEL's labels from
``data/babel/babel_v1-0_release.zip`` for CMU) and writes ``data/processed/<dataset>/*.pt``:
60 Hz canonical clips with foot contacts and their labels, plus ``outputs/m1/<dataset>_build.json``
with each take's checks (worst bone-direction error against the source, standing hips height,
planted-ankle height, share of time planted).

- **100STYLE:** performer ``100style/<Style>`` (splits by style, ``style100.split_of_style``);
  labels: the style and the gait (forwards walk, idle, …).
- **CMU:** performer ``cmu/<subject>`` (splits by subject); each subject is calibrated from their
  own skeleton (``retarget.leg_scale``) and each take is placed on the floor from its planted
  ankles. Some subjects were captured at 60 fps rather than 120: a take's rate comes from BABEL's
  duration where BABEL has the take, else the subject's usual rate, else 120. Labels: BABEL's
  sequence labels and frame segments (seconds), where BABEL has the take.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import fk, rotation, skeleton  # noqa: E402
from motionlab.datasets import bvh, cmu, contacts, retarget, style100  # noqa: E402
from motionlab.datasets.lafan1 import BONE_CHILD, resample  # noqa: E402
from motionlab.formats import FPS, Clip  # noqa: E402

DATA = LAB / "data"  # raw downloads
VERSION = "v1"
REPORTS = ROOT / "outputs"


def checks(clip: Clip, source_bones: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    _, positions = fk.forward_kinematics(rotation.rot6d_to_matrix(clip.rot6d.double()),
                                         clip.root_position.double(), skeleton.load(torch.float64))
    assert clip.contacts is not None
    ankles = positions[:, [skeleton.INDEX["leftFoot"], skeleton.INDEX["rightFoot"]], 1]
    planted = ankles[clip.contacts[:, [0, 2]]]
    return {"seconds": round(clip.frames / FPS, 1),
            "hips_y_median": round(float(clip.root_position[:, 1].median()), 3),
            "planted_ankle_y_median": (round(float(planted.median()), 3) if planted.numel()
                                       else None),
            "planted_share": round(float(clip.contacts.float().mean()), 3)}


def bone_error(local: torch.Tensor, root: torch.Tensor, positions: dict[str, torch.Tensor],
               calibration: retarget.Calibration, profile: retarget.Profile) -> float:
    _, p = fk.forward_kinematics(local, root, skeleton.load(torch.float64))
    worst = 0.0
    for name, child in BONE_CHILD.items():
        if child is None:
            continue
        a = p[:, skeleton.INDEX[child]] - p[:, skeleton.INDEX[name]]
        b = (positions[profile.position_source[child]]
             - positions[profile.position_source[name]]) @ calibration.world.T
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
        worst = max(worst, float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).max()))
    return round(worst, 3)


def save(out: Path, name: str, clip: Clip, labels: dict[str, Any]) -> None:
    torch.save({"rot6d": clip.rot6d, "root_position": clip.root_position,
                "contacts": clip.contacts, "source": clip.source, "subject": clip.subject,
                "notes": clip.notes, "labels": labels}, out / f"{name}.pt")


def make_clip(local: torch.Tensor, root: torch.Tensor, fps: float, source: str, subject: str,
              notes: dict[str, str]) -> Clip:
    local, root = resample(local, root, fps, FPS)
    clip = Clip(rotation.matrix_to_rot6d(local).float(), root.float(), source=source,
                subject=subject, notes=notes)
    clip.validate()
    return contacts.with_contacts(clip)


def build_style100() -> dict[str, Any]:
    archive = zipfile.ZipFile(DATA / "100style" / "100STYLE.zip")
    names = sorted(n for n in archive.namelist() if n.endswith(".bvh")
                   and "__MACOSX" not in n and "/._" not in n)
    read = lambda n: bvh.parse(archive.read(n).decode("utf-8", "replace"))  # noqa: E731
    sample = [read(n) for n in names[::9]]
    style100.PROFILE.check(sample[0].names)
    calibration = retarget.calibrate(style100.rest_positions(sample[0]),
                                     [s.root_position()[0] for s in sample], style100.PROFILE)
    out = ROOT / "data" / "processed" / "100style"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"root_scale": calibration.root_scale, "takes": {}}
    for number, name in enumerate(names, 1):
        source = read(name)
        style, gait = style100.take_info(name)
        rotations, positions = style100.world_and_positions(source)
        local, root = retarget.to_canonical(rotations, positions, calibration, style100.PROFILE)
        clip = make_clip(local, root, source.fps, "100style", f"100style/{style}",
                         {"take": Path(name).stem, "style": style, "gait": gait})
        save(out, Path(name).stem, clip, {"style": style, "gait": style100.GAITS.get(gait, gait)})
        result = checks(clip)
        result["bone_direction_error_deg"] = bone_error(local, root, positions, calibration,
                                                        style100.PROFILE)
        report["takes"][Path(name).stem] = result
        if number % 100 == 0:
            print(f"100STYLE: {number}/{len(names)}")
    return report


def babel_cmu() -> dict[str, dict[str, Any]]:
    path = DATA / "babel" / "babel_v1-0_release.zip"
    if not path.exists():
        return {}
    archive = zipfile.ZipFile(path)
    out: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test", "extra_train", "extra_val"):
        for row in json.loads(archive.read(f"babel_v1.0_release/{split}.json")).values():
            if not row["feat_p"].startswith("CMU/"):
                continue
            take = re.sub(r"_poses\.npz$", "", row["feat_p"].split("/")[-1])
            entry = out.setdefault(take, {"duration": row["dur"], "sequence": [], "segments": []})
            if row.get("seq_ann"):
                entry["sequence"] += [{"label": x["proc_label"], "categories": x["act_cat"]}
                                      for x in row["seq_ann"]["labels"]]
            if row.get("frame_ann") and not entry["segments"]:
                entry["segments"] = [{"start": x["start_t"], "end": x["end_t"],
                                      "label": x["proc_label"], "categories": x["act_cat"]}
                                     for x in row["frame_ann"]["labels"]]
    return out


def build_cmu() -> dict[str, Any]:
    archive = zipfile.ZipFile(DATA / "cmu" / "allasfamc.zip")
    base = "all_asfamc/subjects/"
    takes: dict[str, list[str]] = defaultdict(list)
    for name in archive.namelist():
        if name.endswith(".amc"):
            takes[name.split("/")[-2]].append(name)
    labels = babel_cmu()
    out = ROOT / "data" / "processed" / "cmu"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"subjects": {}, "takes": {}, "skipped": {}}
    for subject in sorted(takes):
        asf = cmu.parse_asf(archive.read(f"{base}{subject}/{subject}.asf")
                            .decode("utf-8", "replace"))
        try:
            cmu.PROFILE.check(asf.bones)
        except ValueError as error:
            report["skipped"][subject] = str(error)
            continue
        rest = cmu.rest_positions(asf)
        calibration = retarget.calibrate(rest, [], cmu.PROFILE)
        calibration = retarget.Calibration(calibration.world, calibration.alignment,
                                           retarget.leg_scale(rest, calibration.world,
                                                              cmu.PROFILE),
                                           calibration.match_bone_directions)
        frames_of = {}
        for name in sorted(takes[subject]):
            frames_of[name] = cmu.parse_amc(archive.read(name).decode("utf-8", "replace"))
        rates = Counter()
        for name, frames in frames_of.items():
            take = Path(name).stem
            if take in labels and frames:
                ratio = labels[take]["duration"] / (len(frames) / cmu.FPS)
                rates[60 if ratio > 1.6 else 120] += 1
        usual = rates.most_common(1)[0][0] if rates else 120
        report["subjects"][subject] = {"root_scale": round(calibration.root_scale, 5),
                                       "usual_fps": usual}
        for name, frames in frames_of.items():
            take = Path(name).stem
            if len(frames) < 60:
                report["skipped"][take] = "fewer than 60 frames"
                continue
            fps = float(usual)
            if take in labels:
                ratio = labels[take]["duration"] / (len(frames) / cmu.FPS)
                fps = 60.0 if ratio > 1.6 else 120.0
            try:
                rotations, positions = cmu.forward(asf, frames)
                local, root = retarget.to_canonical(rotations, positions, calibration,
                                                    cmu.PROFILE)
                root = retarget.place_on_floor(local, root, fps)
                clip = make_clip(local, root, fps, "cmu", f"cmu/{subject}",
                                 {"take": take, "source_fps": f"{fps:g}"})
            except (KeyError, ValueError, RuntimeError, IndexError) as error:
                report["skipped"][take] = f"{type(error).__name__}: {error}"
                continue
            save(out, take, clip, labels.get(take, {}))
            result = checks(clip)
            result["bone_direction_error_deg"] = bone_error(local, root, positions, calibration,
                                                            cmu.PROFILE)
            result["source_fps"] = fps
            result["babel"] = take in labels
            report["takes"][take] = result
        print(f"CMU subject {subject}: {len(frames_of)} takes")
    return report


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    builders = {"style100": build_style100, "cmu": build_cmu}
    if which not in builders:
        raise SystemExit(f"usage: build_datasets.py {{{'|'.join(builders)}}}")
    report = builders[which]()
    takes = report["takes"].values()
    errors = [t["bone_direction_error_deg"] for t in takes]
    report["summary"] = {
        "takes": len(report["takes"]), "hours": round(sum(t["seconds"] for t in takes) / 3600, 2),
        "worst_bone_direction_error_deg": max(errors) if errors else None,
        "skipped": len(report.get("skipped", {}))}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{which}_build_{VERSION}.json").write_text(json.dumps(report, indent=1),
                                                         encoding="utf-8")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
