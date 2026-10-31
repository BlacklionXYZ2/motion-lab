"""Stress the ARDY motion service the way SORA drives it, looking for odd behaviour (2026-10-30).

    .venv\\Scripts\\python stress_service.py [--precision fp16] [--steps 4] [--only NAME ...]

``service.MotionService`` runs in this process and is stepped chunk by chunk (0.4 s each), with a
signal before every chunk, as SORA's player sends one every 0.4 s. In-process stepping makes each
signal apply exactly where the scenario says (over the pipe, a ``--fast`` service runs ahead).

Every scenario's canonical 60 Hz output is scored with motion-lab v5's stress measures
(``versions/v5/motionlab/stress.py``) and a few service-specific ones:

- **jumps:** the largest root step (cm a frame) and joint turn (degrees a frame), and the
  largest joint acceleration against the recordings' 99th percentile (523 rad/s²);
- **tracking:** per segment, the speed reached and the travel direction against the request;
- **the body:** gliding, toes in the floor (below −3 cm), hovering (every foot joint over
  15 cm), legs crossed, a wrist in the torso, the head below the hips, outside the data's limits.

Writes ``outputs/stress-service-<precision>-s<steps>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "external" / "ardy" / "nv-tlabs-ardy"))

import service  # noqa: E402, I001  (first: redirects stdout, and puts motionlab on the path)
import test_service  # noqa: E402
from motionlab import formats, skeleton  # noqa: E402  (v4's package, which stream.py loads)


def _v5_stress():
    """v5's stress measures, loaded by path on top of v4's ``motionlab`` (one package of that
    name per process; the skeleton, FK and limits the measures use are the same in both)."""
    import importlib.util

    path = HERE.parent / "versions" / "v5" / "motionlab" / "stress.py"
    spec = importlib.util.spec_from_file_location("motionlab_v5_stress", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module


stress = _v5_stress()

OUT = sys.__stdout__
DATA_P99 = 522.9  # recordings' 99th-percentile peak joint acceleration (v5 stress report)
CHUNK_S = 0.4


def say(*parts: object) -> None:
    print(*parts, file=OUT, flush=True)


def quats_to_frames(local: np.ndarray, root: np.ndarray) -> list[dict]:
    q = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(len(local), -1, 4)
    return [{"root": root[i].tolist(), "rot": q[i].tolist()} for i in range(len(local))]


def seed_walk() -> dict:
    local, root = test_service.canonical_walk()
    return {"type": "seed", "fps": 60, "frames": quats_to_frames(local, root)}


def seed_t_pose(seconds: float = 1.0) -> dict:
    n = int(seconds * 60)
    local = np.broadcast_to(np.eye(3), (n, 22, 3, 3)).copy()
    root = np.zeros((n, 3))
    root[:, 1] = float(skeleton.load().offsets[0, 1])
    return {"type": "seed", "fps": 60, "frames": quats_to_frames(local, root)}


def seed_frozen() -> dict:
    local, root = test_service.canonical_walk()
    local = np.repeat(local[:1], len(local), 0)
    root = np.repeat(root[:1], len(root), 0)
    return {"type": "seed", "fps": 60, "frames": quats_to_frames(local, root)}


_LYING: dict | None = None


def make_lying_seed(base: service.MotionService, steps: int) -> dict:
    """2 s of her lying on the ground after a fall: ARDY's own "falls to the ground" from a walk,
    with the first random seed whose fall ends with the hips under 0.5 m."""
    for attempt in range(20):
        torch.manual_seed(100 + attempt)
        svc = fresh(base, steps)
        svc.seed(seed_walk())
        frames = []
        for _ in range(15):  # 6 s
            svc.signal({"key": "fall|-|-", "velocity": [0.0, 0.0]})
            frames += svc.next_chunk()["frames"]
        if frames[-1]["root"][1] < 0.5:
            say(f"lying seed: random seed {100 + attempt}, hips {frames[-1]['root'][1]:.2f} m")
            return {"type": "seed", "fps": 60, "frames": frames[-120:]}
    raise RuntimeError("ARDY never fell in 20 tries")


def facing_of(seed: dict | None) -> float:
    if seed is None:
        return 0.0
    r = Rotation.from_quat(seed["frames"][-1]["rot"][0]).as_matrix()
    return float(math.atan2(r[0, 2], r[2, 2]))  # formats.heading: the hips' +Z on the ground


# A segment: (seconds, key, speed m/s or None for no velocity, direction relative to the seed's
# facing in degrees, +90 = her left).
SCENARIOS: dict[str, tuple[str, list[tuple[float, str, float | None, float]]]] = {
    "turns": ("walk", [(6, "walk|-|-", 1.2, 0), (6, "walk|-|-", 1.2, 90),
                       (6, "walk|-|-", 1.2, 270), (6, "run|-|-", 2.5, 270),
                       (6, "idle|-|-", 0.0, 0)]),
    "flicker": ("walk", [(CHUNK_S, "walk|-|-", 1.2, 0), (CHUNK_S, "idle|-|-", 0.0, 0)] * 15),
    "sit while asked to travel": ("walk", [(8, "sit|-|-", 1.2, 0), (6, "walk|-|-", 0.0, 0)]),
    "extremes": ("walk", [(6, "run|-|-", 5.0, 0), (6, "walk|-|-", 1.0, 180),
                          (6, "walk|-|-", 1.0, 90), (4, "idle|-|-", 0.0, 0)]),
    "t-pose seed": ("t-pose", [(6, "idle|-|-", 0.0, 0), (4, "walk|-|-", 1.2, 0)]),
    "frozen seed": ("frozen", [(6, "walk|-|-", 1.2, 0)]),
    "no seed": ("none", [(6, "idle|-|-", 0.0, 0), (4, "walk|-|-", 1.2, 0)]),
    "no velocity": ("walk", [(8, "walk|-|-", None, 0), (8, "run|-|-", None, 0)]),
    "long idle": ("walk", [(120, "idle|-|-", 0.0, 0)]),
    "fall and get up": ("walk", [(6, "fall|-|-", 0.0, 0), (8, "get_up|-|-", 0.0, 0),
                                 (4, "idle|-|-", 0.0, 0)]),
    # what SORA's player sends after a physical fall: "fall", then idle
    "fall then idle": ("walk", [(6, "fall|-|-", 0.0, 0), (10, "idle|-|-", 0.0, 0),
                                (4, "walk|-|-", 1.2, 0)]),
    # from a real lying pose (make_lying_seed), three random seeds each
    **{f"lying, idle (seed {k})": ("lying", [(10, "idle|-|-", 0.0, 0), (4, "walk|-|-", 1.2, 0)])
       for k in range(3)},
    **{f"lying, get up (seed {k})": ("lying", [(8, "get_up|-|-", 0.0, 0),
                                               (4, "idle|-|-", 0.0, 0)]) for k in range(3)},
    # SORA's sequences: sit, then sent somewhere; a wave and a lean while standing
    **{f"sit then walk (seed {k})": ("walk", [(6, "sit|-|-", 0.0, 0), (6, "walk|-|-", 1.2, 0),
                                              (4, "idle|-|-", 0.0, 0)]) for k in range(2)},
    "wave, lean, point": ("walk", [(3, "idle|-|-", 0.0, 0), (5, "wave|-|-", 0.0, 0),
                                   (5, "lean|-|-", 0.0, 0), (5, "point|-|-", 0.0, 0),
                                   (4, "idle|-|-", 0.0, 0)]),
    "emotion": ("walk", [(8, "walk|sad|high", 1.2, 0), (8, "walk|cheerful|high", 1.2, 0),
                         (8, "walk|frustrated|high", 1.2, 0)]),
}


def fresh(base: service.MotionService, steps: int, retries: int = 0,
          anchoring: bool = False) -> service.MotionService:
    """A service in its just-started state that shares ``base``'s loaded model (what
    ``MotionService.__init__`` sets, without loading the networks again)."""
    svc = service.MotionService.__new__(service.MotionService)
    svc.retries, svc.rejected, svc.unresolved, svc.key = retries, 0, 0, "-|-|-"
    svc.anchoring, svc.still, svc.anchors = anchoring, 0, 0
    svc.standing, svc.getting_up, svc.get_ups = service.default_standing(), False, 0
    for name in ("model", "rep", "names", "parents", "contact_names", "fps", "phrases",
                 "load_s"):
        setattr(svc, name, getattr(base, name))
    svc.streamer = service.Streamer(base.model, service.HISTORY, steps)
    svc.text, svc.prompt = svc.phrases.feature("-|-|-")
    svc.velocity, svc.seq, svc.t, svc.last = None, 0, 0.0, None
    return svc


def run(svc: service.MotionService, seed_kind: str,
        segments: list[tuple[float, str, float | None, float]]) -> dict:
    seed = {"walk": seed_walk, "t-pose": seed_t_pose, "frozen": seed_frozen,
            "lying": lambda: _LYING}.get(seed_kind, lambda: None)()
    if seed is not None:
        svc.seed(seed)
    facing = facing_of(seed)
    frames, prompts, marks = [], [], []
    for seconds, key, speed, direction in segments:
        marks.append(len(frames))
        chunks = max(1, round(seconds / CHUNK_S))
        for _ in range(chunks):
            yaw = facing + math.radians(direction)
            velocity = None if speed is None else [speed * math.sin(yaw), speed * math.cos(yaw)]
            svc.signal({"key": key, "velocity": velocity})
            chunk = svc.next_chunk()
            frames += chunk["frames"]
            prompts.append(chunk["prompt"])
    marks.append(len(frames))
    rot = np.stack([Rotation.from_quat(f["rot"]).as_matrix() for f in frames])
    root = np.asarray([f["root"] for f in frames])
    r = torch.tensor(rot, dtype=torch.float32)[None]
    p = torch.tensor(root, dtype=torch.float32)[None]
    first_seam = None
    if seed is not None:
        last = seed["frames"][-1]
        seam_rot = Rotation.from_quat(last["rot"]).inv() * Rotation.from_quat(frames[0]["rot"])
        first_seam = {"root_cm": round(float(np.linalg.norm(np.asarray(last["root"])
                                                             - root[0]) * 100), 1),
                      "worst_joint_deg": round(float(np.degrees(seam_rot.magnitude()).max()), 1)}
    steps = np.linalg.norm(np.diff(root, axis=0), axis=1)
    rel = np.swapaxes(rot[:-1], -1, -2) @ rot[1:]
    turn = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    accel = stress.acceleration_peak(r)[0]
    out: dict[str, object] = {
        "seconds": round(len(frames) / 60, 1),
        "stable": bool(np.isfinite(rot).all() and np.isfinite(root).all()),
        "seam_from_seed": first_seam,
        "largest_root_step_cm": round(float(steps.max() * 100), 2),
        "largest_joint_turn_deg_per_frame": round(float(turn[:, 1:].max()), 1),
        "peak_accel_max_over_data_p99": round(float(accel.max()) / DATA_P99, 2),
        "peak_accel_share_over_data_p99": round(float((accel > DATA_P99).float().mean()), 4),
        "outside_limits_share": round(stress.limits_share(r), 4),
        "energy_rad_s": round(float(stress.energy(r).mean()), 3),
        **stress.body(r, p, 0.0),
        "segments": [],
    }
    yaw = formats.heading(r[0, :, 0]).numpy()
    for (_seconds, key, speed, direction), a, b in zip(segments, marks[:-1], marks[1:],
                                                       strict=True):
        if b - a < 30 or len(segments) > 12:
            continue  # flicker: per-segment numbers mean nothing
        tail = slice(max(a, b - 120), b)  # the last 2 s
        travel = root[tail][-1, [0, 2]] - root[tail][0, [0, 2]]
        dt = (tail.stop - tail.start - 1) / 60
        net = float(np.linalg.norm(travel) / dt)
        path = float(np.linalg.norm(np.diff(root[tail][:, [0, 2]], axis=0), axis=1).sum() / dt)
        seg: dict[str, object] = {"key": key, "asked_m_s": speed, "asked_direction_deg": direction,
                                  "net_speed_last_2s": round(net, 3),
                                  "path_speed_last_2s": round(path, 3),
                                  "hips_m_last_2s": round(float(root[tail, 1].mean()), 3),
                                  "energy_last_2s": round(float(stress.energy(
                                      r[:, tail]).mean()), 3)}
        if speed and net > 0.2:
            want = facing + math.radians(direction)
            got = math.atan2(travel[0], travel[1])
            seg["travel_direction_error_deg"] = round(abs(math.degrees(
                math.atan2(math.sin(want - got), math.cos(want - got)))), 1)
            face_err = np.degrees(np.arctan2(np.sin(want - yaw[tail]), np.cos(want - yaw[tail])))
            seg["facing_error_deg"] = round(float(np.abs(face_err).mean()), 1)
        seg.update({k: v for k, v in stress.body(r[:, a:b], p[:, a:b], 0.0).items()
                    if k in ("glide_share_when_moving", "airborne_share", "toe_in_floor_share",
                             "head_below_hips_share")})
        out["segments"].append(seg)  # type: ignore[union-attr]
    if len(frames) >= 60 * 60:  # long runs: energy and drift by 30 s
        e = stress.energy(r)[0]
        out["energy_by_30s"] = [round(float(e[s:s + 1800].mean()), 3)
                                for s in range(0, len(e) - 1799, 1800)]
        out["drift_m_by_30s"] = [round(float(np.linalg.norm(root[s + 1799, [0, 2]]
                                                             - root[s, [0, 2]])), 3)
                                 for s in range(0, len(root) - 1799, 1800)]
        out["turned_deg_total"] = round(float(np.degrees(np.abs(np.diff(np.unwrap(yaw))).sum())),
                                        1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-anchor", action="store_true", help="as the service's flag")
    parser.add_argument("--retries", type=int, default=0, help="as the service's flag")
    parser.add_argument("--tag", default="", help="added to the report's file name")
    args = parser.parse_args()
    torch.manual_seed(0)
    started = time.time()
    base = service.MotionService(args.precision, args.steps)
    say(f"loaded in {time.time() - started:.1f} s")
    report: dict[str, object] = {"precision": args.precision, "steps": args.steps}
    global _LYING
    for name, (seed_kind, segments) in SCENARIOS.items():
        if args.only and name not in args.only:
            continue
        if seed_kind == "lying" and _LYING is None:
            _LYING = make_lying_seed(base, args.steps)
        if "(seed " in name:  # repeatable per scenario, whatever ran before
            torch.manual_seed(int(name.split("(seed ")[1].rstrip(")")))
        t = time.time()
        svc = fresh(base, args.steps, args.retries, anchoring=not args.no_anchor)
        report[name] = run(svc, seed_kind, segments)
        report[name]["anchors_taken"] = svc.anchors
        report[name]["get_ups"] = svc.get_ups
        report[name]["chunks_failing_guard"] = svc.unresolved
        say(f"{name}: {time.time() - t:.0f} s")
    out = HERE / "outputs" / f"stress-service-{args.precision}-s{args.steps}{args.tag}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    say(json.dumps(report, indent=1))
    say(f"Report: {out}")


if __name__ == "__main__":
    main()
