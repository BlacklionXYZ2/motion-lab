"""Why does ARDY come apart after a minute of idling? (2026-10-30, stress_service.py's finding)

    .venv\\Scripts\\python idle_test.py [--seconds 120] [--variants NAME ...]

Runs the service's own stepping for a long idle under several variants and prints, per 10 s:
energy (rad/s), turning (deg/s), hips height (m) and the share of joint-frames outside the
data's limits. Writes ``outputs/idle-test.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import stress_service as ss
import torch
from scipy.spatial.transform import Rotation

VARIANTS = {
    "anchor x2, neutral, seed 0": ("fp16", 4, 4, "-|-|-", 0.0, 0, 0, (1, 21)),
    "anchor x2, neutral, seed 1": ("fp16", 4, 4, "-|-|-", 0.0, 1, 0, (1, 21)),
    "anchor x2, neutral, seed 3": ("fp16", 4, 4, "-|-|-", 0.0, 3, 0, (1, 21)),
    "anchor x2, idle, seed 1": ("fp16", 4, 4, "idle|-|-", 0.0, 1, 0, (1, 21)),
    "anchor x2, cheerful high, seed 0": ("fp16", 4, 4, "-|cheerful|high", 0.0, 0, 0, (1, 21)),
    "anchor x2, calm, seed 1": ("fp16", 4, 4, "-|calm|-", 0.0, 1, 0, (1, 21)),
    # name: (precision, steps, history, key, speed or None, random seed[, guard retries,
    # idle anchoring])
    "anchor, seed 0": ("fp16", 4, 4, "idle|-|-", 0.0, 0, 0, True),
    "anchor, seed 1": ("fp16", 4, 4, "idle|-|-", 0.0, 1, 0, True),
    "anchor, seed 2": ("fp16", 4, 4, "idle|-|-", 0.0, 2, 0, True),
    "anchor, seed 3": ("fp16", 4, 4, "idle|-|-", 0.0, 3, 0, True),
    "anchor, neutral prompt": ("fp16", 4, 4, "-|-|-", 0.0, 0, 0, True),
    "anchor, neutral prompt, seed 1": ("fp16", 4, 4, "-|-|-", 0.0, 1, 0, True),
    "anchor, neutral prompt, seed 2": ("fp16", 4, 4, "-|-|-", 0.0, 2, 0, True),
    "anchor, neutral prompt, seed 3": ("fp16", 4, 4, "-|-|-", 0.0, 3, 0, True),
    "anchor, calm, seed 0": ("fp16", 4, 4, "-|calm|-", 0.0, 0, 0, True),
    "anchor, calm, seed 1": ("fp16", 4, 4, "-|calm|-", 0.0, 1, 0, True),
    "anchor, cheerful high, seed 0": ("fp16", 4, 4, "-|cheerful|high", 0.0, 0, 0, True),
    "guard 2, seed 0": ("fp16", 4, 4, "idle|-|-", 0.0, 0, 2),
    "guard 2, seed 1": ("fp16", 4, 4, "idle|-|-", 0.0, 1, 2),
    "guard 2, seed 2": ("fp16", 4, 4, "idle|-|-", 0.0, 2, 2),
    "guard 4, seed 1": ("fp16", 4, 4, "idle|-|-", 0.0, 1, 4),
    "service (idle, 0 m/s)": ("fp16", 4, 4, "idle|-|-", 0.0, 0),
    "random seed 1": ("fp16", 4, 4, "idle|-|-", 0.0, 1),
    "random seed 2": ("fp16", 4, 4, "idle|-|-", 0.0, 2),
    "no velocity": ("fp16", 4, 4, "idle|-|-", None, 0),
    "neutral prompt": ("fp16", 4, 4, "-|-|-", 0.0, 0),
    "history 8": ("fp16", 4, 8, "idle|-|-", 0.0, 0),
    "history 16": ("fp16", 4, 16, "idle|-|-", 0.0, 0),
    "10 steps": ("fp16", 10, 4, "idle|-|-", 0.0, 0),
    "fp32": ("fp32", 4, 4, "idle|-|-", 0.0, 0),
}


def series(frames: list[dict]) -> dict[str, list[float]]:
    rot = np.stack([Rotation.from_quat(f["rot"]).as_matrix() for f in frames])
    root = np.asarray([f["root"] for f in frames])
    r = torch.tensor(rot, dtype=torch.float32)[None]
    energy = ss.stress.energy(r)[0].numpy()
    yaw = np.unwrap(ss.formats.heading(r[0, :, 0]).numpy())
    turn = np.abs(np.diff(yaw)) * 60
    lim = ss.stress.m1.joint_limits().without_hips()
    outside = (ss.stress.m1.limit_excess(r[0, :, 1:], lim) > 0).float().mean(-1).numpy()
    n = 600
    cut = lambda x: [round(float(x[s:s + n].mean()), 3) for s in range(0, len(x) - n + 1, n)]  # noqa: E731
    return {"energy": cut(energy), "turn_deg_s": cut(np.degrees(turn)),
            "hips": cut(root[:, 1]), "outside": cut(outside)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--out", default="idle-test.json")
    args = parser.parse_args()
    bases: dict[str, ss.service.MotionService] = {}
    report = {}
    for name, (precision, steps, history, key, speed, seed, *more) in VARIANTS.items():
        if args.variants and name not in args.variants:
            continue
        if precision not in bases:
            bases[precision] = ss.service.MotionService(precision, steps)
        svc = ss.fresh(bases[precision], steps, retries=more[0] if more else 0,
                       anchoring=len(more) > 1 and bool(more[1]))
        if len(more) > 1 and isinstance(more[1], tuple):
            svc.streamer.anchor_frames = more[1]
        svc.streamer.history = history
        torch.manual_seed(seed)
        np.random.seed(seed)
        svc.seed(ss.seed_walk())
        facing = ss.facing_of(ss.seed_walk())
        started = time.time()
        frames = []
        for _ in range(round(args.seconds / ss.CHUNK_S)):
            velocity = None if speed is None else [speed * math.sin(facing),
                                                   speed * math.cos(facing)]
            svc.signal({"key": key, "velocity": velocity})
            frames += svc.next_chunk()["frames"]
        report[name] = series(frames)
        report[name]["anchors"] = svc.anchors
        report[name]["rejected_draws"] = svc.rejected
        report[name]["unresolved_chunks"] = svc.unresolved
        report[name]["generation_s"] = round(time.time() - started, 1)
        ss.say(f"{name} ({time.time() - started:.0f} s): {json.dumps(report[name])}")
    (ss.HERE / "outputs" / args.out).write_text(json.dumps(report, indent=1), "utf-8")


if __name__ == "__main__":
    main()
