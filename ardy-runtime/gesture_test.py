"""Do gestures happen after she has been standing? (2026-10-30, stress_service.py's finding)

    .venv\\Scripts\\python gesture_test.py

After 3 s of standing, wave, point and lean each barely moved her (joint energy 0.05 to 0.13
rad/s). This tries the text guidance weight (the service's cfg is (2.0, 2.0): text, then
constraints) on: 3 s idle, then 6 s of the gesture, two random seeds each. Per gesture it
reports the right hand's highest point above the head, the right arm's mean joint speed, and the
spine's largest bend from upright. Writes ``outputs/gesture-test.json``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import stress_service as ss
import torch
from scipy.spatial.transform import Rotation

JOINT = ss.skeleton.INDEX
ARM = [JOINT[n] for n in ("rightUpperArm", "rightLowerArm", "rightHand")]
SPINE = [JOINT[n] for n in ("spine", "chest", "upperChest")]


def measure(frames: list[dict], start: int) -> dict[str, float]:
    rot = np.stack([Rotation.from_quat(f["rot"]).as_matrix() for f in frames])[start:]
    root = np.asarray([f["root"] for f in frames])[start:]
    r = torch.tensor(rot, dtype=torch.float32)[None]
    p = ss.stress.positions(r, torch.tensor(root, dtype=torch.float32)[None])[0]
    w = ss.stress.rotation.angular_velocity(r[0, :-1, ARM], r[0, 1:, ARM], 1 / 60).norm(dim=-1)
    bend = np.degrees(Rotation.from_matrix(rot[:, SPINE].reshape(-1, 3, 3)).magnitude()
                      ).reshape(len(rot), -1).sum(-1)
    hand_over_head = (p[:, JOINT["rightHand"], 1] - p[:, JOINT["head"], 1]).max()
    return {"hand_over_head_m": round(float(hand_over_head), 3),
            "arm_energy_rad_s": round(float(w.mean()), 3),
            "spine_bend_deg_max": round(float(bend.max()), 1)}


def main() -> None:
    base = ss.service.MotionService("fp16", 4)
    report = {}
    for cfg in (2.0, 3.5, 5.0):
        for gesture in ("wave", "point", "lean"):
            for seed in (0, 1):
                torch.manual_seed(seed)
                svc = ss.fresh(base, 4, anchoring=True)
                svc.streamer.cfg = (cfg, 2.0)
                svc.seed(ss.seed_walk())
                facing = ss.facing_of(ss.seed_walk())
                frames = []
                for k in range(round(9 / ss.CHUNK_S)):
                    key = "idle|-|-" if k * ss.CHUNK_S < 3 else f"{gesture}|-|-"
                    svc.signal({"key": key, "velocity": [0.0 * math.sin(facing), 0.0]})
                    frames += svc.next_chunk()["frames"]
                name = f"cfg {cfg}, {gesture}, seed {seed}"
                report[name] = measure(frames, 180)
                ss.say(name, report[name])
    (ss.HERE / "outputs" / "gesture-test.json").write_text(json.dumps(report, indent=1), "utf-8")


if __name__ == "__main__":
    main()
