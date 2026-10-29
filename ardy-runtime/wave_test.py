"""After standing, does a new prompt take over? History length and replan cadence (2026-10-26).

    .venv\\Scripts\\python wave_test.py

Streams 10 s of standing, then 10 s of a new prompt, and reports the right arm's mean angular
speed (shoulder, upper arm, forearm and hand, rad/s) per 2 s and the right hand's highest point.
Standing is about 0.05 and the hand about 0.95 m. Settings tried:
- the history length (NVIDIA's generate script: 160 frames; its interactive demo: 4);
- how many of each generated 40 frames are kept before replanning (``keep``).

From a fresh start the same prompts reach 1.35 rad/s and lift the hand to 1.63 m
(``text_check.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe  # noqa: E402
from stream import Streamer  # noqa: E402

STAND = "a person stands still, relaxed"
PROMPTS = ["a person waves with their right hand while standing",
           "a person raises both arms above their head"]
ARM = ("RightShoulder", "RightArm", "RightForeArm", "RightHand")
SETTINGS = [(160, 40), (4, 40), (4, 20), (4, 8), (40, 20)]  # (history frames, kept frames)


def arm(streamer: Streamer, start: int) -> tuple[list[float], float]:
    out = streamer.rep.inverse(streamer.motion, is_normalized=True)
    names = streamer.rep.skeleton.bone_order_names
    local = out["local_rot_mats"][0].numpy()[:, [names.index(n) for n in ARM]]
    rel = np.swapaxes(local[:-1], -1, -2) @ local[1:]
    speed = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)).mean(-1) \
        * streamer.fps
    hand = out["posed_joints"][0, start:, names.index("RightHand"), 1].numpy()
    window = 2 * streamer.fps
    per2 = [round(float(speed[s:s + window].mean()), 2) for s in range(start, len(speed), window)]
    return per2, float(hand.max())


def main() -> None:
    model = probe.load("onnx-dml", "fp16")
    features = probe.encode([STAND, *PROMPTS])
    fps = int(model.motion_rep.fps)
    for history, keep in SETTINGS:
        for i, prompt in enumerate(PROMPTS, 1):
            probe.seed_everything(0)
            streamer = Streamer(model, history)
            while streamer.motion is None or streamer.motion.shape[1] < 10 * fps:
                streamer.step(features[0:1], None, keep)
            start = streamer.motion.shape[1]
            while streamer.motion.shape[1] < start + 10 * fps:
                streamer.step(features[i:i + 1], None, keep)
            per2, hand = arm(streamer, start)
            print(f"history {history:3}, keep {keep:2}: {prompt[:45]:45} arm rad/s per 2 s "
                  f"{per2}; hand max {hand:.2f} m", flush=True)


if __name__ == "__main__":
    main()
