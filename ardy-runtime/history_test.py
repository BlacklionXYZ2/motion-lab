"""Does locomotion fade because of the history length or the prompt? (2026-10-26)

    .venv\\Scripts\\python history_test.py

Walks and runs for 60 s with NVIDIA's default 8 s history and with 2 s, two seeds each, and prints
root speed per 10 s. Also tries a prompt that describes ongoing motion ("keeps walking").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe  # noqa: E402

PROMPTS = ["a person walks forward at a normal pace",
           "a person runs forward",
           "a person keeps walking forward at a steady pace without stopping"]


def main() -> None:
    model = probe.load("onnx-dml", "fp16")
    features = probe.encode(PROMPTS)
    fps = int(model.motion_rep.fps)
    for i, prompt in enumerate(PROMPTS):
        for history in (160, 40):
            for seed in (0, 1):
                probe.seed_everything(seed)
                out = probe.generate(model, features[i:i + 1], 60 * fps, 10, history=history)
                root = out["root_positions"][0].numpy()
                speed = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1) * fps
                per10 = [round(float(speed[s:s + 10 * fps].mean()), 2)
                         for s in range(0, len(speed), 10 * fps)]
                print(f"{prompt[:45]:45} history {history:3} seed {seed}: m/s per 10 s {per10}",
                      flush=True)


if __name__ == "__main__":
    main()
