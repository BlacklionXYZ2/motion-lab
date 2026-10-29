"""Is the text actually steering ARDY? (2026-10-26)

    .venv\\Scripts\\python text_check.py

From scratch (no history), 10 s per prompt, same seed: the right arm's mean angular speed per
2 s, on NVIDIA's PyTorch model and on the ONNX networks. Also prints the cosine similarity
between the prompts' text features. If waving and standing give the same arm motion, the text
isn't reaching the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe  # noqa: E402

PROMPTS = ["a person stands still, relaxed",
           "a person waves with their right hand while standing",
           "a person raises both arms above their head",
           "a person does jumping jacks"]
ARM = ("RightShoulder", "RightArm", "RightForeArm", "RightHand")


def arm(out, names) -> list[float]:
    local = out["local_rot_mats"][0].numpy()[:, [names.index(n) for n in ARM]]
    rel = np.swapaxes(local[:-1], -1, -2) @ local[1:]
    angle = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)).mean(-1) * 20
    hand_y = out["posed_joints"][0, :, names.index("RightHand"), 1].numpy()
    return [round(float(angle[s:s + 40].mean()), 2) for s in range(0, len(angle), 40)] + \
        [f"hand max height {hand_y.max():.2f} m"]


def main() -> None:
    features = probe.encode(PROMPTS)
    f = features[:, 0].numpy()
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    print("text cosine similarity:\n", np.round(f @ f.T, 3))
    for backend in ("torch-cpu", "onnx-dml"):
        model = probe.load(backend, "fp16")
        names = model.skeleton.bone_order_names
        for i, prompt in enumerate(PROMPTS):
            probe.seed_everything(0)
            out = probe.generate(model, features[i:i + 1], 200, 10)
            print(f"{backend:9} {prompt[:50]:50} right arm rad/s per 2 s {arm(out, names)}",
                  flush=True)
        probe.seed_everything(0)
        out = probe.generate(model, torch.zeros_like(features[0:1]), 200, 10)
        print(f"{backend:9} {'(empty text: zeros)':50} right arm rad/s per 2 s {arm(out, names)}")


if __name__ == "__main__":
    main()
