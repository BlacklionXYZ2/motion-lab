"""Does a root path keep ARDY walking and running for a whole minute? (2026-10-26)

    .venv\\Scripts\\python path_test.py

Held text alone lets locomotion stop (``history_test.py``). NVIDIA's interactive demo instead
keeps a 2D root path ahead of the character (``scripts/interactive_demo/gen_constraints.py``,
target velocity → root waypoints). This lays a straight path along +Z at a set speed, one
waypoint a second (``Root2DConstraintSet``), for 60 s, and reports the achieved speed per 10 s,
foot slide and the path error, with NVIDIA's 8 s history and with 2 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe  # noqa: E402
from ardy.constraints import Root2DConstraintSet  # noqa: E402

CASES = [("a person walks forward at a normal pace", 1.2),
         ("a person runs forward", 2.5),
         ("a person walks forward happily with a bounce in their step", 1.2)]


def generate_on_path(model, feature, frames: int, speed: float, history: int) -> dict:
    fps = int(model.motion_rep.fps)
    marks = torch.arange(fps, frames, fps)
    root_2d = torch.stack((torch.zeros(len(marks)), marks.float() / fps * speed), -1)
    path = Root2DConstraintSet(model.skeleton, marks, root_2d)
    lengths = torch.tensor([frames])
    observed, mask = model.motion_rep.create_conditions_from_constraints_batched(
        [path], lengths, to_normalize=True, device="cpu")
    with torch.no_grad():
        motion = model(None, frames, num_denoising_steps=10,
                       pad_mask=torch.ones(1, frames, dtype=torch.bool),
                       first_heading_angle=torch.zeros(1), motion_mask=mask,
                       observed_motion=observed, cfg_weight=(2.0, 2.0), text_feat=feature,
                       text_pad_mask=torch.ones(1, 1, dtype=torch.bool),
                       crop_history_length=history, progress_bar=lambda x: x)
        out = model.motion_rep.inverse(motion, is_normalized=True)
    out["path_marks"], out["path_2d"] = marks, root_2d
    return out


def main() -> None:
    model = probe.load("onnx-dml", "fp16")
    features = probe.encode([prompt for prompt, _ in CASES])
    fps = int(model.motion_rep.fps)
    frames = 60 * fps
    for i, (prompt, speed) in enumerate(CASES):
        for history in (160, 40):
            probe.seed_everything(0)
            out = generate_on_path(model, features[i:i + 1], frames, speed, history)
            root = out["root_positions"][0].numpy()
            v = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1) * fps
            per10 = [round(float(v[s:s + 10 * fps].mean()), 2) for s in range(0, len(v), 10 * fps)]
            at = root[out["path_marks"].numpy()][:, [0, 2]]
            error = np.linalg.norm(at - out["path_2d"].numpy(), axis=1)
            print(f"{prompt[:40]:40} {speed} m/s, history {history:3}: achieved per 10 s {per10}; "
                  f"path error median {np.median(error):.2f} m, max {error.max():.2f} m",
                  flush=True)


if __name__ == "__main__":
    main()
