"""Compare two ``probe.py`` outputs made with the same prompt and seed (e.g. NVIDIA's PyTorch
model against the ONNX conversion, or fp16 against fp32).

    .venv\\Scripts\\python compare.py outputs\\a.npz outputs\\b.npz

Reports per 2 s window: mean and largest joint rotation difference (degrees, local rotations)
and root position difference (cm). Diffusion feeds each window's output into the next, so a
difference that grows window by window means the backends drift apart over time.
"""

from __future__ import annotations

import sys

import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(a, -1, -2) @ b
    cos = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def main() -> None:
    a, b = (np.load(p) for p in sys.argv[1:3])
    angles = angle_deg(a["local_rot_mats"][0], b["local_rot_mats"][0])  # (T, J)
    root = np.linalg.norm(a["root_positions"][0] - b["root_positions"][0], axis=-1) * 100
    window = 40
    for start in range(0, angles.shape[0], window):
        s = slice(start, start + window)
        print(f"{start / 20:5.0f} s: rotation mean {angles[s].mean():6.3f}°, "
              f"max {angles[s].max():7.3f}°; root mean {root[s].mean():6.2f} cm, "
              f"max {root[s].max():6.2f} cm")
    print(f"overall: rotation mean {angles.mean():.3f}°, max {angles.max():.3f}°; "
          f"root max {root.max():.2f} cm")


if __name__ == "__main__":
    main()
