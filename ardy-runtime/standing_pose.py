"""Write ``assets/standing_pose.npz``: one natural standing frame in ARDY's own terms (2026-10-30).

    .venv\\Scripts\\python standing_pose.py

The service's get-up target before it has seen her standing (a stream seeded while she lies on
the ground). ARDY generates 3 s of "a person stands naturally calmly" from no history, and the
last frame is kept as ``posed`` (27 joint positions, the root's ground position at the origin)
and ``rots`` (27 global rotations), as ``MotionService._current_pose`` returns them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import service  # noqa: E402

OUT = HERE / "assets" / "standing_pose.npz"


def main() -> None:
    torch.manual_seed(0)
    svc = service.MotionService("fp32", 10, anchoring=False)
    for _ in range(8):  # 3.2 s
        svc.signal({"key": "-|calm|-", "velocity": [0.0, 0.0]})
        chunk = svc.next_chunk()
    hips = chunk["frames"][-1]["root"][1]
    posed, rots = svc._current_pose()
    np.savez(OUT, posed=posed.numpy(), rots=rots.numpy())
    print(f"standing pose: hips {hips:.3f} m (canonical); wrote {OUT}", file=sys.__stdout__)


if __name__ == "__main__":
    main()
