"""Sanity checks for one generated chunk (2026-10-30).

ARDY's stream drifts when it runs open-loop for long: during a 2-minute idle every variant tried
(prompts, history 4 to 16, 4 or 10 steps, fp16 or fp32, three random seeds) came apart within 30
to 60 s, with joints outside any recorded range, the hips floating up to 2.4 m or collapsing to
0.1 m, a spin and limb pops (``stress_service.py``, ``idle_test.py``). The drift starts as small
random excursions that then feed the next replan's history, so a chunk that fails these checks
is drawn again from the same history (``service.MotionService.next_chunk``).

:func:`badness` is 0 for a sound chunk and grows with how far it is off:

- **joints outside the training data's limits** (motion-lab's ``joint_limits.json``) in more
  than 5% of joint-frames (the recordings: 0.5%);
- **hips height** outside 0.70 to 1.02 m for intents where she is on her feet (standing,
  walking, running, gesturing: her rest is 0.89 m, running dips to about 0.85);
- **the head below the hips** for those intents;
- **a pop:** a joint turning more than 60° between two 20 fps frames (her own motion, and from
  the last frame already sent).
"""

from __future__ import annotations

import numpy as np
import torch

from motionlab import m1  # motion-lab v4's package, which stream.py puts on the path

UPRIGHT = {"-", "idle", "walk", "run", "wave", "point", "look", "present", "stand"}
HIPS_BAND = (0.70, 1.02)
OUTSIDE_ALLOWED = 0.05
POP_DEG = 60.0
HEAD, HIPS = 5, 0
_LIMITS = None


def _limits():
    global _LIMITS
    if _LIMITS is None:
        _LIMITS = m1.joint_limits().without_hips()
    return _LIMITS


def _heights(local: np.ndarray, root: np.ndarray) -> np.ndarray:
    """Head height above the hips, per frame, from canonical local rotations (FK on the
    canonical skeleton)."""
    from motionlab import fk, skeleton

    _, positions = fk.forward_kinematics(torch.tensor(local, dtype=torch.float32),
                                         torch.tensor(root, dtype=torch.float32),
                                         skeleton.load())
    return (positions[:, HEAD, 1] - positions[:, HIPS, 1]).numpy()


def badness(local: np.ndarray, root: np.ndarray, key: str,
            last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> float:
    """``local (T, 22, 3, 3)`` and ``root (T, 3)``, canonical, at ARDY's 20 fps; ``key`` the
    phrase key it was asked for; ``last`` the frame already sent before it."""
    intent = key.split("|")[0] or "-"
    score = 0.0
    r = torch.tensor(local[:, 1:], dtype=torch.float32)
    outside = float((m1.limit_excess(r, _limits()) > 0).float().mean())
    score += max(0.0, outside - OUTSIDE_ALLOWED) * 10
    if intent in UPRIGHT:
        hips = root[:, 1]
        score += float(np.clip(HIPS_BAND[0] - hips, 0, None).max()
                       + np.clip(hips - HIPS_BAND[1], 0, None).max()) * 10
        score += float(np.clip(-_heights(local, root), 0, None).max()) * 10
    frames = local if last is None else np.concatenate((last[0][-1:], local))
    rel = np.swapaxes(frames[:-1], -1, -2) @ frames[1:]
    turn = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    score += float(np.clip(turn[:, 1:] - POP_DEG, 0, None).max()) / 30
    return round(score, 4)
