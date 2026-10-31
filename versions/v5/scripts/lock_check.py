"""Does the foot lock reproduce real travel? (v5, 2026-10-31)

    ..\\..\\.venv\\Scripts\\python scripts\\lock_check.py

Replays recorded test windows through ``m1.foot_locked`` tick by tick: each frame's root is
derived from the recorded poses alone (the recorded root is only used as the "model's" root,
which applies in flight), and compared with the recorded root. If the lock can't reproduce the
recordings' own travel, no model can track a command under it. Reports, per speed band, the
recorded and derived speeds and the share of frames the lock was engaged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionlab import m1, skeleton  # noqa: E402
from motionlab.datasets import m1_data  # noqa: E402
from motionlab.formats import DT  # noqa: E402


def main() -> None:
    sampler = m1_data.WindowSampler(
        m1_data.load_clips([ROOT / "data/processed" / d for d in ("lafan1", "100style", "cmu")],
                           "test", clean=True, support_frame=True), "cpu")
    batch = sampler.sample(256, 300, torch.Generator().manual_seed(3), augment=False)
    r, root = batch.rotations, batch.root
    derived = [root[:, 0]]
    air = None
    engaged = []
    floor = m1._floor_heights()
    feet = [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]
    with torch.no_grad():
        for t in range(1, r.shape[1]):
            # the recorded root step, applied from the derived root, stands in for the model's
            model_root = derived[-1] + (root[:, t] - root[:, t - 1])
            nxt, air = m1.foot_locked(r[:, t - 1], r[:, t], derived[-1], model_root, air)
            derived.append(nxt)
            lift = (m1.relative_positions(r[:, t - 1])[:, feet, 1] + derived[-2][:, 1:2]
                    - floor).amin(-1)
            engaged.append((lift < m1.LOCK_NONE_M).float())
    derived = torch.stack(derived, 1)
    engaged = torch.stack(engaged, 1)
    speed = lambda x: (x[:, 1:, [0, 2]] - x[:, :-1, [0, 2]]).norm(dim=-1) / DT  # noqa: E731
    rec, der = speed(root), speed(derived)
    print("lowest-foot lift at rest frames: median", round(float(
        (m1.relative_positions(r[:, :, :, :, :].flatten(0, 1))[:, feet, 1]
         + root.flatten(0, 1)[:, 1:2] - floor).amin(-1).median()), 3), "m")
    for lo, hi in ((0, 0.2), (0.2, 0.8), (0.8, 1.6), (1.6, 3.0), (3.0, 9.0)):
        m = (rec >= lo) & (rec < hi)
        if m.any():
            print(f"recorded {lo:.1f}-{hi:.1f} m/s: recorded {float(rec[m].mean()):.3f}, "
                  f"derived {float(der[m].mean()):.3f} m/s, lock engaged "
                  f"{float(engaged[m].mean()):.2f}, share {float(m.float().mean()):.3f}")
    print(f"fastest frame: recorded {float(rec.max()):.2f} m/s, derived {float(der.max()):.2f} "
          f"m/s; derived frames over 5 m/s {float((der > 5).float().mean()):.5f}, recorded "
          f"{float((rec > 5).float().mean()):.5f}")
    jump = (der - rec).abs()
    print(f"largest derived-minus-recorded speed in a frame: {float(jump.max()):.2f} m/s; "
          f"p99.9 {float(torch.quantile(jump.flatten()[::2], 0.999)):.2f}")
    drift = (derived[:, -1, [0, 2]] - root[:, -1, [0, 2]]).norm(dim=-1)
    travel = (root[:, -1, [0, 2]] - root[:, 0, [0, 2]]).norm(dim=-1)
    print(f"over 5 s: recorded travel {float(travel.mean()):.2f} m, derived ends "
          f"{float(drift.mean()):.2f} m from the recorded end")


if __name__ == "__main__":
    main()
