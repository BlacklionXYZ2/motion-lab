"""Does the GRU's memory drift in long runs? (v5, 2026-10-30)

    ..\\..\\.venv\\Scripts\\python scripts\\hidden_drift.py CHECKPOINT [--seconds 300] [--reset S]

Training windows are at most 2.8 s long, but SORA runs the model for minutes. This runs the
stress battery's long free run and reports, per 10 s: the hidden state's mean size, the share of
units near saturation (|h| > 0.9), and the joint energy, so the two can be compared. With
``--reset S`` the memory is rebuilt every S seconds from a short burn-in over the newest frames
(two staggered memories, each restarted from zero and given half a period to warm up), which
keeps it inside what training saw. Prints a table and writes ``outputs/hidden-drift-<run>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionlab import m1, motion_signal, rotation, stress  # noqa: E402
from motionlab.datasets import m1_data  # noqa: E402
from motionlab.formats import FPS  # noqa: E402
from scripts.stress_m1 import load  # noqa: E402


@torch.no_grad()
def run(model, norm, seed, context: int, ticks: int, signal, reset: int = 0):
    b = seed.rot6d.shape[0]
    features = motion_signal.features(signal).expand(b, -1)
    rots, roots = [seed.rotations[:, 0], seed.rotations[:, 1]], [seed.root[:, 0], seed.root[:, 1]]
    h = model.initial_state(b)
    spare = model.initial_state(b)  # the staggered second memory (reset mode)
    sizes, saturated = [], []
    scale = torch.tensor(1.0)
    for t in range(1, context + ticks - 1):
        pair = rotation.matrix_to_rot6d(torch.stack((rots[t - 1], rots[t]), 1))
        state = norm(m1_data.states(pair, torch.stack((roots[t - 1], roots[t]), 1))[:, 0])
        raw, h = model(state, features, h)
        if reset:
            _, spare = model(state, features, spare)
            if t >= context and (t - context) % reset == reset // 2:
                h, spare = spare, model.initial_state(b)  # the warmed-up one takes over
        step = m1.advance(model, raw, rots[t], roots[t], scale)
        if t + 1 < context:
            rots.append(seed.rotations[:, t + 1])
            roots.append(seed.root[:, t + 1])
        else:
            rots.append(step.rotations)
            roots.append(step.root)
            sizes.append(float(h.abs().mean()))
            saturated.append(float((h.abs() > 0.9).float().mean()))
    r = torch.stack(rots, 1)[:, context:]
    root = torch.stack(roots, 1)[:, context:]
    return r, root, torch.tensor(sizes), torch.tensor(saturated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--reset", type=float, default=0, help="rebuild the memory every S s")
    parser.add_argument("--intent", default=None)
    parser.add_argument("--every", type=int, default=10, help="seconds per row")
    parser.add_argument("--show", type=int, default=3, help="print every Nth row")
    args = parser.parse_args()
    torch.set_num_threads(8)
    model, norm, ckpt = load(args.checkpoint, torch.device("cpu"))
    config = ckpt["config"]
    sampler = m1_data.WindowSampler(
        m1_data.load_clips([ROOT / f for f in config["data"]], "test",
                           clean=config.get("clean_data", False),
                           support_frame=config.get("support_frame", False)), "cpu")
    seed = sampler.sample(args.seeds, 61, torch.Generator().manual_seed(7), augment=False)
    signal = motion_signal.MotionSignal(intent=args.intent)
    r, root, size, sat = run(model, norm, seed, 60, int(args.seconds * FPS), signal,
                             int(args.reset * FPS))
    energy = stress.energy(r).mean(0)
    rows = []
    n = args.every * FPS
    for s in range(0, len(size) - n + 1, n):
        rows.append({"second": s // FPS, "h_mean_abs": round(float(size[s:s + n].mean()), 3),
                     "h_saturated": round(float(sat[s:s + n].mean()), 3),
                     "energy": round(float(energy[s:s + n].mean()), 3)})
    body = stress.body(r, root, -0.107)
    out = {"checkpoint": str(args.checkpoint), "reset_s": args.reset, "intent": args.intent,
           "by_10s": rows, "body": body,
           "outside_limits_share": round(stress.limits_share(r), 4),
           "frozen_seed_share_last_30s": round(float(
               (stress.energy(r)[:, -30 * FPS:].mean(-1) < 0.1).float().mean()), 3)}
    tag = f"{args.checkpoint.parent.name}-reset{args.reset:g}-{args.intent or 'unknown'}"
    (ROOT / "outputs" / f"hidden-drift-{tag}.json").write_text(json.dumps(out, indent=1),
                                                                "utf-8")
    for row in rows[::args.show]:
        print(row)
    print({k: out[k] for k in ("outside_limits_share", "frozen_seed_share_last_30s")}, body)


if __name__ == "__main__":
    main()
