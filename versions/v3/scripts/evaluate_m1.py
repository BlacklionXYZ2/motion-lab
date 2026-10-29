"""Evaluate an M1 checkpoint: long free rollouts on the test subject, against the recordings.

    .venv\\Scripts\\python scripts\\evaluate_m1.py runs\\<name>\\best.pt [--seconds 60] [--seeds 16]

Writes ``outputs/m1/eval-<run>.json`` and prints it. What each measure means:
``motionlab/m1_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import m1, m1_eval  # noqa: E402
from motionlab.datasets import m1_data  # noqa: E402
from motionlab.formats import FPS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--context", type=int, default=60, help="recorded ticks before it runs")
    parser.add_argument("--split", default="test", choices=sorted(m1_data.SPLITS))
    parser.add_argument("--clean", action="store_true",
                        help="test on quality-filtered clips even if the model trained without "
                             "(to compare an earlier version's model on v3's test set)")
    parser.add_argument("--out", type=Path, help="report path (default outputs/eval-<run>.json)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = m1.MotionPrior(m1.ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normaliser = m1_data.Normaliser(checkpoint["normaliser"]["mean"].to(device),
                                    checkpoint["normaliser"]["std"].to(device))
    config = checkpoint["config"]
    sources = config["data"] if isinstance(config["data"], list) else [config["data"]]
    sampler = m1_data.WindowSampler(
        m1_data.load_clips([ROOT / folder for folder in sources], args.split,
                           clean=args.clean or config.get("clean_data", False)), device,
        storage=config.get("storage"), source_weights=config.get("source_weights"))
    ticks = int(args.seconds * FPS)
    generator = torch.Generator().manual_seed(7)

    seeds = m1_eval.seed_windows(sampler, args.seeds, args.context + 2 * FPS + 1, generator)
    rollout = m1_eval.free_rollout(model, normaliser, seeds, args.context, ticks)
    free = m1_eval.Motion(rollout.rotations[:, args.context:], rollout.root[:, args.context:],
                          rollout.contacts[:, args.context:])
    reference = m1_eval.recorded_motion(sampler, args.seeds, min(ticks, 20 * FPS), generator)
    report = {
        "checkpoint": str(args.checkpoint.relative_to(ROOT) if args.checkpoint.is_absolute()
                          and ROOT in args.checkpoint.parents else args.checkpoint),
        "step": checkpoint["step"], "split": args.split, "seeds": args.seeds,
        "seconds": args.seconds, "context_ticks": args.context,
        "parameters": model.parameter_count(), "head": model.config.head,
        "model": m1_eval.measures(free),
        "recordings": m1_eval.measures(reference),
        "short_horizon": m1_eval.short_horizon(model, normaliser, seeds, args.context),
    }
    out = args.out or ROOT / "outputs" / f"eval-{args.checkpoint.parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
