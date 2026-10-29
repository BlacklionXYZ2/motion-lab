"""Make a preview clip of one trained version for SORA's Motion window.

    .venv\\Scripts\\python tools\\version_preview.py v4 [--seconds 20] [--seed 0]

Imports that version's own ``motionlab`` (nothing in the version folder changes), loads
``runs/combined/best.pt``, seeds a free rollout with recorded test frames, lets the model run on
its own, and writes ``catalog/previews/<version>-<size>.json``: joint positions (millimetres) at
10 fps, with joint names and parents, for the window's stick-figure preview. v4 and later walk
with the "walk" intent; earlier versions had no keywords.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
PREVIEW_FPS = 10


def size_label(parameters: int) -> str:
    return f"{parameters / 1e6:.1f}M" if parameters < 1e9 else f"{parameters / 1e9:.1f}B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("version")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", default="combined")
    args = parser.parse_args()
    root = LAB / "versions" / args.version
    sys.path.insert(0, str(root))
    from motionlab import fk, m1, m1_eval, skeleton
    from motionlab.datasets import m1_data
    from motionlab.formats import FPS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(root / "runs" / args.run / "best.pt", map_location=device,
                            weights_only=False)
    model = m1.MotionPrior(m1.ModelConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    normaliser = m1_data.Normaliser(checkpoint["normaliser"]["mean"].to(device),
                                    checkpoint["normaliser"]["std"].to(device))
    config = checkpoint["config"]
    sources = config["data"] if isinstance(config["data"], list) else [config["data"]]
    options = {name: config.get(key, False) for name, key in
               (("clean", "clean_data"), ("support_frame", "support_frame"))
               if name in inspect.signature(m1_data.load_clips).parameters}
    clips = m1_data.load_clips([root / folder for folder in sources], "test", **options)
    sampler = m1_data.WindowSampler(clips, device, storage=config.get("storage"))
    context = 60
    seed = sampler.sample(1, context + 2, torch.Generator().manual_seed(args.seed), augment=False)
    extra = {}
    label = "free run, no keywords"
    if "signal" in inspect.signature(m1_eval.free_rollout).parameters:
        from motionlab.motion_signal import MotionSignal

        extra["signal"] = MotionSignal(intent="walk")
        label = "free run, intent walk"
    with torch.no_grad():
        motion = m1_eval.free_rollout(model, normaliser, seed, context,
                                      int(args.seconds * FPS), **extra)
        _, positions = fk.forward_kinematics(motion.rotations, motion.root, skeleton.load())
    positions = positions[0, context::FPS // PREVIEW_FPS].cpu()
    parameters = model.parameter_count()
    out = LAB / "catalog" / "previews" / f"{args.version}-{size_label(parameters)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": f"{args.version} {size_label(parameters)}: {label}, {args.seconds:g} s",
        "fps": PREVIEW_FPS,
        "joint_names": list(skeleton.NAMES),
        "joint_parents": list(skeleton.PARENTS),
        "frames": [[round(float(c) * 1000) for c in frame.flatten()] for frame in positions],
    }), encoding="utf-8")
    print(f"{out.relative_to(LAB)}: {positions.shape[0]} frames, {parameters:,} parameters")


if __name__ == "__main__":
    main()
