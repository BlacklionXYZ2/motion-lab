"""How should a latent model sample at run time? (v5, 2026-10-30)

    ..\\..\\.venv\\Scripts\\python scripts\\sampling_sweep.py CHECKPOINT [--seconds 60]

For each prior temperature and noise correlation (``MotionPrior.temperature``,
``noise_correlation``), a free run from the same 16 test seeds with every keyword unknown,
measured as the 60 s evaluation does (``m1_eval.measures``): jitter, energy, foot slide, limits,
hips. Writes ``outputs/sampling-sweep-<run>.json`` and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionlab import m1_eval  # noqa: E402
from motionlab.datasets import m1_data  # noqa: E402
from motionlab.formats import FPS  # noqa: E402
from scripts.stress_m1 import load  # noqa: E402

SETTINGS = [(1.0, 0.0), (1.0, 0.9), (1.0, 0.98), (0.5, 0.0), (0.5, 0.9), (0.5, 0.98),
            (0.25, 0.95), (0.0, 0.0)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    torch.set_num_threads(8)
    model, norm, ckpt = load(args.checkpoint, torch.device("cpu"))
    config = ckpt["config"]
    sampler = m1_data.WindowSampler(
        m1_data.load_clips([ROOT / f for f in config["data"]], "test",
                           clean=config.get("clean_data", False),
                           support_frame=config.get("support_frame", False)), "cpu")
    seed = sampler.sample(16, 61, torch.Generator().manual_seed(7), augment=False)
    reference = m1_eval.measures(m1_eval.recorded_motion(sampler, 16, 20 * FPS,
                                                         torch.Generator().manual_seed(8)))
    report = {"recordings": reference}
    keys = ("jitter_m_s3", "energy_by_10s_rad_s", "planted_foot_slide_m_s",
            "outside_data_limits_share", "hips_height_m", "root_speed_m_s")
    print("temperature, correlation:", {k: reference[k] for k in keys})
    for temperature, rho in SETTINGS:
        model.temperature, model.noise_correlation = temperature, rho
        model.reset_noise()
        torch.manual_seed(0)
        motion = m1_eval.free_rollout(model, norm, seed, 60, int(args.seconds * FPS))
        measured = m1_eval.measures(m1_eval.Motion(motion.rotations[:, 60:],
                                                   motion.root[:, 60:], motion.contacts[:, 60:]))
        report[f"{temperature}, {rho}"] = measured
        print(f"{temperature}, {rho}:", {k: measured[k] for k in keys}, flush=True)
    out = ROOT / "outputs" / f"sampling-sweep-{args.checkpoint.parent.name}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
