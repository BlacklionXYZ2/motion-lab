"""Evaluate ARDY on motion-lab's own 60 s measures, beside our trained versions.

    .venv\\Scripts\\python evaluate.py [--seconds 60] [--seeds 4] [--precision fp16]
        [--backend onnx-dml] [--steps 10] [--prompts walk,idle,...] [--calibrate]

For each prompt (``PROMPTS``) and seed: generate with ARDY (``probe.py``'s path: NVIDIA's code,
the ONNX networks, a precomputed text feature, NVIDIA's default 8 s history), retarget onto the
canonical skeleton at 60 Hz (``retarget.py``), and score it with motion-lab v4's
``m1_eval.measures``:
- limits outside the data-measured ranges;
- energy over time;
- foot slide, jitter;
- hips height;
- root speed.

The raw output is scored: NVIDIA's foot-skate post-processing needs its C++ extension, which
isn't built here.

``--calibrate`` measures ARDY's standing hips height from "stands still" generations, for
``retarget.ARDY_STANDING_HIPS``. Writes ``outputs/eval-<backend>-<precision>-<steps>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "versions" / "v4"))  # motion-lab's measures

import probe  # noqa: E402
import retarget  # noqa: E402

from motionlab import m1_eval  # noqa: E402

# Short names → prompts in plain, physical wording (ARDY's model card advises neutral, physical
# descriptions). The first six mirror v4's keyword conditions.
PROMPTS = {
    "idle": "a person stands still, relaxed, shifting their weight slightly",
    "walk": "a person walks forward at a normal pace",
    "run": "a person runs forward",
    "sit": "a person sits down on a chair and stays seated",
    "wave": "a person waves with their right hand while standing",
    "look": "a person stands and looks around",
    "walk-cheerful": "a person walks forward happily with a bounce in their step",
    "walk-sad": "a person walks forward sadly with their head down",
    "walk-angry": "a person walks forward angrily",
    "walk-proud": "a person walks forward proudly and confidently",
    "talk": "a person stands and talks, gesturing with their hands",
    "stand-still": "a person stands still",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--backend", default="onnx-dml")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    names = ["stand-still"] if args.calibrate else args.prompts.split(",")
    model = probe.load(args.backend, args.precision)
    features = probe.encode([PROMPTS[n] for n in names])
    skeleton = model.skeleton
    fps = int(model.motion_rep.fps)
    frames = int(args.seconds * fps) // 4 * 4
    report: dict[str, object] = {"backend": args.backend, "precision": args.precision,
                                 "steps": args.steps, "seconds": args.seconds,
                                 "seeds": args.seeds,
                                 "history_frames": probe.default_history(model),
                                 "prompts": {n: PROMPTS[n] for n in names}, "results": {}}
    for i, name in enumerate(names):
        rotations, roots, contacts, hips, timings = [], [], [], [], []
        for seed in range(args.seeds):
            probe.seed_everything(seed)
            started = time.time()
            out = probe.generate(model, features[i:i + 1], frames, args.steps)
            timings.append(time.time() - started)
            g, r, c = (out[k][0].numpy() for k in ("global_rot_mats", "root_positions",
                                                    "foot_contacts"))
            hips.append(float(np.median(r[:, 1])))
            local, root, contact = retarget.to_canonical(g, r, c, skeleton.bone_order_names,
                                                         skeleton.foot_joint_names)
            local, root, contact = retarget.resample(local, root, contact, fps)
            rotations.append(local)
            roots.append(root)
            contacts.append(contact)
        if args.calibrate:
            print(f"ARDY standing hips: median {np.median(hips):.3f} m over {args.seeds} seeds "
                  f"({[round(h, 3) for h in hips]})")
            return
        length = min(len(r) for r in rotations)
        motion = m1_eval.Motion(
            torch.tensor(np.stack([r[:length] for r in rotations]), dtype=torch.float32),
            torch.tensor(np.stack([r[:length] for r in roots]), dtype=torch.float32),
            torch.tensor(np.stack([c[:length] for c in contacts])))
        measured = m1_eval.measures(motion)
        measured["seconds_to_generate"] = round(float(np.mean(timings)), 1)
        measured["real_time_factor"] = round(args.seconds / float(np.mean(timings)), 2)
        report["results"][name] = measured  # type: ignore[index]
        print(f"{name:14} speed {measured['root_speed_m_s']} m/s, energy "
              f"{measured['energy_by_10s_rad_s']}, "
              f"outside {measured['outside_data_limits_share']}, "
              f"slide {measured['planted_foot_slide_m_s']}, jitter {measured['jitter_m_s3']}, "
              f"hips {measured['hips_height_m']}, {measured['real_time_factor']}x real time",
              flush=True)
    out_path = HERE / "outputs" / f"eval-{args.backend}-{args.precision}-{args.steps}.json"
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
