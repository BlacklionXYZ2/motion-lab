"""Streaming ARDY the way SORA would drive it: one 2 s step at a time, text plus a target velocity.

    .venv\\Scripts\\python stream.py [--precision fp16|fp32] [--history 160] [--seed 0]

Mirrors NVIDIA's interactive demo (``scripts/interactive_demo/generation.py`` and
``gen_constraints.py``):
- each step takes up to ``--history`` frames of the motion so far as history;
- it lays 2D root waypoints every 10 frames along a path that eases from her current velocity to
  the target over 2 s;
- it generates the 40-frame horizon and appends it.

Text changes between steps take effect at the next step. With no target velocity, there are no
root waypoints.

The script plays a fixed 60 s scenario:
- walk for 20 s at 1.2 m/s;
- stand still for 10 s;
- wave for 10 s;
- run for 20 s at 2.5 m/s while turning from +Z towards +X.

It reports each step's latency and the motion's speed per 2 s, its largest root jump, and our
measures on the canonical skeleton, and saves ``outputs/stream-<precision>-<seed>.npz``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "versions" / "v4"))

import probe  # noqa: E402
import retarget  # noqa: E402
from ardy.constraints import Root2DConstraintSet  # noqa: E402

from motionlab import m1_eval  # noqa: E402

WAYPOINT_EVERY = 10  # frames (the demo's TARGET_VELOCITY_GOAL_FRAME_INTERVAL)
EASE_S = 2.0  # seconds (the demo's VELOCITY_TRANSITION_DURATION)

# (start second, prompt, target velocity in m/s as (vx, vz) or None)
SCENARIO = [
    (0, "a person walks forward at a normal pace", lambda t: (0.0, 1.2)),
    (20, "a person stands still, relaxed", None),
    (30, "a person waves with their right hand while standing", None),
    (40, "a person runs forward",
     lambda t: (2.5 * math.sin(min(1.0, (t - 40) / 10) * math.pi / 2),
                2.5 * math.cos(min(1.0, (t - 40) / 10) * math.pi / 2))),
]


def per2_energy(local: np.ndarray, fps: int) -> list[float]:
    """Mean joint angular speed (rad/s) per 2 s, from canonical local rotations."""
    rel = np.swapaxes(local[:-1], -1, -2) @ local[1:]
    angle = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)) * fps
    return [float(angle[s:s + 2 * fps].mean()) for s in range(0, len(angle), 2 * fps)]


class Streamer:
    def __init__(self, model, history: int, steps: int = 10, cfg=(2.0, 2.0)) -> None:
        self.model, self.history, self.steps, self.cfg = model, history, steps, cfg
        self.rep = model.motion_rep
        self.fps = int(self.rep.fps)
        self.patch = model.num_frames_per_token
        self.horizon = model.gen_horizon_len
        self.motion: torch.Tensor | None = None  # normalised explicit motion (1, T, 330)
        # The commanded velocity eases to a new target over EASE_S of *played* time, across
        # replans. Restarting the ease from her measured velocity at every replan (the first
        # version) only shed about a fifth of her speed per 0.4 s replan, so she crept on at
        # 0.15 to 0.26 m/s seconds after being told to stop (test_service.py, 2026-10-27).
        self._ramp_from: torch.Tensor | None = None
        self._ramp_goal: torch.Tensor | None = None
        self._ramp_done = 0  # frames of the current ease already played

    def _commanded(self, i: int) -> torch.Tensor:
        """The commanded velocity ``i`` frames after the last played frame."""
        assert self._ramp_from is not None and self._ramp_goal is not None
        alpha = min(1.0, (self._ramp_done + i) / int(EASE_S * self.fps))
        return (1 - alpha) * self._ramp_from + alpha * self._ramp_goal

    def _root_now(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.motion is not None
        out = self.rep.inverse(self.motion[:, -2:], is_normalized=True)
        root = out["root_positions"][0]
        return root[-1], (root[-1] - root[-2]) * self.fps

    def _path(self, target: tuple[float, float], frame0: int, length: int):
        """Root waypoints for frames ``frame0 + i`` (window-relative), easing to the target
        velocity over ``EASE_S``."""
        position, velocity = self._root_now() if self.motion is not None else \
            (torch.zeros(3), torch.zeros(3))
        goal = torch.tensor([target[0], 0.0, target[1]])
        if self._ramp_goal is None:  # first path: start from how she is actually moving
            self._ramp_from, self._ramp_goal, self._ramp_done = velocity * torch.tensor(
                [1.0, 0.0, 1.0]), goal, 0
        elif not torch.allclose(goal, self._ramp_goal):  # a new target: ease from the command
            self._ramp_from, self._ramp_goal, self._ramp_done = self._commanded(0), goal, 0
        points, marks, p = [], [], position.clone()
        for i in range(1, length + 1):
            p = p + self._commanded(i) / self.fps
            if i % WAYPOINT_EVERY == 0:
                marks.append(frame0 + i - 1)
                points.append(p[[0, 2]].clone())
        return Root2DConstraintSet(self.rep.skeleton, torch.tensor(marks), torch.stack(points))

    def step(self, text_feat: torch.Tensor, target: tuple[float, float] | None,
             keep: int | None = None) -> None:
        """Generate the next horizon and append its first ``keep`` frames (all by default).
        Keeping fewer replans sooner, as the demo does when the input changes."""
        if self.motion is None:
            history = None
            hist_len = 0
        else:
            hist_len = min(self.history, self.motion.shape[1]) // self.patch * self.patch
            history = self.motion[:, -hist_len:]
        frames = hist_len + self.horizon
        mask = observed = None
        if target is not None:
            path = self._path(target, hist_len, self.horizon)
            observed, mask = self.rep.create_conditions_from_constraints(
                [path], length=frames, to_normalize=False, device="cpu")
            observed = self.rep.normalize(observed) * mask
            observed, mask = observed[None], mask[None]
            mask[:, :hist_len] = 0.0
            observed[:, :hist_len] = 0.0
        with torch.no_grad():
            samples = self.model.autoregressive_step(
                num_frames=frames, num_denoising_steps=self.steps, motion_mask=mask,
                observed_motion=observed, cfg_weight=self.cfg, texts=None, text_feat=text_feat,
                text_pad_mask=torch.ones(1, 1, dtype=torch.bool), init_history_sequence=history,
                init_global_translation=None if history is not None else torch.zeros(1, 3),
                init_first_heading_angle=None if history is not None else torch.zeros(1))
        new = samples[:, hist_len:] if keep is None else samples[:, hist_len:hist_len + keep]
        self.motion = new if self.motion is None else torch.cat((self.motion, new), 1)
        if target is None:
            self._ramp_from = self._ramp_goal = None  # no command: the next starts from her
        else:
            self._ramp_done += new.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--history", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg-text", type=float, default=2.0, help="text guidance weight")
    parser.add_argument("--keep", type=int, default=40,
                        help="frames played from each generation before replanning "
                             "(NVIDIA's demo replans within 4 frames of the end; 8 ≈ 0.4 s)")
    args = parser.parse_args()
    model = probe.load("onnx-dml", args.precision)
    prompts = [p for _, p, _ in SCENARIO]
    features = probe.encode(prompts)
    probe.seed_everything(args.seed)
    streamer = Streamer(model, args.history, args.steps, cfg=(args.cfg_text, 2.0))
    latencies = []
    t = 0.0
    while t < 60:
        phase = max(i for i, (start, _, _) in enumerate(SCENARIO) if t >= start)
        _, _, velocity = SCENARIO[phase]
        started = time.perf_counter()
        streamer.step(features[phase:phase + 1], velocity(t) if velocity else None, args.keep)
        latencies.append(time.perf_counter() - started)
        t += args.keep / streamer.fps
    out = streamer.rep.inverse(streamer.motion, is_normalized=True)
    root = out["root_positions"][0].numpy()
    fps = streamer.fps
    jump = np.linalg.norm(np.diff(root, axis=0), axis=1)
    v = jump * fps
    per2 = [round(float(v[s:s + 2 * fps].mean()), 2) for s in range(0, len(v), 2 * fps)]
    g, c = out["global_rot_mats"][0].numpy(), out["foot_contacts"][0].numpy()
    local, r, contact = retarget.to_canonical(g, root, c, streamer.rep.skeleton.bone_order_names,
                                              streamer.rep.skeleton.foot_joint_names)
    local, r, contact = retarget.resample(local, r, contact, fps)
    measured = m1_eval.measures(m1_eval.Motion(torch.tensor(local[None], dtype=torch.float32),
                                               torch.tensor(r[None], dtype=torch.float32),
                                               torch.tensor(contact[None])))
    budget = args.keep / streamer.fps
    late = sum(lat > budget for lat in latencies)
    print(f"latency per step: median {np.median(latencies) * 1000:.0f} ms, "
          f"max {max(latencies) * 1000:.0f} ms (first {latencies[0] * 1000:.0f} ms); "
          f"real-time budget {budget * 1000:.0f} ms, missed by {late} of {len(latencies)} steps")
    print(f"speed per 2 s (walk 0-20, stand 20-30, wave 30-40, run 40-60): {per2}")
    print(f"largest root step {jump.max() * 100:.1f} cm in one frame "
          f"({jump.max() * fps:.1f} m/s); final heading position {root[-1].round(2)}")
    print({k: measured[k] for k in ("outside_data_limits_share", "planted_foot_slide_m_s",
                                    "jitter_m_s3", "hips_height_m", "energy_by_10s_rad_s")})
    print(f"energy per 2 s: {[round(float(x), 2) for x in per2_energy(local, 60)]}")
    tag = (f"stream-{args.precision}-s{args.steps}-h{args.history}-k{args.keep}"
           f"-cfg{args.cfg_text:g}-seed{args.seed}")
    np.savez(HERE / "outputs" / f"{tag}.npz", fps=fps,
             joint_names=np.asarray(streamer.rep.skeleton.bone_order_names),
             joint_parents=np.asarray(streamer.rep.skeleton.joint_parents),
             **{k: v.numpy() for k, v in out.items() if isinstance(v, torch.Tensor)})
    walk = v[2 * fps:20 * fps]  # the walking phase, after its first 2 s of speeding up
    summary = {
        "settings": {"precision": args.precision, "steps": args.steps, "history": args.history,
                     "keep": args.keep, "cfg_text": args.cfg_text, "seed": args.seed},
        "step_latency_ms": {"median": round(float(np.median(latencies)) * 1000),
                            "max": round(max(latencies) * 1000)},
        "steps_late": int(late), "steps": len(latencies),
        "walk_speed_m_s": round(float(walk.mean()), 3),
        "outside_limits_share": measured["outside_data_limits_share"],
        "foot_slide_m_s": measured["planted_foot_slide_m_s"],
        "jitter_m_s3": measured["jitter_m_s3"],
        "energy_rad_s": round(float(np.mean(measured["energy_by_10s_rad_s"])), 3),
        "speed_per_2s": per2,
    }
    (HERE / "outputs" / f"{tag}.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(f"saved outputs/{tag}.npz and .json")


if __name__ == "__main__":
    main()
