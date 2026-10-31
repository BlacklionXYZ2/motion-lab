"""M1 evaluation: long free rollouts against the recordings (roadmap M1 gate, first version).

From each seed window (``context`` recorded ticks of a test-subject clip) the model runs on its
own, with the neutral ``MotionSignal``, for ``seconds``. Every measure is also taken on the
recordings themselves, so each number has a reference:

- **stable:** no NaN or Inf anywhere;
- **beyond limits:** share of joint-frames rotated further from rest than the placeholder
  maximum (``skeleton``), kept so v1 and v2 compare;
- **outside data limits** (v3): share of joint-frames outside the limits measured from the
  training data (``m1.joint_limits``), overall and per joint;
- **ω saturated:** share of joint-frames turning faster than 95% of ``ω_max``;
- **foot slide:** horizontal speed of a foot or toe joint while it is planted (the model's own
  contact output for rollouts, the recorded flags for the data);
- **jitter:** mean joint-position jerk (third difference), m/s³;
- **energy by 10 s segment:** mean joint angular speed. Falling towards 0 means the motion froze;
  climbing far above the data means it ran away;
- **hips height:** mean, lowest and highest;
- **short-horizon error:** against the recorded continuation at 0.5, 1 and 2 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from motionlab import fk, m1, motion_signal, rotation, skeleton
from motionlab.datasets import m1_data
from motionlab.formats import DT, FPS


@dataclass
class Motion:
    rotations: Tensor  # (B, T, J, 3, 3)
    root: Tensor  # (B, T, 3)
    contacts: Tensor  # (B, T, 4) bool


@torch.no_grad()
def free_rollout(model: m1.MotionPrior, normaliser: m1_data.Normaliser, seed: m1_data.Batch,
                 context: int, ticks: int,
                 signal: motion_signal.MotionSignal | None = None) -> Motion:
    """Recorded inputs for ``context`` ticks, then the model alone for ``ticks``, with one
    ``MotionSignal`` throughout (default: everything unknown)."""
    device = seed.rot6d.device
    b = seed.rot6d.shape[0]
    if hasattr(model, "reset_noise"):  # v5: a fresh run (noise process, flight timer)
        model.reset_noise()
    chosen = signal if signal is not None else motion_signal.neutral()
    features = motion_signal.features(chosen).to(device).expand(b, -1)
    h = model.initial_state(b, device)
    recorded = seed.rotations
    rots, roots = [recorded[:, 0], recorded[:, 1]], [seed.root[:, 0], seed.root[:, 1]]
    planted = [seed.contacts[:, 0] > 0.5, seed.contacts[:, 1] > 0.5]
    scale = torch.tensor(1.0, device=device)
    for t in range(1, context + ticks - 1):
        pair = rotation.matrix_to_rot6d(torch.stack((rots[t - 1], rots[t]), 1))
        state = normaliser(m1_data.states(pair, torch.stack((roots[t - 1], roots[t]), 1))[:, 0])
        raw, h = model(state, features, h)
        step = m1.advance(model, raw, rots[t], roots[t], scale)
        if t + 1 < context:
            rots.append(recorded[:, t + 1])
            roots.append(seed.root[:, t + 1])
            planted.append(seed.contacts[:, t + 1] > 0.5)
        else:
            rots.append(step.rotations)
            roots.append(step.root)
            planted.append(torch.sigmoid(step.contact_logits) > 0.5)
    return Motion(torch.stack(rots, 1), torch.stack(roots, 1), torch.stack(planted, 1))


def measures(motion: Motion) -> dict[str, object]:
    r, root = motion.rotations, motion.root
    skel = skeleton.load()
    finite = bool(torch.isfinite(r).all() and torch.isfinite(root).all())
    angles = torch.rad2deg(rotation.rotation_angle(r[:, :, 1:]))  # all but the root
    limits = torch.tensor([a or 180.0 for a in skel.max_angle_deg[1:]], device=r.device)
    beyond = float((angles > limits).float().mean())
    # v3: the data-measured limits (m1.joint_limits), per joint. The recordings themselves are
    # outside them in about 0.1% of joint-frames by construction (two 0.05% tails per axis).
    outside = m1.limit_excess(r[:, :, 1:], m1.joint_limits().to(r.device).without_hips()) > 0
    by_joint = {name: round(float(outside[..., j].float().mean()), 4)
                for j, name in enumerate(skel.names[1:]) if bool(outside[..., j].any())}
    omega = rotation.angular_velocity(r[:, :-1], r[:, 1:], DT).norm(dim=-1)  # (B, T−1, J)
    saturated = float((omega > 0.95 * skel.omega_max.to(r.device)).float().mean())
    _, positions = fk.forward_kinematics(r, root, skel)
    feet = positions[:, :, [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]]
    speed = (feet[:, 1:] - feet[:, :-1])[..., [0, 2]].norm(dim=-1) / DT
    planted = motion.contacts[:, 1:] & motion.contacts[:, :-1]
    slide = float(speed[planted].mean()) if bool(planted.any()) else float("nan")
    jerk = (positions[:, 3:] - 3 * positions[:, 2:-1] + 3 * positions[:, 1:-2]
            - positions[:, :-3]).norm(dim=-1) / DT ** 3
    segment = 10 * FPS
    energy = [round(float(omega[:, s:s + segment].mean()), 3)
              for s in range(0, omega.shape[1] - segment + 1, segment)]
    hips = root[..., 1]
    travel = (root[:, 1:, [0, 2]] - root[:, :-1, [0, 2]]).norm(dim=-1) / DT
    return {
        "root_speed_m_s": round(float(travel.mean()), 3),
        "stable": finite,
        "beyond_limits_share": round(beyond, 4),
        "outside_data_limits_share": round(float(outside.float().mean()), 4),
        "outside_data_limits_by_joint": by_joint,
        "omega_saturated_share": round(saturated, 4),
        "planted_foot_slide_m_s": round(slide, 3),
        "planted_share": round(float(motion.contacts.float().mean()), 3),
        "jitter_m_s3": round(float(jerk.mean()), 1),
        "energy_by_10s_rad_s": energy,
        "hips_height_m": [round(float(hips.mean()), 3), round(float(hips.min()), 3),
                          round(float(hips.max()), 3)],
    }


@torch.no_grad()
def short_horizon(model: m1.MotionPrior, normaliser: m1_data.Normaliser, seed: m1_data.Batch,
                  context: int) -> dict[str, list[float]]:
    """Mean joint rotation error (degrees) and relative joint position error (cm) at 0.5, 1 and
    2 s, against the recording."""
    ticks = seed.rot6d.shape[1] - context
    motion = free_rollout(model, normaliser, seed, context, ticks)
    out: dict[str, list[float]] = {"seconds": [], "rotation_deg": [], "position_cm": []}
    for seconds in (0.5, 1.0, 2.0):
        t = context + int(seconds * FPS) - 1
        if t >= seed.rot6d.shape[1]:
            continue
        pred, target = motion.rotations[:, t], seed.rotations[:, t]
        out["seconds"].append(seconds)
        out["rotation_deg"].append(round(float(torch.rad2deg(
            rotation.geodesic(pred[:, 1:], target[:, 1:]).mean())), 2))
        out["position_cm"].append(round(float((m1.relative_positions(pred)
                                               - m1.relative_positions(target))
                                              .norm(dim=-1).mean()) * 100, 2))
    return out


def recorded_motion(sampler: m1_data.WindowSampler, windows: int, ticks: int,
                    generator: torch.Generator) -> Motion:
    batch = sampler.sample(windows, ticks, generator, augment=False)
    return Motion(batch.rotations, batch.root, batch.contacts > 0.5)


def seed_windows(sampler: m1_data.WindowSampler, windows: int, ticks: int,
                 generator: torch.Generator) -> m1_data.Batch:
    return sampler.sample(windows, ticks, generator, augment=False)
