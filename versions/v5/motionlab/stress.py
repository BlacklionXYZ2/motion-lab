"""Stress battery (v5, 2026-10-30): probes that try to make a motion prior misbehave.

The M1 evaluation (``m1_eval``) reports averages over calm, in-distribution free runs. These
probes look for the behaviours that averages hide, each measured on the recordings too where a
reference makes sense:

- **long runs** (5 min): freezing, a constant spin, drifting into a lean;
- **gliding:** the root travels while no foot moves faster than it (a "moonwalk");
- **floor:** feet below the support, or both feet off it for long while not running;
- **switches:** the keyword changes every few seconds. Do switches pop, and does she respond?
- **flicker:** the keyword changes every tick (SORA's upstream hysteresis can fail);
- **pushes:** the realised pose is knocked off the model's output, as MA's physics body will do;
- **cold starts:** a T-pose or a frozen copy of one frame as the seed, as when her player starts;
- **mirror bias:** a seed and its mirror image should turn by opposite amounts;
- **body sanity:** legs crossing through each other, a hand inside the torso, the head below the
  hips while she isn't falling.

Everything runs on whatever device the model is on; the CPU is fast enough for a 1M model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from motionlab import fk, formats, m1, motion_signal, rotation, skeleton
from motionlab.datasets import m1_data
from motionlab.formats import DT, FPS
from motionlab.motion_signal import MotionSignal

JOINT = skeleton.INDEX
FEET = [JOINT[n] for n in skeleton.CONTACT_JOINTS]


@dataclass
class Driven:
    rotations: Tensor  # (B, T, J, 3, 3), the free part only
    root: Tensor  # (B, T, 3)
    contacts: Tensor  # (B, T, 4) bool, the model's own contact output


def signal_schedule(signals: list[MotionSignal], ticks: int, every: int, batch: int) -> Tensor:
    """``(B, ticks, F)``: ``signals`` in turn, each held for ``every`` ticks."""
    rows = [motion_signal.features(signals[(t // every) % len(signals)]) for t in range(ticks)]
    return torch.stack(rows).unsqueeze(0).expand(batch, -1, -1)


@torch.no_grad()
def drive(model: m1.MotionPrior, normaliser: m1_data.Normaliser, seed: m1_data.Batch,
          context: int, ticks: int, signal: Tensor | MotionSignal | None = None,
          perturb: Callable[[int, Tensor], Tensor] | None = None,
          controller: Callable[[int, Tensor, Tensor], Tensor] | None = None,
          reset: int = 0) -> Driven:
    """Like ``m1_eval.free_rollout``, plus a per-tick signal ``(B, ticks, F)`` and an optional
    ``perturb(t, rotations) -> rotations`` applied to the realised pose of free tick ``t`` (what
    the model sees next, as if physics moved her). ``controller(t, rotations, root)`` gives the
    features of free tick ``t`` from her current pose, closing the loop as SORA does.

    ``reset`` (ticks, v5): staggered memories. A second hidden state runs alongside, restarted
    from zero; every ``reset`` ticks, half a period after its restart, it takes over. The
    model then never runs on more memory than training gave it (windows up to 2.8 s). Without
    this, v4's memory grew 60% over 5 minutes and she hovered with her feet tucked up."""
    device = seed.rot6d.device
    b = seed.rot6d.shape[0]
    model.reset_noise()  # a fresh run: noise process and flight timer
    if signal is None:
        signal = motion_signal.neutral()
    if isinstance(signal, MotionSignal):
        signal = motion_signal.features(signal).to(device).expand(b, ticks, -1)
    signal = signal.to(device)
    h = model.initial_state(b, device)
    spare = model.initial_state(b, device)
    recorded = seed.rotations
    rots, roots = [recorded[:, 0], recorded[:, 1]], [seed.root[:, 0], seed.root[:, 1]]
    contacts: list[Tensor] = []
    scale = torch.tensor(1.0, device=device)
    for t in range(1, context + ticks - 1):
        free = t + 1 - context  # index of the free tick being produced, when ≥ 0
        features = signal[:, max(free, 0)] if free < ticks else signal[:, -1]
        if controller is not None and free >= 0:
            features = controller(free, rots[t], roots[t])
        pair = rotation.matrix_to_rot6d(torch.stack((rots[t - 1], rots[t]), 1))
        state = normaliser(m1_data.states(pair, torch.stack((roots[t - 1], roots[t]), 1))[:, 0])
        raw, h = model(state, features, h)
        if reset:
            _, spare = model(state, features, spare)
            if t >= context and (t - context) % reset == reset // 2:
                h, spare = spare, model.initial_state(b, device)
        step = m1.advance(model, raw, rots[t], roots[t], scale)
        if t + 1 < context:
            rots.append(recorded[:, t + 1])
            roots.append(seed.root[:, t + 1])
        else:
            nxt = step.rotations
            if perturb is not None:
                nxt = perturb(free, nxt)
            rots.append(nxt)
            roots.append(step.root)
            contacts.append(torch.sigmoid(step.contact_logits) > 0.5)
    r = torch.stack(rots, 1)[:, context:]
    root = torch.stack(roots, 1)[:, context:]
    return Driven(r, root, torch.stack(contacts, 1)[:, -r.shape[1]:])


def go_towards(base: MotionSignal, world_yaw: Tensor, speed: float | Callable[[int], float]
               ) -> Callable[[int, Tensor, Tensor], Tensor]:
    """A controller (v5): travel in the world direction ``world_yaw`` (per seed, radians, as
    ``formats.heading`` measures it) at ``speed`` m/s, given to the model in her heading frame
    every tick, as SORA's player does for ARDY."""
    features = motion_signal.features(base)
    loco = motion_signal.LOCOMOTION

    def control(t: int, rotations: Tensor, root: Tensor) -> Tensor:
        s = speed(t) if callable(speed) else speed
        rel = world_yaw.to(rotations.device) - formats.heading(rotations[:, 0])
        out = features.to(rotations.device).expand(rotations.shape[0], -1).clone()
        out[:, loco[0]] = s * torch.sin(rel)  # her left
        out[:, loco[1]] = s * torch.cos(rel)  # her forward
        out[:, loco[2]] = 1.0
        return out
    return control


# --- seeds -------------------------------------------------------------------------------------

def mirrored(seed: m1_data.Batch) -> m1_data.Batch:
    b = seed.rot6d.shape[0]
    return m1_data.augment_batch(seed, torch.ones(b, dtype=torch.bool, device=seed.rot6d.device),
                                 torch.zeros(b, device=seed.rot6d.device))


def frozen(seed: m1_data.Batch, context: int) -> m1_data.Batch:
    """Every context frame a copy of the seed's first frame: standing perfectly still."""
    rot6d = seed.rot6d[:, :1].expand(-1, context + 1, -1, -1).clone()
    root = seed.root[:, :1].expand(-1, context + 1, -1).clone()
    return m1_data.Batch(rot6d, root, seed.contacts[:, :1].expand(-1, context + 1, -1).clone(),
                         None)


def t_pose(batch: int, context: int, device: torch.device | str = "cpu") -> m1_data.Batch:
    """The canonical rest pose (a T-pose), held still, hips at rest height: her player's start."""
    eye = torch.eye(3, device=device).expand(batch, context + 1, skeleton.J, 3, 3)
    root = torch.zeros(batch, context + 1, 3, device=device)
    root[..., 1] = float(skeleton.load().offsets[0, 1])
    return m1_data.Batch(rotation.matrix_to_rot6d(eye), root,
                         torch.ones(batch, context + 1, 4, device=device), None)


def push(at: int, angle: float, generator: torch.Generator) -> Callable[[int, Tensor], Tensor]:
    """At free tick ``at``, turn every joint but the root by a random axis through ``angle``."""
    def apply(t: int, r: Tensor) -> Tensor:
        if t != at:
            return r
        axis = torch.randn(r.shape[0], skeleton.J, 3, generator=generator).to(r.device)
        axis = axis / axis.norm(dim=-1, keepdim=True) * angle
        axis[:, 0] = 0
        return r @ rotation.so3_exp(axis)
    return apply


# --- measures ----------------------------------------------------------------------------------

def positions(r: Tensor, root: Tensor) -> Tensor:
    return fk.forward_kinematics(r, root, skeleton.load())[1]


def turn_rate(r: Tensor) -> Tensor:
    """``(B, T−1)`` signed yaw rate of her facing, rad/s."""
    yaw = formats.heading(r[:, :, 0])
    d = yaw[:, 1:] - yaw[:, :-1]
    return torch.atan2(torch.sin(d), torch.cos(d)) / DT


def energy(r: Tensor) -> Tensor:
    """``(B, T−1)`` mean joint angular speed, rad/s, all joints but the root."""
    return rotation.angular_velocity(r[:, :-1, 1:], r[:, 1:, 1:], DT).norm(dim=-1).mean(-1)


def ground_speed(p: Tensor) -> Tensor:
    return (p[:, 1:] - p[:, :-1])[..., [0, 2]].norm(dim=-1) / DT


def body(r: Tensor, root: Tensor, toe_floor: float) -> dict[str, float]:
    """Gliding, floor and body-sanity shares for a motion (``toe_floor``: the recordings' 0.5th
    percentile toe height, below which a toe is "in the floor")."""
    p = positions(r, root)
    root_speed = ground_speed(p[:, :, 0])  # (B, T−1)
    feet_speed = ground_speed(p[:, :, FEET]).amax(-1)  # the fastest foot joint
    moving = root_speed > 0.3
    glide = moving & (feet_speed < 1.2 * root_speed)
    # "glide" also counts the body moving over planted feet (a stance phase: the recordings
    # score 30%). "Carried" is the real glide: even the slowest foot travels at more than
    # 0.1 m/s + half the root's speed, so no foot is planted (v5, 2026-10-30).
    slowest = ground_speed(p[:, :, FEET]).amin(-1)
    carried = moving & (slowest > 0.1 + 0.5 * root_speed)
    toes = p[:, :, [JOINT["leftToes"], JOINT["rightToes"]], 1]
    feet_y = p[:, :, FEET, 1].amin(-1)
    sink = (toes.amin(-1) < toe_floor - 0.03)
    airborne = feet_y > 0.15  # every foot joint more than 15 cm up
    # legs crossing: the left ankle to the right of the right ankle, in her heading frame
    yaw = formats.heading(r[:, :, 0])
    side = torch.stack((torch.cos(yaw), -torch.sin(yaw)), -1)  # her +X (left) on the ground
    ankles = p[:, :, JOINT["leftFoot"]] - p[:, :, JOINT["rightFoot"]]
    across = (ankles[..., [0, 2]] * side).sum(-1)
    hand_torso = torch.stack([(p[:, :, JOINT[h]] - p[:, :, JOINT[c]]).norm(dim=-1)
                              for h in ("leftHand", "rightHand")
                              for c in ("spine", "chest", "upperChest")], -1).amin(-1)
    head_low = p[:, :, JOINT["head"], 1] < p[:, :, 0, 1]
    return {
        "glide_share_when_moving": round(float(glide.float().sum() / moving.float().sum()), 4)
        if bool(moving.any()) else float("nan"),
        "carried_share_when_moving": round(float(carried.float().sum() / moving.float().sum()),
                                           4) if bool(moving.any()) else float("nan"),
        "toe_in_floor_share": round(float(sink.float().mean()), 4),
        "airborne_share": round(float(airborne.float().mean()), 4),
        "legs_crossed_share": round(float((across < -0.02).float().mean()), 4),
        "hand_in_torso_share": round(float((hand_torso < 0.08).float().mean()), 4),
        "head_below_hips_share": round(float(head_low.float().mean()), 4),
    }


def limits_share(r: Tensor) -> float:
    lim = m1.joint_limits().to(r.device).without_hips()
    return float((m1.limit_excess(r[:, :, 1:], lim) > 0).float().mean())


def segments(x: Tensor, seconds: float) -> list[float]:
    n = int(seconds * FPS)
    return [round(float(x[:, s:s + n].mean()), 3) for s in range(0, x.shape[1] - n + 1, n)]


def acceleration_peak(r: Tensor) -> Tensor:
    """``(B, T−2)`` largest joint angular acceleration per frame, rad/s²."""
    w = rotation.angular_velocity(r[:, :-1, 1:], r[:, 1:, 1:], DT)
    return ((w[:, 1:] - w[:, :-1]) / DT).norm(dim=-1).amax(-1)


def straight_line_share(root: Tensor, seconds: float = 10.0) -> float:
    """How far she ends up from where she started over ``seconds``, against the path length:
    1 is a straight line, 0 is circling on the spot."""
    n = int(seconds * FPS)
    out = []
    for s in range(0, root.shape[1] - n, n):
        seg = root[:, s:s + n, :][..., [0, 2]]
        path = (seg[:, 1:] - seg[:, :-1]).norm(dim=-1).sum(-1)
        net = (seg[:, -1] - seg[:, 0]).norm(dim=-1)
        keep = path > 0.5
        if bool(keep.any()):
            out.append((net[keep] / path[keep]).mean())
    return round(float(torch.stack(out).mean()), 3) if out else float("nan")


ROOT_SPIKE_M_S = 5.0  # faster than any frame of the recordings (their fastest: 4.9 m/s)


def summary(r: Tensor, root: Tensor, toe_floor: float) -> dict[str, object]:
    turn = turn_rate(r)
    speed = ground_speed(root)
    return {
        # v5: root jumps. Tracking measured over 0.5 s taught latent-vel-free5 single-frame
        # jumps of up to 10.5 m/s that no other measure caught.
        "root_speed_p99_m_s": round(float(torch.quantile(speed.flatten()[::3], 0.99)), 3),
        "root_speed_max_m_s": round(float(speed.max()), 2),
        "root_spike_share": round(float((speed > ROOT_SPIKE_M_S).float().mean()), 5),
        "stable": bool(torch.isfinite(r).all() and torch.isfinite(root).all()),
        "energy_by_30s": segments(energy(r), 30),
        "root_speed_by_30s": segments(ground_speed(root), 30),
        "turn_rate_abs_rad_s": round(float(turn.abs().mean()), 3),
        "turn_rate_signed_mean_rad_s": round(float(turn.mean()), 3),
        "straightness_10s": straight_line_share(root),
        "outside_limits_share": round(limits_share(r), 4),
        "hips_m": [round(float(root[..., 1].mean()), 3), round(float(root[..., 1].min()), 3),
                   round(float(root[..., 1].max()), 3)],
        "frozen_seed_share_last_30s": round(float(
            (energy(r)[:, -30 * FPS:].mean(-1) < 0.1).float().mean()), 3),
        **body(r, root, toe_floor),
    }


def math_ok(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))
