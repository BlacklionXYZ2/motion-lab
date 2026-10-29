"""M1, the motion prior: model, rollout and losses (roadmap M1).

**Model.** One tick per call, ``f(state, signal, h) → (raw, h')``, the shape M0's export path
takes (``export.CellStep`` uses :meth:`MotionPrior.encode`, the GRU and the decoder):

1. **Encoders, one per input** (roadmap decision 2). The pose (the motion state) has a two-layer
   network; each ``MotionSignal`` group (``SIGNAL_GROUPS``: intent, conversation, emotion,
   attention, audio, context) has its own small two-layer network. A keyword arrives one-hot, so
   its first layer acts as a learned embedding table, using only operations every runtime has.
2. **FiLM:** the signal groups together produce a scale and a shift for each pose feature,
   ``pose · (1 + γ) + β``, so a signal changes *how* she moves rather than only adding to it.
   It starts at zero (no effect) and learns its way in.
3. **Fusion:** a two-layer network turns the modulated pose and the signal features into the one
   learned vector the GRU reads.
4. **GRU,** then a two-layer **decoder**. Its last layer starts at zero, so an untrained model
   predicts "no change".

Activations are GELU throughout (the GRU has its own sigmoid and tanh gates).

**Signal neutralisation.** In training each signal group is replaced by its neutral value with
probability ``signal_dropout`` per window, even when its label is known, so neutral means
"unknown" to the model and it copes when SORA can't supply an input.

Two output heads (decision 1):

- ``rotation``: a residual on each joint's 6D rotation, ``R' = GS(rot6d(R) + Δ)``;
- ``omega``: bounded body-frame angular velocities, ``ω = ω_max · tanh(raw)``, integrated as
  ``R' = R · Exp(ω·Δt)``.

Either way the decoder also gives the root's horizontal velocity in her heading frame, the hips'
height above the floor (absolute, as an offset from her rest height, so a zero output stands
her up; integrating a vertical velocity instead let the first smoke model sink 14 m in 60 s),
and foot-contact logits. Decoding a raw output into the next pose (:func:`advance`)
happens outside the network, so the runtime must do the same arithmetic.

**Rollout.** :func:`rollout` runs a window tick by tick. The first ``context`` ticks see the
recorded states (and warm the hidden state up); after that each input is built from the model's
own previous output, as at run time. With no physics yet, the realised pose is the predicted
pose. The roadmap's rule 5 (state from the realised pose) holds once MA's body is in the loop.
Teacher forcing is ``context = window length``; the curriculum shortens it so rollouts grow
1 → 8 → 32 ticks (v3: on to 64 and 128, with longer windows).

**Losses** (per predicted frame, against the recording): geodesic rotation error, joint
positions relative to the root (FK on the canonical skeleton), angular velocity, angular
acceleration, root velocity, and contacts. v3 adds two that need no recording: joints outside
their data-measured limits (:func:`joint_limits`), and the hips outside their height range.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from motionlab import fk, formats, motion_signal, rotation, skeleton
from motionlab.datasets.m1_data import Batch, Normaliser, states
from motionlab.formats import DT


def _signal_groups() -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(motion_signal.FEATURE_NAMES):
        groups.setdefault(name.split(".")[0], []).append(i)
    return groups


# Feature indices per MotionSignal group, from the contract's own names (so a vocabulary change
# only changes the sizes).
SIGNAL_GROUPS = _signal_groups()


def neutral_signal() -> Tensor:
    return motion_signal.features(motion_signal.neutral())


def neutralise(signal: Tensor, probability: float, generator: torch.Generator | None = None
               ) -> Tensor:
    """Replace each signal group, per batch item, by its neutral value with ``probability``.
    ``signal`` is ``(B, F)`` or ``(B, L, F)``; a window keeps one choice throughout."""
    if probability <= 0:
        return signal
    out = signal.clone()
    neutral = neutral_signal().to(signal.device, signal.dtype)
    for indices in SIGNAL_GROUPS.values():
        drop = torch.rand(signal.shape[0], generator=generator).to(signal.device) < probability
        index = torch.tensor(indices, device=signal.device)
        mask = drop.view(-1, *([1] * (signal.dim() - 1)))
        out[..., index] = torch.where(mask, neutral[index], out[..., index])
    return out


def _mlp(inputs: int, width: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, outputs))


@dataclass
class ModelConfig:
    hidden: int = 256
    layers: int = 1
    head: str = "rotation"  # rotation | omega
    pose_width: int = 256
    fusion_width: int = 256
    decoder_width: int = 256
    signal_widths: dict[str, int] = field(default_factory=lambda: {
        "intent": 32, "conversation": 16, "emotion": 16, "attention": 16, "audio": 16,
        "context": 8})


@dataclass
class LossWeights:
    rotation: float = 1.0
    position: float = 4.0
    omega: float = 0.05
    acceleration: float = 0.02
    root_velocity: float = 0.5
    root_height: float = 2.0
    root_height_velocity: float = 0.5
    contacts: float = 0.1
    joint_limits: float = 1.0
    hips_limits: float = 2.0
    extra: dict[str, float] = field(default_factory=dict)


LIMITS_ASSET = Path(__file__).with_name("assets") / "joint_limits.json"


@dataclass(frozen=True)
class JointLimits:
    """v3's limits from the training data (``scripts/joint_limits.py``). The hips joint (her
    facing) has none."""

    box_min: Tensor  # (J, 3) rotation vector, radians
    box_max: Tensor  # (J, 3)
    angle_max: Tensor  # (J,) radians from rest
    hips_height: tuple[float, float]  # metres

    def to(self, device: torch.device | str) -> JointLimits:
        return JointLimits(self.box_min.to(device), self.box_max.to(device),
                           self.angle_max.to(device), self.hips_height)

    def without_hips(self) -> JointLimits:
        return JointLimits(self.box_min[1:], self.box_max[1:], self.angle_max[1:],
                           self.hips_height)


@cache
def joint_limits() -> JointLimits:
    data = json.loads(LIMITS_ASSET.read_text("utf-8"))
    free = 1e3  # no limit
    box_min = torch.full((skeleton.J, 3), -free)
    box_max = torch.full((skeleton.J, 3), free)
    angle = torch.full((skeleton.J,), free)
    for name, limits in data["joints"].items():
        j = skeleton.INDEX[name]
        box_min[j] = torch.tensor(limits["box_min_rad"])
        box_max[j] = torch.tensor(limits["box_max_rad"])
        angle[j] = math.radians(limits["angle_max_deg"])
    low, high = data["hips_height_m"]
    return JointLimits(box_min, box_max, angle, (float(low), float(high)))


def limit_excess(rotations: Tensor, limits: JointLimits) -> Tensor:
    """``(..., J, 3, 3)`` → ``(..., J)``: how far (radians) each joint is outside its limits,
    the box's excess on each axis plus the angle's; 0 inside."""
    w = rotation.so3_log(rotations)
    box = (F.relu(w - limits.box_max) + F.relu(limits.box_min - w)).sum(-1)
    return box + F.relu(rotation.rotation_angle(rotations) - limits.angle_max)


class MotionPrior(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.head not in ("rotation", "omega"):
            raise ValueError(f"unknown head {config.head!r}")
        self.config = config
        self.hidden, self.layers = config.hidden, config.layers
        missing = set(SIGNAL_GROUPS) - set(config.signal_widths)
        if missing:
            raise ValueError(f"no encoder width for signal groups {sorted(missing)}")
        self.pose_encoder = nn.Sequential(_mlp(formats.STATE_SIZE, config.pose_width,
                                               config.pose_width), nn.GELU())
        self.signal_encoders = nn.ModuleDict({
            name: nn.Sequential(_mlp(len(indices), config.signal_widths[name],
                                     config.signal_widths[name]), nn.GELU())
            for name, indices in SIGNAL_GROUPS.items()})
        for name, indices in SIGNAL_GROUPS.items():
            # A fixed selection matrix rather than indexing, so the export uses only the
            # multiplications the runtime spike already ran in Unity (no Gather).
            select = torch.zeros(motion_signal.FEATURE_SIZE, len(indices))
            select[indices, range(len(indices))] = 1
            self.register_buffer(f"select_{name}", select, persistent=False)
        signal_width = sum(config.signal_widths[name] for name in SIGNAL_GROUPS)
        self.film = nn.Linear(signal_width, 2 * config.pose_width)
        nn.init.zeros_(self.film.weight)  # starts as no modulation
        nn.init.zeros_(self.film.bias)
        self.fusion = nn.Sequential(_mlp(config.pose_width + signal_width, config.fusion_width,
                                         config.fusion_width), nn.GELU())
        self.gru = nn.GRU(config.fusion_width, config.hidden, num_layers=config.layers,
                          batch_first=True)
        per_joint = 6 if config.head == "rotation" else 3
        self.decoder = _mlp(config.hidden, config.decoder_width, skeleton.J * per_joint + 3 + 4)
        last = self.decoder[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)  # starts by predicting "no change"
        nn.init.zeros_(last.bias)
        self.register_buffer("omega_max", skeleton.load().omega_max.clone())

    def encode(self, state: Tensor, signal: Tensor) -> Tensor:
        """The fused vector the GRU reads for one tick."""
        pose = self.pose_encoder(state)
        signals = torch.cat([encoder(signal @ getattr(self, f"select_{name}"))
                             for name, encoder in self.signal_encoders.items()], -1)
        gamma, beta = self.film(signals).chunk(2, -1)
        return self.fusion(torch.cat((pose * (1 + gamma) + beta, signals), -1))

    def forward(self, state: Tensor, signal: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """``state (B, STATE_SIZE)`` normalised, ``signal (B, FEATURE_SIZE)``,
        ``h (layers, B, hidden)`` → raw output and ``h'``."""
        out, h_next = self.gru(self.encode(state, signal).unsqueeze(1), h)
        return self.decoder(out[:, 0]), h_next

    def initial_state(self, batch: int = 1, device: torch.device | str = "cpu") -> Tensor:
        return torch.zeros(self.layers, batch, self.hidden, device=device)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class Step:
    rotations: Tensor  # (B, J, 3, 3) the next pose
    root: Tensor  # (B, 3)
    contact_logits: Tensor  # (B, 4)
    omega_raw: Tensor | None  # (B, J, 3) before the bound (omega head)


def advance(model: MotionPrior, raw: Tensor, rotations: Tensor, root: Tensor,
            root_velocity_scale: Tensor) -> Step:
    """Decode a raw output into the next pose, from the current ``rotations`` and ``root``."""
    j = skeleton.J
    if model.config.head == "rotation":
        delta = raw[:, :j * 6].view(-1, j, 6)
        nxt = rotation.rot6d_to_matrix(rotation.matrix_to_rot6d(rotations) + delta)
        omega_raw = None
        split = j * 6
    else:
        omega_raw = raw[:, :j * 3].view(-1, j, 3)
        omega = model.omega_max.view(1, j, 1) * torch.tanh(omega_raw)
        nxt = rotation.integrate(rotations, omega, DT)
        split = j * 3
    velocity = raw[:, split:split + 3] * root_velocity_scale  # her heading frame, m/s
    velocity = velocity * velocity.new_tensor([1.0, 0.0, 1.0])  # height is set, not integrated
    turn = formats.yaw_matrix(formats.heading(nxt[:, 0]))
    root_next = root + (turn @ velocity.unsqueeze(-1)).squeeze(-1) * DT
    height = skeleton.load().offsets[0, 1].to(raw.device) + raw[:, split + 1]
    root_next = torch.cat((root_next[:, :1], height.unsqueeze(-1), root_next[:, 2:]), -1)
    return Step(nxt, root_next, raw[:, split + 3:split + 7], omega_raw)


@dataclass
class Rollout:
    """Predictions for frames 2…L−1 (each made from the two frames before it)."""

    rotations: Tensor  # (B, L−2, J, 3, 3)
    root: Tensor  # (B, L−2, 3)
    contact_logits: Tensor  # (B, L−2, 4)
    previous: Tensor  # (B, L−2, J, 3, 3): the input pose each prediction was made from
    previous_root: Tensor  # (B, L−2, 3): and its root
    omega_raw: Tensor | None  # (B, L−2, J, 3), omega head only


def rollout(model: MotionPrior, batch: Batch, normaliser: Normaliser, *, context: int,
            signal: Tensor | None = None, noise: float = 0.0,
            root_velocity_scale: float = 1.0, feedback_gradient: bool = False,
            signal_dropout: float = 0.0) -> Rollout:
    """Run a window tick by tick. Inputs are the recording while ``t < context`` (teacher
    forcing, with optional state noise) and the model's own previous outputs afterwards.

    By default no gradient flows back through a fed-back output: the state contains angular
    velocity (a finite difference, ×60 per second), so through 32 ticks of feedback the
    gradient grew to ~10⁹ in the first smoke run (2026-10-23). The model still learns from
    its own inputs; ``feedback_gradient=True`` restores full backpropagation."""
    device = batch.rot6d.device
    b, length = batch.rot6d.shape[:2]
    recorded = batch.rotations
    if signal is None:
        signal = neutral_signal().to(device).expand(b, -1)
    signal = neutralise(signal, signal_dropout)
    h = model.initial_state(b, device)
    scale = torch.tensor(root_velocity_scale, device=device)
    inputs = [recorded[:, 0], recorded[:, 1]]
    input_roots = [batch.root[:, 0], batch.root[:, 1]]
    steps: list[Step] = []
    for t in range(1, length - 1):
        pair = rotation.matrix_to_rot6d(torch.stack((inputs[t - 1], inputs[t]), 1))
        state = normaliser(states(pair, torch.stack((input_roots[t - 1], input_roots[t]), 1))
                           [:, 0])
        if noise and t < context:
            state = state + noise * torch.randn_like(state)
        raw, h = model(state, signal[:, t] if signal.dim() == 3 else signal, h)
        step = advance(model, raw, inputs[t], input_roots[t], scale)
        steps.append(step)
        own = t + 1 >= context
        fed_rotations = step.rotations if feedback_gradient else step.rotations.detach()
        fed_root = step.root if feedback_gradient else step.root.detach()
        inputs.append(fed_rotations if own else recorded[:, t + 1])
        input_roots.append(fed_root if own else batch.root[:, t + 1])
    return Rollout(
        torch.stack([s.rotations for s in steps], 1),
        torch.stack([s.root for s in steps], 1),
        torch.stack([s.contact_logits for s in steps], 1),
        torch.stack(inputs[1:-1], 1),
        torch.stack(input_roots[1:-1], 1),
        torch.stack([s.omega_raw for s in steps], 1) if steps[0].omega_raw is not None
        else None)


def relative_positions(rotations: Tensor) -> Tensor:
    """Joint positions relative to the root, from local rotations ``(..., J, 3, 3)``."""
    skel = skeleton.load()
    zero = torch.zeros(*rotations.shape[:-3], 3, device=rotations.device,
                       dtype=rotations.dtype)
    return fk.forward_kinematics(rotations, zero, skel)[1]


def losses(out: Rollout, batch: Batch, weights: LossWeights, *, skip: int = 0
           ) -> dict[str, Tensor]:
    """Loss terms against the recording, ignoring the first ``skip`` predictions (hidden-state
    warm-up)."""
    target = batch.rotations[:, 2:]
    previous_target = batch.rotations[:, 1:-1]
    sl = slice(skip, None)
    pred, prev = out.rotations[:, sl], out.previous[:, sl]
    tgt, prev_tgt = target[:, sl], previous_target[:, sl]
    terms = {
        "rotation": rotation.geodesic(pred, tgt).mean(),
        "position": (relative_positions(pred) - relative_positions(tgt)).norm(dim=-1).mean(),
    }
    omega = rotation.angular_velocity(prev, pred, DT)
    omega_target = rotation.angular_velocity(prev_tgt, tgt, DT)
    terms["omega"] = (omega - omega_target).norm(dim=-1).mean()
    terms["acceleration"] = ((omega[:, 1:] - omega[:, :-1])
                             - (omega_target[:, 1:] - omega_target[:, :-1])).norm(dim=-1).mean()
    velocity = out.root[:, sl] - out.previous_root[:, sl]
    velocity_target = (batch.root[:, 2:] - batch.root[:, 1:-1])[:, sl]
    terms["root_velocity"] = ((velocity - velocity_target) / DT)[..., [0, 2]].norm(dim=-1).mean()
    terms["root_height"] = (out.root[:, sl, 1] - batch.root[:, 2:][:, sl, 1]).abs().mean()
    # The hips' vertical speed against the recording: without it, the first baseline's hips
    # height buzzed about 2.3 cm per frame (jerk 19,400 m/s³; the data's is about 500).
    terms["root_height_velocity"] = ((velocity[..., 1] - velocity_target[..., 1]) / DT
                                     ).abs().mean()
    terms["contacts"] = F.binary_cross_entropy_with_logits(out.contact_logits[:, sl],
                                                           batch.contacts[:, 2:][:, sl])
    # v3: nothing above notices a toe or a hand bent past anything a person does (they are
    # light, at the ends of chains, and averaged with 21 other joints), and v2 drifted there
    # in long free runs. Summed over joints, so one joint's excess counts in full.
    limits = joint_limits().to(pred.device)
    terms["joint_limits"] = limit_excess(pred[:, :, 1:], limits.without_hips()).sum(-1).mean()
    low, high = limits.hips_height
    hips = out.root[:, sl, 1]
    terms["hips_limits"] = (F.relu(hips - high) + F.relu(low - hips)).mean()
    terms["total"] = sum(getattr(weights, k) * v for k, v in terms.items())  # type: ignore[assignment]
    return terms
