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
OPTIONAL_GROUPS = frozenset({"locomotion"})  # v5: only models whose config lists them


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
    # v5: size of the per-tick random latent (0: v4's deterministic model). See MotionPrior.
    latent: int = 0
    latent_width: int = 256
    # v5: travel comes from the planted foot (foot_locked); no new weights
    foot_lock: bool = False
    # v5: with the foot lock, the lowest foot also rests on the floor (no hovering); no weights
    ground_lock: bool = False


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
    # v5: the latent's KL divergence, with "free bits": each latent dimension's mean KL below
    # ``kl_free_bits`` nats costs nothing, so the posterior isn't pushed into carrying no
    # information (posterior collapse)
    kl: float = 0.0
    kl_free_bits: float = 0.0
    kl_raw: float = 0.0  # reported, never weighted
    # v5: the feet in the world, root included. v4 scored the legs only relative to the root
    # and the root on its own, so nothing tied them together: in 5-minute runs she glided
    # (the root moving while no foot did, 92% of moving frames against the recordings' 30%)
    # and hovered with every foot joint over 15 cm up (33 to 40% of frames, recordings 0.07%).
    foot_height: float = 0.0  # world foot-joint heights against the recording, m
    foot_skate: float = 0.0  # a foot planted in the recording moving over the ground, m/s
    # v5 free runs (free_run_losses): what must hold at any horizon, with no recording to match
    ground: float = 0.0  # the lowest foot joint off its floor height (flight allowed), m
    self_contact: float = 0.0  # a wrist inside the torso, m
    leg_cross: float = 0.0  # one ankle past the other sideways, beyond a natural cross, m
    energy_floor: float = 0.0  # joint speed below half the recorded window's, rad/s
    # ...and above 1.5× it: with a floor alone the first free-run latent model ran away (joint
    # speed 0.8 → 2.4 rad/s within 2.5 s, the recordings' 1.0)
    energy_ceiling: float = 0.0
    # latent-vel-free3 sampled with correlated noise matched the recordings' energy but
    # crouched (hips 0.54 m on average against 0.83), shook (jitter 2.3× the recordings') and
    # spent 12% of joint-frames outside the limits:
    posture: float = 0.0  # mean hips height more than 8 cm off the recorded window's, m
    jerk_ceiling: float = 0.0  # joint jerk above twice the recorded window's (a ratio)
    free_limits: float = 1.0  # the joint and hips limits' weight in free runs
    # following the velocity input in a free run (per 0.5 s, m/s): latent-vel-free3 reached a
    # fifth of the asked speed and never turned to face a new heading
    track: float = 0.0
    free_skate: float = 0.0  # a foot within 3 cm of the floor sliding, in free runs (m/s)
    # latent-vel-free7 followed commands by gliding (98% of moving frames, feet skimming above
    # free_skate's 3 cm) and never turned to face where it went:
    stance: float = 0.0  # the slowest foot joint faster than 0.1 + 0.15 × root speed, m/s
    facing: float = 0.0  # heading away from a counterfactual command's direction, 1 − cos
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
        missing = set(SIGNAL_GROUPS) - set(config.signal_widths) - OPTIONAL_GROUPS
        if missing:
            raise ValueError(f"no encoder width for signal groups {sorted(missing)}")
        unknown = set(config.signal_widths) - set(SIGNAL_GROUPS)
        if unknown:
            raise ValueError(f"no such signal groups {sorted(unknown)}")
        # v5: an optional group (``locomotion``) has an encoder only if the config lists it, so
        # v4-style checkpoints load unchanged
        groups = {name: indices for name, indices in SIGNAL_GROUPS.items()
                  if name in config.signal_widths}
        self.pose_encoder = nn.Sequential(_mlp(formats.STATE_SIZE, config.pose_width,
                                               config.pose_width), nn.GELU())
        self.signal_encoders = nn.ModuleDict({
            name: nn.Sequential(_mlp(len(indices), config.signal_widths[name],
                                     config.signal_widths[name]), nn.GELU())
            for name, indices in groups.items()})
        for name, indices in groups.items():
            # A fixed selection matrix rather than indexing, so the export uses only the
            # multiplications the runtime spike already ran in Unity (no Gather).
            select = torch.zeros(motion_signal.FEATURE_SIZE, len(indices))
            select[indices, range(len(indices))] = 1
            self.register_buffer(f"select_{name}", select, persistent=False)
        signal_width = sum(config.signal_widths[name] for name in groups)
        self.film = nn.Linear(signal_width, 2 * config.pose_width)
        nn.init.zeros_(self.film.weight)  # starts as no modulation
        nn.init.zeros_(self.film.bias)
        self.fusion = nn.Sequential(_mlp(config.pose_width + signal_width, config.fusion_width,
                                         config.fusion_width), nn.GELU())
        self.gru = nn.GRU(config.fusion_width, config.hidden, num_layers=config.layers,
                          batch_first=True)
        per_joint = 6 if config.head == "rotation" else 3
        self.decoder = _mlp(config.hidden + config.latent, config.decoder_width,
                            skeleton.J * per_joint + 3 + 4)
        last = self.decoder[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)  # starts by predicting "no change"
        nn.init.zeros_(last.bias)
        self.register_buffer("omega_max", skeleton.load().omega_max.clone())
        self.temperature = 1.0  # scales the prior's noise when sampling (v5)
        # v5: run-time noise as an AR(1) process, eps_t = ρ·eps_(t−1) + √(1 − ρ²)·n_t (still
        # standard normal per tick). Independent draws every tick made the first free-run model
        # shake: jitter 3,666 m/s³ against the recordings' 471. The posterior's z, which the
        # decoder trained on, changes smoothly from tick to tick.
        self.noise_correlation = 0.0
        self._eps: Tensor | None = None
        if config.latent:
            # v5: a VRNN-style latent (Chung et al. 2015; Ling et al. 2020's Motion VAE uses a
            # fixed prior). Each tick a latent z is drawn: in training from the posterior, which
            # sees the recorded next frame, and at run time from the learned prior, which sees
            # only the GRU's memory and the current state. The decoder then doesn't have to
            # average every plausible next frame into one calm one: z picks which.
            n, w = config.latent, config.latent_width
            self.prior = _mlp(config.hidden + formats.STATE_SIZE, w, 2 * n)
            self.posterior = _mlp(config.hidden + 2 * formats.STATE_SIZE, w, 2 * n)
            for net in (self.prior, self.posterior):  # both start as N(0, I): no KL at first
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
            self.latent_in = nn.Linear(n, config.fusion_width)
            nn.init.normal_(self.latent_in.weight, std=0.01)
            nn.init.zeros_(self.latent_in.bias)

    def reset_noise(self) -> None:
        """Forget the run-time state kept between ticks: the noise process and the foot lock's
        airborne timer (a new run starts fresh)."""
        self._eps = None
        self._air = None

    def airborne(self, batch: int, device: torch.device | str) -> Tensor:
        """``(B,)`` seconds every foot has been off the floor (the foot lock's flight timer)."""
        air = getattr(self, "_air", None)
        if air is None or air.shape[0] != batch:
            air = torch.zeros(batch, device=device)
        return air

    def _correlated_noise(self, like: Tensor) -> Tensor:
        fresh = torch.randn_like(like)
        rho = self.noise_correlation
        if self._eps is not None and self._eps.shape == like.shape:
            fresh = rho * self._eps + math.sqrt(1 - rho * rho) * fresh
        self._eps = fresh.detach()
        return fresh

    def encode(self, state: Tensor, signal: Tensor) -> Tensor:
        """The fused vector the GRU reads for one tick."""
        pose = self.pose_encoder(state)
        signals = torch.cat([encoder(signal @ getattr(self, f"select_{name}"))
                             for name, encoder in self.signal_encoders.items()], -1)
        gamma, beta = self.film(signals).chunk(2, -1)
        return self.fusion(torch.cat((pose * (1 + gamma) + beta, signals), -1))

    def step(self, state: Tensor, signal: Tensor, h: Tensor, *, target: Tensor | None = None,
             eps: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
        """One tick: raw output, ``h'`` and (v5, in training) the latent's KL per dimension
        ``(B, latent)``. ``target`` is the recorded next state (normalised), which only the
        posterior sees; without it z comes from the prior. ``eps`` is the standard normal noise
        (drawn here if not given; zeros give the prior's mean)."""
        x = self.encode(state, signal)
        z = kl = None
        if self.config.latent:
            top = h[-1]
            p_mean, p_logvar = self.prior(torch.cat((top, state), -1)).chunk(2, -1)
            p_logvar = p_logvar.clamp(-8.0, 4.0)
            if target is not None:
                q_mean, q_logvar = self.posterior(torch.cat((top, state, target), -1)).chunk(2, -1)
                q_logvar = q_logvar.clamp(-8.0, 4.0)
                kl = 0.5 * (p_logvar - q_logvar
                            + (q_logvar.exp() + (q_mean - p_mean) ** 2) / p_logvar.exp() - 1)
                mean, logvar, scale = q_mean, q_logvar, 1.0
            else:
                mean, logvar, scale = p_mean, p_logvar, self.temperature
                if eps is None and self.noise_correlation > 0:
                    eps = self._correlated_noise(mean)
            if eps is None:
                eps = torch.randn_like(mean)
            z = mean + (0.5 * logvar).exp() * eps * scale
            x = x + self.latent_in(z)
        out, h_next = self.gru(x.unsqueeze(1), h)
        top_out = out[:, 0] if z is None else torch.cat((out[:, 0], z), -1)
        return self.decoder(top_out), h_next, kl

    def forward(self, state: Tensor, signal: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """``state (B, STATE_SIZE)`` normalised, ``signal (B, FEATURE_SIZE)``,
        ``h (layers, B, hidden)`` → raw output and ``h'``. With a latent (v5), z is drawn from
        the prior."""
        raw, h_next, _ = self.step(state, signal, h)
        return raw, h_next

    def load_deterministic(self, state_dict: dict[str, Tensor]) -> None:
        """Start a bigger model from a smaller checkpoint (v5): a v4-style model without the
        latent or the ``locomotion`` encoder. Every shared weight is copied. New inputs are always
        appended, so a grown weight keeps the old block and gets new columns: zero for FiLM and
        fusion (the model starts exactly as the old one), small for the decoder's latent
        columns. New modules keep their own initialisation."""
        own = self.state_dict()
        for key, value in state_dict.items():
            if key not in own:
                continue
            if own[key].shape == value.shape:
                own[key] = value
                continue
            if own[key].dim() != 2 or own[key].shape[0] != value.shape[0]:
                raise ValueError(f"{key}: can't grow {tuple(value.shape)} into "
                                 f"{tuple(own[key].shape)}")
            merged = (torch.randn_like(own[key]) * 0.01 if key == "decoder.0.weight"
                      else torch.zeros_like(own[key]))
            merged[:, :value.shape[1]] = value
            own[key] = merged
        self.load_state_dict(own)

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
    if model.config.foot_lock:
        root_next, model._air = foot_locked(rotations, nxt, root, root_next,
                                            model.airborne(root.shape[0], root.device),
                                            ground_height=model.config.ground_lock)
    return Step(nxt, root_next, raw[:, split + 3:split + 7], omega_raw)


LOCK_FULL_M, LOCK_NONE_M = 0.03, 0.08  # the lowest foot's lift: fully locked below, free above
LOCK_SOFTMIN_M = 0.005  # 1 cm let a toe lifted 1.5 cm keep a fifth of the support
# No one stays airborne: after this long with every foot up, the lock takes the lowest foot
# again wherever it is. free9 skimmed every foot just above the lock's 8 cm and glided "in
# flight" (95% of steering frames); real running flight lasts about 0.1 to 0.15 s.
AIR_LIMIT_S = 0.3
LOCK_MAX_SPEED = 6.0  # m/s: the most the lock moves the root in one tick


def foot_locked(rotations: Tensor, nxt: Tensor, root: Tensor, root_model: Tensor,
                air: Tensor | None = None, ground_height: bool = False
                ) -> tuple[Tensor, Tensor]:
    """v5 foot lock (the roadmap's M2 foot lock, built into the step): while a foot is near the
    floor, the root moves so that foot stays where it was, so she travels only by stepping.
    Penalties couldn't stop the gliding (free7: 98% of moving frames; free8's stance term made
    standing still the easy way out) because the root velocity output could always move her
    without the legs. The support is a soft minimum over the contact joints' heights; the lock
    fades out between 3 and 8 cm of lift (running flight), where the model's own root velocity
    takes over. Height and facing stay the model's. Differentiable: travel reaches the legs."""
    feet = [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]
    before = relative_positions(rotations)[:, feet]  # (B, 4, 3), world axes, root at origin
    after = relative_positions(nxt)[:, feet]
    floor = _floor_heights().to(root.device)
    lift = before[..., 1] + root[:, 1:2] - floor  # (B, 4)
    # A foot leaving the floor isn't support: each foot counts by its higher position over the
    # two frames (with both feet down, an even split let the planted foot slide half the
    # moving one's way)
    lift_after = after[..., 1] + root_model[:, 1:2] - floor
    weight = torch.softmax(-torch.maximum(lift, lift_after) / LOCK_SOFTMIN_M, -1)
    gate = ((LOCK_NONE_M - lift.amin(-1)) / (LOCK_NONE_M - LOCK_FULL_M)).clamp(0, 1)
    if air is None:
        air = torch.zeros_like(gate)
    air = torch.where(gate < 0.5, air + DT, torch.zeros_like(air)).detach()
    gate = torch.where(air > AIR_LIMIT_S, torch.ones_like(gate), gate)  # flight is over
    shift = ((before - after)[..., [0, 2]] * weight.unsqueeze(-1)).sum(1)  # keeps support still
    # When the support switches sharply between feet, the root briefly moves with the swinging
    # foot: replaying recordings through the lock gave single-frame errors up to 4.5 m/s, and
    # lock-sup's root reached 8.4 m/s in one frame. Capped a little above the recordings'
    # fastest frame (5.1 m/s).
    size = shift.norm(dim=-1, keepdim=True)
    shift = shift * (LOCK_MAX_SPEED * DT / size.clamp_min(1e-9)).clamp(max=1.0)
    locked = root[:, [0, 2]] + shift
    ground = gate.unsqueeze(-1) * locked + (1 - gate.unsqueeze(-1)) * root_model[:, [0, 2]]
    height = root_model[:, 1]
    if ground_height:
        # v5: the lowest foot rests on the floor while locked, so she can't fold her legs up
        # under hips that keep their height (free12b hovered with every foot over 15 cm in
        # 93% of idle frames, hips at 0.57 m)
        settle = (torch.softmax(-lift_after / LOCK_SOFTMIN_M, -1) * lift_after).sum(-1)
        height = height - gate * settle
    return torch.stack((ground[:, 0], height, ground[:, 1]), -1), air


def _damped(x: Tensor, decay: float) -> Tensor:
    """``x`` unchanged, with its gradient scaled by ``decay``: credit through fed-back ticks
    decays geometrically instead of compounding (free12's first try, undamped 8-tick windows,
    went to NaN at step 85)."""
    if decay >= 1.0:
        return x
    return x.detach() + decay * (x - x.detach())


def with_command(features: Tensor, command_world: Tensor, world_root: Tensor) -> Tensor:
    """``features (B, F)`` with the locomotion group set to ``command_world (B, 2)`` (x, z, m/s)
    turned into the heading frame of ``world_root (B, 3, 3)``, and the intent set to match the
    speed (idle under 0.1 m/s, walk under 1.8, run above)."""
    loco = motion_signal.LOCOMOTION
    yaw = formats.heading(world_root)
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, z = command_world[:, 0], command_world[:, 1]
    out = features.clone()
    out[:, loco[0]] = c * x - s * z  # her left
    out[:, loco[1]] = s * x + c * z  # her forward
    out[:, loco[2]] = 1.0
    speed = command_world.norm(dim=-1)
    intents = [motion_signal.INDEX[f"intent.{n}"] for n in motion_signal.INTENTS]
    out[:, intents] = 0.0
    for name, chosen in (("idle", speed < 0.1), ("walk", (speed >= 0.1) & (speed < 1.8)),
                         ("run", speed >= 1.8)):
        out[:, motion_signal.INDEX[f"intent.{name}"]] = chosen.to(out.dtype)
    return out


@dataclass
class Rollout:
    """Predictions for frames 2…L−1 (each made from the two frames before it)."""

    rotations: Tensor  # (B, L−2, J, 3, 3)
    root: Tensor  # (B, L−2, 3)
    contact_logits: Tensor  # (B, L−2, 4)
    previous: Tensor  # (B, L−2, J, 3, 3): the input pose each prediction was made from
    previous_root: Tensor  # (B, L−2, 3): and its root
    omega_raw: Tensor | None  # (B, L−2, J, 3), omega head only
    kl: Tensor | None = None  # (B, L−2, latent), v5 latent models only


def rollout(model: MotionPrior, batch: Batch, normaliser: Normaliser, *, context: int,
            signal: Tensor | None = None, noise: float = 0.0,
            root_velocity_scale: float = 1.0, feedback_gradient: bool = False,
            feedback_window: int = 1, feedback_decay: float = 1.0,
            signal_dropout: float = 0.0, use_prior: bool = False,
            command_world: Tensor | None = None) -> Rollout:
    """Run a window tick by tick. Inputs are the recording while ``t < context`` (teacher
    forcing, with optional state noise) and the model's own previous outputs afterwards.

    By default no gradient flows back through a fed-back output: the state contains angular
    velocity (a finite difference, ×60 per second), so through 32 ticks of feedback the
    gradient grew to ~10⁹ in the first smoke run (2026-10-23). The model still learns from
    its own inputs; ``feedback_gradient=True`` restores full backpropagation."""
    device = batch.rot6d.device
    b, length = batch.rot6d.shape[:2]
    recorded = batch.rotations
    model.reset_noise()  # a fresh run: noise process and flight timer
    if signal is None:
        signal = neutral_signal().to(device).expand(b, -1)
    signal = neutralise(signal, signal_dropout)
    h = model.initial_state(b, device)
    scale = torch.tensor(root_velocity_scale, device=device)
    inputs = [recorded[:, 0], recorded[:, 1]]
    input_roots = [batch.root[:, 0], batch.root[:, 1]]
    steps: list[Step] = []
    kls: list[Tensor] = []
    # v5: the recorded next state for the latent's posterior; index t is frame t + 1's state
    # (``use_prior``: z from the prior as at run time, for validation)
    targets = (normaliser(states(batch.rot6d, batch.root))
               if model.config.latent and not use_prior else None)
    for t in range(1, length - 1):
        pair = rotation.matrix_to_rot6d(torch.stack((inputs[t - 1], inputs[t]), 1))
        state = normaliser(states(pair, torch.stack((input_roots[t - 1], input_roots[t]), 1))
                           [:, 0])
        if noise and t < context:
            state = state + noise * torch.randn_like(state)
        features = signal[:, t] if signal.dim() == 3 else signal
        if command_world is not None and t + 1 >= context:
            # v5: a world-frame velocity command, given in her heading frame each free tick
            features = with_command(features, command_world, inputs[t][:, 0])
        raw, h, kl = model.step(state, features, h,
                                target=None if targets is None else targets[:, t])
        if kl is not None:
            kls.append(kl)
        step = advance(model, raw, inputs[t], input_roots[t], scale)
        steps.append(step)
        own = t + 1 >= context
        # v5: ``feedback_window`` k lets the gradient run back through k consecutive own ticks
        # (cut between windows). Stepping is a multi-tick skill (lift a foot now, place it
        # ahead later); with a cut every tick, the foot-locked models never learned to travel
        # to a command (free10: tracking stuck at 1.0 m/s).
        keep = feedback_gradient or (feedback_window > 1 and (t + 1) % feedback_window != 0)
        fed_rotations = _damped(step.rotations, feedback_decay) if keep else step.rotations.detach()
        fed_root = _damped(step.root, feedback_decay) if keep else step.root.detach()
        inputs.append(fed_rotations if own else recorded[:, t + 1])
        input_roots.append(fed_root if own else batch.root[:, t + 1])
    return Rollout(
        torch.stack([s.rotations for s in steps], 1),
        torch.stack([s.root for s in steps], 1),
        torch.stack([s.contact_logits for s in steps], 1),
        torch.stack(inputs[1:-1], 1),
        torch.stack(input_roots[1:-1], 1),
        torch.stack([s.omega_raw for s in steps], 1) if steps[0].omega_raw is not None
        else None,
        torch.stack(kls, 1) if kls else None)


FREE_RUN_TERMS = frozenset({"ground", "self_contact", "leg_cross", "energy_floor",
                            "energy_ceiling", "posture", "jerk_ceiling", "track",
                            "free_skate", "stance", "facing"})
FLIGHT_M = 0.10  # how far the lowest foot joint may leave its floor height (running), m
SINK_M = 0.02
TORSO_RADIUS_M = 0.10  # a wrist this close to the spine, chest or upper chest is inside her
CROSS_M = 0.08  # ankles crossing further than this sideways (the recordings cross ~2 cm, 10%)


@cache
def _floor_heights() -> Tensor:
    """``(4,)`` each contact joint's height above the floor in the rest pose, m."""
    skel = skeleton.load()
    eye = torch.eye(3).expand(skeleton.J, 3, 3)
    root = skel.offsets[0].clone()
    _, positions = fk.forward_kinematics(eye[None], root[None], skel)
    return positions[0, [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS], 1]


def plausibility(rotations: Tensor, root: Tensor) -> dict[str, Tensor]:
    """v5: recording-free terms for ``(B, T, J, 3, 3)`` rotations and ``(B, T, 3)`` roots in the
    support frame (her hips' height measured from what she stands on). Each is 0 when she is
    plausible; the stress battery found v4 and the first latent model breaking all three in long
    runs (hovering in a tuck, a wrist inside the torso, legs crossed)."""
    rel = relative_positions(rotations)
    world_y = rel[..., 1] + root[..., 1:2]
    feet = [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]
    lift = (world_y[..., feet] - _floor_heights().to(rel.device)).amin(-1)  # (B, T)
    ground = F.relu(lift - FLIGHT_M) + F.relu(-lift - SINK_M)
    torso = rel[..., [skeleton.INDEX[n] for n in ("spine", "chest", "upperChest")], :]
    hands = rel[..., [skeleton.INDEX["leftHand"], skeleton.INDEX["rightHand"]], :]
    gap = (hands.unsqueeze(-2) - torso.unsqueeze(-3)).norm(dim=-1).amin(-1)  # (B, T, 2)
    contact = F.relu(TORSO_RADIUS_M - gap).sum(-1)
    yaw = formats.heading(rotations[..., 0, :, :])
    left = torch.stack((torch.cos(yaw), -torch.sin(yaw)), -1)  # her +X on the ground
    across = ((rel[..., skeleton.INDEX["leftFoot"], :] - rel[..., skeleton.INDEX["rightFoot"], :]
               )[..., [0, 2]] * left).sum(-1)
    cross = F.relu(-across - CROSS_M)
    return {"ground": ground.mean(), "self_contact": contact.mean(), "leg_cross": cross.mean()}


def relative_positions(rotations: Tensor) -> Tensor:
    """Joint positions relative to the root, from local rotations ``(..., J, 3, 3)``."""
    skel = skeleton.load()
    zero = torch.zeros(*rotations.shape[:-3], 3, device=rotations.device,
                       dtype=rotations.dtype)
    return fk.forward_kinematics(rotations, zero, skel)[1]


def random_commands(batch: int, generator: torch.Generator | None = None,
                    device: torch.device | str = "cpu", heading: Tensor | None = None,
                    ahead_share: float = 0.0) -> Tensor:
    """``(B, 2)`` world velocities: a quarter standing still, the rest 0.3 to 3 m/s (v5,
    counterfactual commands). Directions are uniform, except that with ``heading`` (B,) a share
    ``ahead_share`` points roughly ahead of her (normal, 30° spread). Under the foot lock, a
    command off to the side means stepping sideways or turning first, and with uniform
    directions the foot-locked models never learned to track at all (free10 to free12b)."""
    speed = 0.3 + 2.7 * torch.rand(batch, generator=generator)
    speed[torch.rand(batch, generator=generator) < 0.25] = 0.0
    angle = 2 * math.pi * torch.rand(batch, generator=generator)
    if heading is not None and ahead_share > 0:
        ahead = torch.rand(batch, generator=generator) < ahead_share
        offset = torch.randn(batch, generator=generator) * math.radians(30)
        angle = torch.where(ahead, heading.cpu() + offset, angle)
    return torch.stack((speed * torch.sin(angle), speed * torch.cos(angle)), -1).to(device)


def counterfactual_losses(model: MotionPrior, batch: Batch, normaliser: Normaliser,
                          weights: LossWeights, *, context: int, signal: Tensor | None = None,
                          generator: torch.Generator | None = None,
                          feedback_window: int = 1,
                          feedback_decay: float = 1.0,
                          ahead_share: float = 0.0) -> dict[str, Tensor]:
    """v5: a free run told to go somewhere the recording didn't (``random_commands``),
    closed-loop as SORA's player does it. latent-vel-free5 learned to follow commands that
    agreed with what she was already doing, and nothing else: asked to walk at 1.2 m/s she made
    0.09 and never turned. Scored on following the command (after a 1 s start) and the
    recording-free terms only: the energy, posture and jerk references come from a recording
    the command no longer matches."""
    heading = formats.heading(batch.rotations[:, max(context - 1, 0), 0])
    command = random_commands(batch.root.shape[0], generator, batch.root.device, heading,
                              ahead_share)
    model.reset_noise()
    out = rollout(model, batch, normaliser, context=context, signal=signal, use_prior=True,
                  command_world=command, feedback_window=feedback_window,
                  feedback_decay=feedback_decay)
    model.reset_noise()
    free = slice(max(context - 2, 0), None)
    pred, root = out.rotations[:, free], out.root[:, free]
    terms = plausibility(pred, root)
    limits = joint_limits().to(pred.device)
    terms["free_joint_limits"] = limit_excess(pred[:, :, 1:], limits.without_hips()).sum(-1).mean()
    low, high = limits.hips_height
    terms["free_hips_limits"] = (F.relu(root[..., 1] - high) + F.relu(low - root[..., 1])).mean()
    terms["free_skate"] = free_skate(pred, root)
    terms["stance"] = stance(pred, root)
    terms["facing"] = facing(pred, command, round(1.5 / DT))  # 1.5 s to turn
    velocity = step_velocity(root, out.previous_root[:, free])[:, round(1 / DT):]  # after 1 s
    n = velocity.shape[1] // TRACK_TICKS * TRACK_TICKS
    achieved = velocity[:, :n].unflatten(1, (-1, TRACK_TICKS)).mean(2)  # per 0.5 s
    terms["track"] = (achieved - command.unsqueeze(1)).norm(dim=-1).mean()
    terms["free_total"] = (weights.track * terms["track"]
                           + weights.ground * terms["ground"]
                           + weights.self_contact * terms["self_contact"]
                           + weights.leg_cross * terms["leg_cross"]
                           + weights.free_limits * weights.joint_limits
                           * terms["free_joint_limits"]
                           + weights.free_limits * weights.hips_limits
                           * terms["free_hips_limits"]
                           + weights.free_skate * terms["free_skate"]
                           + weights.stance * terms["stance"]
                           + weights.facing * terms["facing"])
    return {f"cf_{k}" if not k.startswith("free_") else k.replace("free_", "cf_", 1): v
            for k, v in terms.items()}


def stance(rotations: Tensor, root: Tensor) -> Tensor:
    """Recording-free stepping (v5): the slowest contact joint's ground speed above
    ``0.1 + 0.15 × root speed`` m/s. Walking keeps one foot nearly still (the recordings'
    slowest foot: 0.13 m/s on average at 0.5 to 1.5 m/s, 0.37 at 1.5 to 3 m/s); gliding moves
    every foot with the root."""
    rel = relative_positions(rotations)
    feet = [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]
    world = rel[..., feet, :] + root.unsqueeze(-2)
    slowest = (world[:, 1:, :, [0, 2]] - world[:, :-1, :, [0, 2]]).norm(dim=-1).amin(-1) / DT
    travel = (root[:, 1:, [0, 2]] - root[:, :-1, [0, 2]]).norm(dim=-1) / DT
    return F.relu(slowest - 0.1 - 0.15 * travel.detach()).mean()


def facing(rotations: Tensor, command: Tensor, start: int) -> Tensor:
    """1 − cos of the angle between her heading and ``command (B, 2)``'s direction, from tick
    ``start``, for commands faster than 0.5 m/s (people walk forwards); 0 without any."""
    fast = (command.norm(dim=-1) > 0.5).to(rotations.dtype)
    if not bool(fast.any()):
        return rotations.new_zeros(())
    want = torch.atan2(command[:, 0], command[:, 1])
    heading = formats.heading(rotations[:, start:, 0])
    off = 1 - torch.cos(heading - want.unsqueeze(1))
    return (off.mean(1) * fast).sum() / fast.sum()


SKATE_HEIGHT_M = 0.03


def free_skate(rotations: Tensor, root: Tensor) -> Tensor:
    """Recording-free foot skate (v5): each contact joint's ground speed, weighted by how close
    it is to its floor height (full weight on the floor, none at 3 cm up), m/s. Free runs had no
    contact labels to hold feet still, and the first models glided in 70% of moving frames."""
    rel = relative_positions(rotations)
    feet = [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]
    world = rel[..., feet, :] + root.unsqueeze(-2)
    lift = world[..., 1] - _floor_heights().to(rel.device)
    weight = (1 - lift / SKATE_HEIGHT_M).clamp(0, 1)
    speed = (world[:, 1:, :, [0, 2]] - world[:, :-1, :, [0, 2]]).norm(dim=-1) / DT
    pair = torch.minimum(weight[:, 1:], weight[:, :-1])
    return (pair * speed).sum() / pair.sum().clamp_min(1.0)


def free_run_losses(model: MotionPrior, batch: Batch, normaliser: Normaliser,
                    weights: LossWeights, *, context: int, signal: Tensor | None = None,
                    signal_dropout: float = 0.0, feedback_window: int = 1,
                    feedback_decay: float = 1.0) -> dict[str, Tensor]:
    """v5: run alone from ``context`` recorded ticks to the end of the window, as at run time
    (a latent model draws z from its prior), and score only what holds at any horizon:
    :func:`plausibility`, the joint and hips limits, and an energy floor of half the recorded
    window's own joint speed (v4 froze in every 5-minute run). Supervised rollouts never visit
    the states a free run drifts into, so nothing taught the model to come back from them."""
    model.reset_noise()  # each free run starts its own noise process
    if signal is not None:  # dropped here, so tracking knows which commands survived
        signal = neutralise(signal, signal_dropout)
    out = rollout(model, batch, normaliser, context=context, signal=signal, use_prior=True,
                  feedback_window=feedback_window, feedback_decay=feedback_decay)
    model.reset_noise()
    free = slice(max(context - 2, 0), None)
    pred, root = out.rotations[:, free], out.root[:, free]
    terms = plausibility(pred, root)
    limits = joint_limits().to(pred.device)
    terms["free_joint_limits"] = limit_excess(pred[:, :, 1:], limits.without_hips()).sum(-1).mean()
    low, high = limits.hips_height
    terms["free_hips_limits"] = (F.relu(root[..., 1] - high) + F.relu(low - root[..., 1])).mean()
    speed = rotation.angular_velocity(pred[:, :-1, 1:], pred[:, 1:, 1:], DT).norm(dim=-1).mean(
        (1, 2))
    recorded = batch.rotations[:, context:]
    recorded_speed = rotation.angular_velocity(recorded[:, :-1, 1:], recorded[:, 1:, 1:], DT
                                               ).norm(dim=-1).mean((1, 2))
    terms["energy_floor"] = F.relu(0.5 * recorded_speed - speed).mean()
    terms["energy_ceiling"] = F.relu(speed - 1.5 * recorded_speed).mean()
    recorded_root = batch.root[:, context:]
    terms["posture"] = F.relu((root[..., 1].mean(1) - recorded_root[..., 1].mean(1)).abs()
                              - POSTURE_M).mean()
    jerk = _jerk(relative_positions(pred))
    recorded_jerk = _jerk(relative_positions(recorded)).clamp_min(1.0)
    terms["jerk_ceiling"] = F.relu(jerk / recorded_jerk - 2.0).mean()
    terms["track"] = _tracking(pred, step_velocity(root, out.previous_root[:, free]), signal,
                               max(context - 2, 0))
    terms["free_skate"] = free_skate(pred, root)
    terms["stance"] = stance(pred, root)
    terms["free_total"] = (weights.stance * terms["stance"]
                           + weights.free_skate * terms["free_skate"]
                           + weights.track * terms["track"]
                           + weights.energy_ceiling * terms["energy_ceiling"]
                           + weights.ground * terms["ground"]
                           + weights.self_contact * terms["self_contact"]
                           + weights.leg_cross * terms["leg_cross"]
                           + weights.free_limits * weights.joint_limits
                           * terms["free_joint_limits"]
                           + weights.free_limits * weights.hips_limits
                           * terms["free_hips_limits"]
                           + weights.energy_floor * terms["energy_floor"]
                           + weights.posture * terms["posture"]
                           + weights.jerk_ceiling * terms["jerk_ceiling"])
    return terms


POSTURE_M = 0.08
TRACK_TICKS = 30  # 0.5 s


def step_velocity(out_root: Tensor, previous_root: Tensor) -> Tensor:
    """``(B, T, 2)`` ground velocity of each predicted tick from its own input, m/s. Every tick
    carries gradient (fed-back positions are detached), unlike a displacement between two
    distant frames: tracking measured that way handed a half second's error to one tick, which
    learned to jump (latent-vel-free5's root reached 10.5 m/s in one frame; free6 diverged)."""
    return (out_root - previous_root)[..., [0, 2]] / DT


def _tracking(rotations: Tensor, velocity: Tensor, signal: Tensor | None, offset: int) -> Tensor:
    """Mean distance (m/s) between her mean ground velocity over each 0.5 s of a free run
    (``velocity``, :func:`step_velocity`) and the velocity commanded at its start, turned into
    the world by her own heading then. ``rotations``/``velocity`` are the free run's
    predictions, which start at window frame ``offset + 2``; ``signal`` is the window's
    (B, L, F) features. Commands dropped to unknown don't count; 0 without any."""
    if signal is None or signal.dim() != 3:
        return rotations.new_zeros(())
    loco = motion_signal.LOCOMOTION
    errors, weights = [], []
    for t in range(0, velocity.shape[1] - TRACK_TICKS, TRACK_TICKS):
        command = signal[:, offset + 2 + t]
        has = command[:, loco[2]]
        local = torch.stack((command[:, loco[0]], torch.zeros_like(has), command[:, loco[1]]), -1)
        turn = formats.yaw_matrix(formats.heading(rotations[:, t, 0]))
        wanted = (turn @ local.unsqueeze(-1)).squeeze(-1)[:, [0, 2]]
        achieved = velocity[:, t:t + TRACK_TICKS].mean(1)
        errors.append((achieved - wanted).norm(dim=-1) * has)
        weights.append(has)
    if not errors:
        return rotations.new_zeros(())
    return torch.stack(errors).sum() / torch.stack(weights).sum().clamp_min(1.0)


def _jerk(positions: Tensor) -> Tensor:
    """``(B, T, J, 3)`` → ``(B,)`` mean joint-position jerk (third difference), m/s³."""
    third = (positions[:, 3:] - 3 * positions[:, 2:-1] + 3 * positions[:, 1:-2]
             - positions[:, :-3])
    return third.norm(dim=-1).mean((1, 2)) / DT ** 3


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
    if weights.foot_height or weights.foot_skate:
        feet = [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]
        world = (relative_positions(pred)[..., feet, :] + out.root[:, sl].unsqueeze(-2))
        world_target = (relative_positions(tgt)[..., feet, :]
                        + batch.root[:, 2:][:, sl].unsqueeze(-2))
        terms["foot_height"] = (world[..., 1] - world_target[..., 1]).abs().mean()
        before = (relative_positions(prev)[..., feet, :]
                  + out.previous_root[:, sl].unsqueeze(-2))
        ground = ((world - before)[..., [0, 2]].norm(dim=-1) / DT)  # (B, T, 4)
        planted = batch.contacts[:, 2:][:, sl] * batch.contacts[:, 1:-1][:, sl]
        terms["foot_skate"] = (ground * planted).sum() / planted.sum().clamp_min(1.0)
    if out.kl is not None:
        per_dim = out.kl[:, sl].mean((0, 1))  # (latent,) nats per tick
        terms["kl"] = per_dim.clamp_min(weights.kl_free_bits).sum()
        terms["kl_raw"] = per_dim.sum().detach()  # reported only (weight 0)
    terms["total"] = sum(getattr(weights, k) * v for k, v in terms.items()  # type: ignore[assignment]
                         if k not in FREE_RUN_TERMS)
    return terms
