"""Data formats: clips, windows and the per-frame motion state (roadmap M0).

Units: metres, radians, seconds. Tick: 60 Hz (LAFAN1's 30 fps and AMASS's various rates are
resampled to it in M1). Joint order and rest frames: ``skeleton``.

A **clip** stores local joint rotations as 6D, plus the root (hips) position. The **motion state**
the network sees each tick is derived from the realised pose:

- every joint's local rotation as 6D (J × 6);
- every joint's body-frame angular velocity ω (J × 3), finite-differenced from the *realised*
  previous pose (roadmap rule 5), never from the model's own output;
- the root's velocity expressed in her heading frame (3): sideways, up, forward. Her facing is the
  hips' forward direction projected onto the ground, so the state doesn't depend on where she is
  or which way the world faces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from motionlab import rotation, skeleton

FPS = 60
DT = 1.0 / FPS
WINDOW = 120  # 2 s
STRIDE = 30  # 0.5 s
STATE_LAYOUT = {
    "rot6d": (0, skeleton.J * 6),
    "omega": (skeleton.J * 6, skeleton.J * 9),
    "root_velocity": (skeleton.J * 9, skeleton.J * 9 + 3),
}
STATE_SIZE = skeleton.J * 9 + 3


@dataclass
class Clip:
    rot6d: Tensor  # (T, J, 6) local rotations; joint 0 is her world orientation
    root_position: Tensor  # (T, 3) metres
    source: str = ""
    subject: str = ""  # splits are by subject, so no person appears in train and test
    fps: int = FPS
    intent: Tensor | None = None  # (T,) indices into motion_signal.INTENTS, when labelled
    contacts: Tensor | None = None  # (T, 4) bool for skeleton.CONTACT_JOINTS
    notes: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        t = self.rot6d.shape[0]
        if self.fps != FPS:
            raise ValueError(f"clips are stored at {FPS} Hz, not {self.fps}")
        if self.rot6d.shape != (t, skeleton.J, 6) or self.root_position.shape != (t, 3):
            raise ValueError("rot6d must be (T, J, 6) and root_position (T, 3)")
        if not (torch.isfinite(self.rot6d).all() and torch.isfinite(self.root_position).all()):
            raise ValueError("non-finite values in a clip")
        columns = self.rot6d.view(t, skeleton.J, 2, 3).norm(dim=-1)
        if bool((columns < 1e-6).any()):
            raise ValueError("a 6D rotation has a zero column")
        for name, labels, width in (("intent", self.intent, None), ("contacts", self.contacts, 4)):
            if labels is not None and (labels.shape[0] != t
                                       or (width and labels.shape[1:] != (width,))):
                raise ValueError(f"{name} labels don't match the clip length")

    @property
    def frames(self) -> int:
        return int(self.rot6d.shape[0])


def windows(frames: int, length: int = WINDOW, stride: int = STRIDE) -> list[tuple[int, int]]:
    """``[start, end)`` training windows over a clip of ``frames`` frames."""
    if frames < length:
        return []
    return [(s, s + length) for s in range(0, frames - length + 1, stride)]


def heading(world_root: Tensor) -> Tensor:
    """Yaw angle (radians) of her facing: the hips' +Z axis projected onto the ground."""
    forward = world_root[..., :, 2]
    return torch.atan2(forward[..., 0], forward[..., 2])


def yaw_matrix(angle: Tensor) -> Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    o, i = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack((torch.stack((c, o, s), -1), torch.stack((o, i, o), -1),
                        torch.stack((-s, o, c), -1)), -2)


def motion_state(clip: Clip) -> Tensor:
    """``(T−1, STATE_SIZE)`` states for frames 1…T−1 (frame 0 has no previous pose)."""
    r = rotation.rot6d_to_matrix(clip.rot6d)  # (T, J, 3, 3)
    omega = rotation.angular_velocity(r[:-1], r[1:], DT)  # (T−1, J, 3)
    velocity = (clip.root_position[1:] - clip.root_position[:-1]) / DT  # world, (T−1, 3)
    yaw = heading(r[1:, 0])
    local_velocity = (yaw_matrix(yaw).transpose(-1, -2) @ velocity.unsqueeze(-1)).squeeze(-1)
    return torch.cat((clip.rot6d[1:].flatten(1), omega.flatten(1), local_velocity), -1)


def mirror_clip(clip: Clip) -> Clip:
    """Left/right mirror image (augmentation): swap partner joints and reflect x."""
    r = rotation.mirror_rotation(rotation.rot6d_to_matrix(clip.rot6d))[:, list(skeleton.MIRROR)]
    position = clip.root_position * torch.tensor([-1.0, 1.0, 1.0], dtype=clip.root_position.dtype)
    contacts = None
    if clip.contacts is not None:
        contacts = clip.contacts[:, [2, 3, 0, 1]]  # left foot/toes ↔ right foot/toes
    return Clip(rotation.matrix_to_rot6d(r), position, clip.source, clip.subject, clip.fps,
                clip.intent, contacts, {**clip.notes, "mirrored": "true"})


def constant_spin_clip(frames: int, omega: tuple[float, float, float], joint: str) -> Clip:
    """A synthetic clip in which one joint spins at a constant body-frame ω (for tests)."""
    r = torch.eye(3).repeat(frames, skeleton.J, 1, 1)
    w = torch.tensor(omega)
    for t in range(1, frames):
        r[t, skeleton.INDEX[joint]] = rotation.so3_exp(w * DT * t)
    root = torch.zeros(frames, 3)
    root[:, 1] = float(skeleton.load().offsets[0, 1])
    return Clip(rotation.matrix_to_rot6d(r), root, source="synthetic")
