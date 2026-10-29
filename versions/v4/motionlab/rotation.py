"""Rotation maths for the motion core (roadmap M0).

Conventions (docs/M0.md, fixed for every stage):

- A rotation is a 3×3 matrix acting on column vectors, batched as ``(..., 3, 3)``.
- **6D** is the first two columns of R, concatenated: ``(r00, r10, r20, r01, r11, r21)``.
- **Angular velocity ω is in the body (local) frame:** ``R(t+Δt) = R(t)·Exp(Δt·[ω]×)``, so
  ``ω = Log(R(t)ᵀ·R(t+Δt)) / Δt``. Units: radians per second.
- Angles come from ``atan2`` rather than ``arccos``, so they and their gradients stay finite near
  0 and π.

Everything is PyTorch, batched and differentiable, so training, data processing and evaluation
use the same functions.
"""

from __future__ import annotations

import torch
from torch import Tensor

SMALL_ANGLE = 1e-4  # below this, Taylor series replace sin/θ terms


def hat(w: Tensor) -> Tensor:
    """``(..., 3)`` → the skew-symmetric matrix ``[w]×`` ``(..., 3, 3)``."""
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack(
        (torch.stack((o, -z, y), -1), torch.stack((z, o, -x), -1), torch.stack((-y, x, o), -1)),
        -2,
    )


def vee(s: Tensor) -> Tensor:
    """Inverse of :func:`hat` for a skew-symmetric ``(..., 3, 3)``."""
    return torch.stack((s[..., 2, 1], s[..., 0, 2], s[..., 1, 0]), -1)


def matrix_to_rot6d(r: Tensor) -> Tensor:
    """``(..., 3, 3)`` → ``(..., 6)``: the first two columns."""
    return torch.cat((r[..., :, 0], r[..., :, 1]), -1)


def rot6d_to_matrix(x: Tensor, eps: float = 1e-8) -> Tensor:
    """``(..., 6)`` → a valid rotation ``(..., 3, 3)`` by Gram–Schmidt (Zhou et al. 2019)."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(eps)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(eps)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), -1)


def orthonormalize(r: Tensor) -> Tensor:
    """Project a nearly-rotation matrix back onto SO(3) (through its 6D form)."""
    return rot6d_to_matrix(matrix_to_rot6d(r))


def so3_exp(w: Tensor) -> Tensor:
    """Rodrigues: rotation vector ``(..., 3)`` (axis × angle, radians) → ``(..., 3, 3)``."""
    theta2 = (w * w).sum(-1, keepdim=True).unsqueeze(-1)
    theta = theta2.clamp_min(1e-20).sqrt()  # clamped: a finite gradient at the zero rotation
    small = theta < SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, 1 - theta2 / 6 + theta2 * theta2 / 120, torch.sin(safe) / safe)
    b = torch.where(small, 0.5 - theta2 / 24 + theta2 * theta2 / 720,
                    (1 - torch.cos(safe)) / (safe * safe))
    k = hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(k.shape)
    return eye + a * k + b * (k @ k)


def rotation_angle(r: Tensor) -> Tensor:
    """The angle of each rotation, in ``[0, π]``, computed stably with ``atan2``.

    The squared norm is clamped before the square root, so the gradient at the identity (a
    perfect prediction, the most common case in training) is zero rather than NaN.
    """
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    skew = vee(r - r.transpose(-1, -2))
    sin_part = (skew * skew).sum(-1).clamp_min(1e-20).sqrt() / 2
    return torch.atan2(sin_part, (trace - 1) / 2)


def so3_log(r: Tensor) -> Tensor:
    """``(..., 3, 3)`` → rotation vector ``(..., 3)`` with angle in ``[0, π]``.

    Up to π/2 the axis comes from the skew part (``R − Rᵀ = 2 sinθ [n]×``). Beyond it, sinθ
    shrinks towards 0 and that loses precision, so the axis comes from the symmetric part
    (``(R + Rᵀ)/2 = cosθ I + (1 − cosθ) n nᵀ``) and the skew part only picks the sign. Both
    branches use safe denominators, so neither leaks NaN gradients into the other.
    """
    theta = rotation_angle(r)
    skew = vee(r - r.transpose(-1, -2))  # = 2 sinθ · n
    small = theta < SMALL_ANGLE
    wide = theta > torch.pi / 2

    safe = torch.where(small | wide, torch.ones_like(theta), theta)
    factor = torch.where(small, 0.5 + theta * theta / 12, safe / (2 * torch.sin(safe)))
    w_skew = factor.unsqueeze(-1) * skew

    cos = torch.cos(theta)
    denominator = torch.where(wide, 1 - cos, torch.ones_like(cos))
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    sym = (r + r.transpose(-1, -2)) / 2
    outer = (sym - cos[..., None, None] * eye) / denominator[..., None, None]  # n nᵀ when wide
    diag = torch.diagonal(outer, dim1=-2, dim2=-1)
    column = diag.argmax(-1, keepdim=True)
    picked = torch.gather(outer, -1, column.unsqueeze(-2).expand(*outer.shape[:-1], 1))[..., 0]
    n = picked / torch.gather(diag, -1, column).clamp_min(1e-20).sqrt()
    sign = torch.where((n * skew).sum(-1, keepdim=True) < 0, -1.0, 1.0).to(r.dtype)
    w_sym = sign * n * theta.unsqueeze(-1)

    return torch.where(wide.unsqueeze(-1), w_sym, w_skew)


def integrate(r: Tensor, omega: Tensor, dt: float, *, normalize: bool = True) -> Tensor:
    """One tick: ``R' = R·Exp(Δt·[ω]×)``, re-projected onto SO(3) unless told not to."""
    out = r @ so3_exp(omega * dt)
    return orthonormalize(out) if normalize else out


def angular_velocity(r_prev: Tensor, r_now: Tensor, dt: float) -> Tensor:
    """Body-frame ω that takes ``r_prev`` to ``r_now`` in ``dt`` seconds (inverse of
    :func:`integrate`). Feed it from the *realised* pose, never the model's raw output."""
    return so3_log(r_prev.transpose(-1, -2) @ r_now) / dt


def geodesic(r1: Tensor, r2: Tensor) -> Tensor:
    """Angle between two rotations (radians): the geodesic loss and distance."""
    return rotation_angle(r1.transpose(-1, -2) @ r2)


def orthonormality_error(r: Tensor) -> Tensor:
    """Largest absolute entry of ``RᵀR − I`` per matrix, plus ``|det R − 1|``."""
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    gram = (r.transpose(-1, -2) @ r - eye).abs().amax((-1, -2))
    return gram + (torch.linalg.det(r) - 1).abs()


MIRROR = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))  # reflection across the YZ plane (x → −x)


def mirror_rotation(r: Tensor) -> Tensor:
    """The left/right mirror image of a rotation: ``M·R·M`` with ``M = diag(−1, 1, 1)``."""
    m = MIRROR.to(dtype=r.dtype, device=r.device)
    return m @ r @ m
