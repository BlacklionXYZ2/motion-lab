"""M0 gate: rotation maths (round trips, Exp/Log, long integration, ω, gradients, mirroring)."""

from __future__ import annotations

import math

import pytest
import torch

from motionlab import rotation as rot

GEN = torch.Generator().manual_seed(7)


def random_rotations(n: int, dtype: torch.dtype = torch.float32, max_angle: float = math.pi
                     ) -> torch.Tensor:
    axis = torch.randn(n, 3, generator=GEN, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.rand(n, 1, generator=GEN, dtype=torch.float64) * max_angle
    return rot.so3_exp(axis * angle).to(dtype)


def test_6d_round_trip_within_gate_tolerance() -> None:
    r = random_rotations(10_000)
    back = rot.rot6d_to_matrix(rot.matrix_to_rot6d(r))
    assert float((back - r).abs().max()) < 1e-5
    six = rot.matrix_to_rot6d(r)
    assert float((rot.matrix_to_rot6d(rot.rot6d_to_matrix(six)) - six).abs().max()) < 1e-5


def test_any_6d_vector_becomes_a_valid_rotation() -> None:
    six = torch.randn(10_000, 6, generator=GEN)
    assert float(rot.orthonormality_error(rot.rot6d_to_matrix(six)).max()) < 1e-5


def test_6d_layout_is_the_first_two_columns() -> None:
    r = random_rotations(1)[0]
    assert torch.equal(rot.matrix_to_rot6d(r), torch.cat((r[:, 0], r[:, 1])))


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_exp_and_log_are_inverses(dtype: torch.dtype, tolerance: float) -> None:
    r = random_rotations(10_000, dtype, max_angle=math.pi - 0.01)
    assert float((rot.so3_exp(rot.so3_log(r)) - r).abs().max()) < tolerance


def test_log_handles_tiny_and_near_pi_angles() -> None:
    axis = torch.tensor([[0.0, 0.6, 0.8]], dtype=torch.float64)
    for angle in (0.0, 1e-7, 1e-5, math.pi - 1e-6, math.pi - 1e-4, math.pi):
        w = rot.so3_log(rot.so3_exp(axis * angle))
        assert torch.isfinite(w).all()
        assert abs(float(w.norm()) - angle) < 1e-6
        assert float((rot.so3_exp(w) - rot.so3_exp(axis * angle)).abs().max()) < 1e-6


def test_100k_integration_steps_stay_on_so3_in_float32() -> None:
    r = torch.eye(3).repeat(64, 1, 1)
    omega = torch.randn(64, 3, generator=GEN) * 3  # rad/s
    for _ in range(100_000 // 100):  # 100 ticks per call keeps the test fast
        for _ in range(100):
            r = rot.integrate(r, omega, 1 / 60)
    assert float(rot.orthonormality_error(r).max()) < 1e-4


def test_unnormalised_integration_is_what_normalisation_protects_against() -> None:
    r = torch.eye(3).repeat(64, 1, 1)
    omega = torch.randn(64, 3, generator=GEN) * 3
    for _ in range(20_000):
        r = rot.integrate(r, omega, 1 / 60, normalize=False)
    # Recorded, not gated: shows why integrate() re-projects by default.
    assert float(rot.orthonormality_error(r).max()) > 1e-6


def test_angular_velocity_inverts_integration() -> None:
    r0 = random_rotations(1000)
    omega = torch.randn(1000, 3, generator=GEN) * 5
    r1 = rot.integrate(r0, omega, 1 / 60)
    assert float((rot.angular_velocity(r0, r1, 1 / 60) - omega).abs().max()) < 1e-2  # rad/s


def test_geodesic_loss_has_finite_gradients_at_identity_and_near_pi() -> None:
    for angle in (0.0, 1e-6, math.pi - 1e-3):
        w = torch.tensor([[0.0, 0.0, angle]], requires_grad=True)
        loss = rot.geodesic(rot.so3_exp(w), torch.eye(3)[None]).sum()
        loss.backward()
        assert w.grad is not None and torch.isfinite(w.grad).all(), angle


def test_6d_gradients_are_finite() -> None:
    six = torch.randn(256, 6, generator=GEN, requires_grad=True)
    loss = rot.geodesic(rot.rot6d_to_matrix(six), random_rotations(256)).mean()
    loss.backward()
    assert six.grad is not None and torch.isfinite(six.grad).all()


def test_mirroring_is_an_involution_and_keeps_proper_rotations() -> None:
    r = random_rotations(1000)
    mirrored = rot.mirror_rotation(r)
    assert float((rot.mirror_rotation(mirrored) - r).abs().max()) < 1e-6
    assert float((torch.linalg.det(mirrored) - 1).abs().max()) < 1e-5
    # A turn to her left (about +Y) becomes a turn to her right.
    w = rot.so3_log(rot.mirror_rotation(rot.so3_exp(torch.tensor([0.0, 0.5, 0.0]))))
    assert torch.allclose(w, torch.tensor([0.0, -0.5, 0.0]), atol=1e-6)
