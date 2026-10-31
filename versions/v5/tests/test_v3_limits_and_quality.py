"""v3: the quality filter, the data-measured joint limits and the curriculum's windows."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

from motionlab import m1, rotation, skeleton
from motionlab.datasets import quality
from motionlab.formats import Clip

ROOT = Path(__file__).resolve().parents[1]


def _clip(frames: int, source: str = "cmu", subject: str = "cmu/01") -> Clip:
    eye = rotation.matrix_to_rot6d(torch.eye(3)).expand(frames, skeleton.J, 6).clone()
    return Clip(eye, torch.zeros(frames, 3), source, subject,
                contacts=torch.zeros(frames, 4, dtype=torch.bool))


def _turn(degrees: float) -> torch.Tensor:
    return rotation.so3_exp(torch.tensor([0.0, 0.0, math.radians(degrees)]))


def test_pieces_keep_away_from_broken_frames_and_drop_short_runs() -> None:
    bad = torch.zeros(100, dtype=torch.bool)
    bad[50] = True
    runs, dropped = quality.pieces(bad, margin=5, min_frames=30)
    assert runs == [(0, 45), (56, 100)] and dropped == 0
    runs, dropped = quality.pieces(bad, margin=5, min_frames=46)
    assert runs == [] and dropped == 2


def test_clean_cuts_flipped_frames_and_drops_duplicate_takes() -> None:
    good = _clip(400)
    broken = _clip(400, subject="cmu/02")
    broken.rot6d[200:210, skeleton.INDEX["rightHand"]] = rotation.matrix_to_rot6d(_turn(179))
    copy = _clip(400, subject="cmu/03")
    clips, report = quality.clean([("cmu/01_01.pt", good), ("cmu/03_01.pt", copy),
                                   ("cmu/02_01.pt", broken)], margin=10, min_frames=180)
    assert report.duplicates == ["cmu/03_01.pt"]
    assert report.broken_frames == 10 and report.broken_by_joint == {"rightHand": 10}
    # The flipped clip is cut into [0, 190) and [220, 400); 180 frames is long enough.
    assert sorted(c.frames for c in clips) == [180, 190, 400]
    assert report.frames_out == 770


def test_sanity_limits_are_generous_and_leave_the_root_free() -> None:
    assert quality.SANITY_DEG[0] == 180.0
    assert quality.SANITY_DEG[skeleton.INDEX["leftHand"]] == 120.0
    assert quality.SANITY_DEG[skeleton.INDEX["leftLowerArm"]] == 180.0


def test_limit_excess_is_zero_inside_and_grows_outside() -> None:
    limits = m1.joint_limits()
    j = skeleton.INDEX["leftToes"]
    rest = torch.eye(3).expand(skeleton.J, 3, 3)
    assert float(m1.limit_excess(rest, limits).abs().max()) == 0.0
    poses = []
    for degrees in (10.0, 100.0, 120.0):
        r = rest.clone()
        r[j] = rotation.so3_exp(torch.tensor([math.radians(degrees), 0.0, 0.0]))
        poses.append(m1.limit_excess(r, limits)[j])
    assert float(poses[0]) == 0.0 and 0 < float(poses[1]) < float(poses[2])
    # Measured symmetric: each side's limits are the other's, mirrored.
    left, right = skeleton.INDEX["leftHand"], skeleton.INDEX["rightHand"]
    flip = torch.tensor([1.0, -1.0, -1.0])
    lo = torch.minimum(limits.box_min[left] * flip, limits.box_max[left] * flip)
    assert torch.allclose(lo, limits.box_min[right], atol=1e-4)


def test_limit_excess_has_a_finite_gradient_past_the_limit() -> None:
    limits = m1.joint_limits().without_hips()
    w = torch.zeros(4, skeleton.J - 1, 3)
    w[:, skeleton.INDEX["leftToes"] - 1, 0] = math.radians(110)
    w.requires_grad_(True)
    excess = m1.limit_excess(rotation.so3_exp(w), limits).sum()
    excess.backward()
    assert w.grad is not None and torch.isfinite(w.grad).all() and float(w.grad.abs().sum()) > 0


def test_curriculum_phases_carry_their_window() -> None:
    spec = importlib.util.spec_from_file_location("train_m1", ROOT / "scripts" / "train_m1.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("train_m1", module)
    spec.loader.exec_module(module)
    curriculum = [[0.0, 1, 72], [0.5, 32], [0.8, 128, 168]]
    assert module.phase_at(curriculum, 0.1, 72) == (1, 72)
    assert module.phase_at(curriculum, 0.6, 72) == (32, 72)
    assert module.phase_at(curriculum, 1.0, 72) == (128, 168)
