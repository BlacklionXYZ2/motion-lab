"""M0 gate: the canonical skeleton table and forward kinematics."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from motionlab import rotation as rot
from motionlab import skeleton
from motionlab.fk import forward_kinematics
from scripts.build_skeleton import DEFAULT_VRM, build

SKEL = skeleton.load()


def test_table_is_ordered_and_matches_the_asset() -> None:
    assert skeleton.J == 22 and SKEL.names == skeleton.NAMES
    assert all(parent < j for j, parent in enumerate(skeleton.PARENTS))
    assert skeleton.PARENTS[0] == -1 and skeleton.PARENTS.count(-1) == 1
    assert SKEL.offsets.shape == (22, 3)


def test_mirror_map_pairs_left_with_right() -> None:
    for j, m in enumerate(skeleton.MIRROR):
        assert skeleton.MIRROR[m] == j
        a, b = skeleton.NAMES[j], skeleton.NAMES[m]
        assert a.replace("left", "").replace("right", "") == b.replace("left", "").replace(
            "right", "")


def test_rest_pose_is_symmetric_grounded_and_faces_plus_z() -> None:
    rest = SKEL.rest_positions
    mirrored = rest * torch.tensor([-1.0, 1.0, 1.0])
    assert float((mirrored - rest[list(skeleton.MIRROR)]).abs().max()) < 1e-4  # 0.1 mm
    contacts = [skeleton.INDEX[n] for n in skeleton.CONTACT_JOINTS]
    assert abs(float(rest[contacts, 1].min())) < 1e-5  # feet on the ground (y = 0)
    left_toes, left_foot = rest[skeleton.INDEX["leftToes"]], rest[skeleton.INDEX["leftFoot"]]
    assert left_toes[2] > left_foot[2]  # toes in front: she faces +Z
    assert rest[skeleton.INDEX["leftHand"], 0] > 0  # her left is +X


@pytest.mark.skipif(not DEFAULT_VRM.is_file(), reason="her VRM isn't beside motion-lab")
def test_asset_is_current_for_her_vrm() -> None:
    fresh = build(DEFAULT_VRM)
    assert fresh["source"]["vrm_sha256"] == SKEL.source["vrm_sha256"]  # type: ignore[index]
    assert torch.allclose(torch.tensor(fresh["offsets"]), SKEL.offsets, atol=1e-6)


def test_fk_rest_pose_reproduces_rest_positions() -> None:
    eye = torch.eye(3).repeat(skeleton.J, 1, 1)
    _, positions = forward_kinematics(eye, SKEL.offsets[0], SKEL)
    assert torch.allclose(positions, SKEL.rest_positions, atol=1e-6)


def test_fk_against_a_hand_computed_chain() -> None:
    # A three-joint chain along +X: root at the origin, then 1 m, then 1 m.
    tiny = replace(SKEL, names=("a", "b", "c"), parents=(-1, 0, 1),
                   offsets=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]]))
    quarter_z = rot.so3_exp(torch.tensor([0.0, 0.0, math.pi / 2]))
    local = torch.stack((torch.eye(3), quarter_z, torch.eye(3)))
    world, pos = forward_kinematics(local, torch.zeros(3), tiny)
    # b turns 90° about +Z, so c sits 1 m from b along +Y: (1, 1, 0).
    assert torch.allclose(pos, torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]]), atol=1e-6)
    assert torch.allclose(world[2], quarter_z, atol=1e-6)


def test_fk_keeps_bone_lengths_under_any_rotations() -> None:
    gen = torch.Generator().manual_seed(3)
    local = rot.so3_exp(torch.randn(32, skeleton.J, 3, generator=gen))
    _, pos = forward_kinematics(local, torch.zeros(32, 3), SKEL)
    for j, parent in enumerate(skeleton.PARENTS):
        if parent >= 0:
            length = (pos[:, j] - pos[:, parent]).norm(dim=-1)
            assert torch.allclose(length, SKEL.offsets[j].norm().expand(32), atol=1e-5)


def test_limits_are_placeholders_with_a_value_for_every_joint() -> None:
    assert SKEL.max_angle_deg[0] is None  # the root's rotation is her facing
    assert all(a is not None and 0 < a <= 180 for a in SKEL.max_angle_deg[1:])
    assert bool((SKEL.omega_max > 0).all())
