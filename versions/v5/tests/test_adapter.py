"""Canonical 22 joints → an avatar that lacks some optional joints."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from motionlab import rotation as rot
from motionlab import skeleton
from motionlab.adapter import AvatarMap, avatar_skeleton
from motionlab.fk import forward_kinematics
from scripts.build_skeleton import DEFAULT_VRM, SEED_SAN, build

SKEL = skeleton.load()
GEN = torch.Generator().manual_seed(5)


def random_local(batch: int, bend_degrees: float = 25.0) -> torch.Tensor:
    w = torch.randn(batch, skeleton.J, 3, generator=GEN)
    w = w / w.norm(dim=-1, keepdim=True) * torch.rand(batch, skeleton.J, 1, generator=GEN)
    return rot.so3_exp(w * math.radians(bend_degrees))


def test_full_avatar_is_the_identity_map() -> None:
    mapping = AvatarMap.for_joints(set(skeleton.NAMES))
    local = random_local(4)
    assert mapping.joints == skeleton.NAMES and not mapping.dropped
    assert torch.equal(mapping.to_avatar(local), local)


def test_fold_rules() -> None:
    present = set(skeleton.NAMES) - {"upperChest", "neck", "leftShoulder", "rightToes"}
    mapping = AvatarMap.for_joints(present)
    chains = {n: [skeleton.NAMES[i] for i in c]
              for n, c in zip(mapping.joints, mapping.compose, strict=True)}
    assert chains["chest"] == ["chest", "upperChest"]  # only child: into the parent
    assert chains["head"] == ["neck", "head"]  # has siblings: into its child
    assert chains["leftUpperArm"] == ["leftShoulder", "leftUpperArm"]
    assert mapping.dropped == ("rightToes",)  # a leaf: dropped
    with pytest.raises(ValueError, match="required"):
        AvatarMap.for_joints(set(skeleton.NAMES) - {"head"})


@pytest.mark.parametrize("vrm", [DEFAULT_VRM, SEED_SAN], ids=["alice", "seed-san"])
def test_an_avatar_without_upper_chest_keeps_orientations_and_nearly_positions(vrm) -> None:
    if not vrm.is_file():
        pytest.skip(f"{vrm.name} isn't beside motion-lab")
    avatar, mapping = avatar_skeleton(vrm)
    assert len(mapping.joints) == 21 and "upperChest" not in mapping.joints
    # The canonical skeleton at this avatar's own proportions, so only the fold differs.
    canon = replace(SKEL, offsets=torch.tensor(build(vrm)["offsets"]))
    root = canon.offsets[0].expand(64, 3)
    local = random_local(64, bend_degrees=20)
    canon_rot, canon_pos = forward_kinematics(local, root, canon)
    avatar_rot, avatar_pos = forward_kinematics(mapping.to_avatar(local), root, avatar)
    keep = [skeleton.INDEX[n] for n in mapping.joints]
    # Every avatar joint takes the orientation of the last canonical joint folded into it
    # (Seed-san's chest carries upperChest's), so everything hanging from it is exact.
    last = [chain[-1] for chain in mapping.compose]
    assert torch.allclose(avatar_rot, canon_rot[:, last], atol=1e-5)
    # Positions shift only by the upperChest bend on a short bone (≤ 20° here).
    assert float((avatar_pos - canon_pos[:, keep]).norm(dim=-1).max()) < 0.04
    # And not at all when upperChest is straight.
    local[:, skeleton.INDEX["upperChest"]] = torch.eye(3)
    _, canon_pos = forward_kinematics(local, root, canon)
    _, avatar_pos = forward_kinematics(mapping.to_avatar(local), root, avatar)
    assert float((avatar_pos - canon_pos[:, keep]).abs().max()) < 1e-5
