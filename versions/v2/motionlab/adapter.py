"""Canonical skeleton → a particular avatar's skeleton (the output side; roadmap rule 3).

The model always produces all 22 canonical joints. An avatar may lack optional VRM joints
(Seed-san has no ``upperChest``; other avatars lack toes, a neck or shoulders). Each missing joint
is folded away by composing rotations, so every joint the avatar has keeps exactly the world
orientation the model meant:

- a missing **leaf** (toes) is dropped: its bend moves nothing else;
- a missing joint that is its parent's **only child** folds **into the parent**
  (``chest ← chest · upperChest``), so the parent takes on the folded joint's orientation and
  everything hanging from it is unchanged;
- a missing joint **with siblings and one child** folds **into that child**
  (``head ← neck · head``; ``upperArm ← shoulder · upperArm``).

Positions shift only by the folded joint's own bend acting on a short bone, ``|(R − I)·offset|``:
about 3.5 cm at the neck base for a 20° ``upperChest`` bend on Seed-san, and zero when it is
straight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from motionlab import skeleton
from motionlab.skeleton import Skeleton
from motionlab.vrm import read_skeleton

REQUIRED = tuple(n for n in skeleton.NAMES if n not in skeleton.OPTIONAL_JOINTS)
_CHILDREN: dict[str, list[str]] = {n: [] for n in skeleton.NAMES}
for _name, _parent in skeleton.JOINTS:
    if _parent is not None:
        _CHILDREN[_parent].append(_name)


@dataclass(frozen=True)
class AvatarMap:
    joints: tuple[str, ...]  # the avatar's joints, in canonical order
    compose: tuple[tuple[int, ...], ...]  # canonical joint indices multiplied into each one
    dropped: tuple[str, ...]

    @classmethod
    def for_joints(cls, present: set[str]) -> AvatarMap:
        missing_required = [n for n in REQUIRED if n not in present]
        if missing_required:
            raise ValueError(f"an avatar must have the VRM's required joints: {missing_required}")
        lists: dict[str, list[str]] = {n: [n] for n in skeleton.NAMES if n in present}
        pending: dict[str, list[str]] = {}  # child → missing joints to put in front of it
        dropped: list[str] = []
        target: dict[str, str] = {}
        for name, parent in skeleton.JOINTS:
            if name in present:
                target[name] = name
                lists[name] = pending.pop(name, []) + lists[name]
                continue
            children = _CHILDREN[name]
            if not children:
                dropped.append(name)  # a leaf (toes): its bend has nothing to move
            elif parent is not None and _CHILDREN[parent] == [name] and parent in target:
                owner = target[parent]
                lists[owner].append(name)
                target[name] = owner
            elif len(children) == 1:
                pending[children[0]] = [*pending.pop(name, []), name]
            else:
                raise ValueError(f"can't fold away {name}: it has several children")
        joints = tuple(n for n in skeleton.NAMES if n in present)
        return cls(joints, tuple(tuple(skeleton.INDEX[c] for c in lists[n]) for n in joints),
                   tuple(dropped))

    def to_avatar(self, local_rotations: Tensor) -> Tensor:
        """``(..., 22, 3, 3)`` canonical local rotations → ``(..., A, 3, 3)`` for the avatar."""
        out = []
        for chain in self.compose:
            r = local_rotations[..., chain[0], :, :]
            for index in chain[1:]:
                r = r @ local_rotations[..., index, :, :]
            out.append(r)
        return torch.stack(out, -3)


def avatar_skeleton(vrm_path: Path, dtype: torch.dtype = torch.float32) -> tuple[Skeleton,
                                                                                  AvatarMap]:
    """The avatar's own skeleton (its real rest offsets) and the map onto it."""
    vrm = read_skeleton(vrm_path)
    present = {n for n in skeleton.NAMES if n in vrm.bone_nodes}
    mapping = AvatarMap.for_joints(present)
    ground = min(vrm.world_positions[n][1] for n in skeleton.CONTACT_JOINTS if n in present)
    pos = {n: vrm.world_positions[n] - np.array([0.0, ground, 0.0]) for n in mapping.joints}
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    for name in mapping.joints:
        parent = dict(skeleton.JOINTS)[name]
        while parent is not None and parent not in present:
            parent = dict(skeleton.JOINTS)[parent]
        parents.append(-1 if parent is None else mapping.joints.index(parent))
        offsets.append(pos[name] if parent is None else pos[name] - pos[parent])
    canonical = skeleton.load(dtype)
    keep = [skeleton.INDEX[n] for n in mapping.joints]
    return Skeleton(
        names=mapping.joints, parents=tuple(parents),
        offsets=torch.tensor(np.array(offsets), dtype=dtype),
        max_angle_deg=tuple(canonical.max_angle_deg[i] for i in keep),
        omega_max=canonical.omega_max[keep],
        source={"vrm": vrm_path.name},
    ), mapping
