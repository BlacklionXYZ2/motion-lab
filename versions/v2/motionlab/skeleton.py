"""The canonical skeleton: the single definition every module imports (roadmap M0).

It is the **full VRM 1.0 humanoid body: 22 joints**, with no fingers and no eyes (gaze stays
deterministic IK). That is also the body of the training data: AMASS (SMPL) and LAFAN1 both have a
hips → three-spine → neck → head chain with collars, so every dataset maps onto it one to one and
nothing is merged in training (owner, 2026-10-11). An avatar without some optional joints (Seed-san
has no ``upperChest``) is handled at the output by ``adapter.py``.

**Rest frames are world-aligned.** In the rest pose every joint's local rotation is identity, and
a joint's offset is the vector from its parent's rest position in world axes (metres; +Y up,
facing +Z, her left = +X). For ``hips`` the "offset" is its rest position above the ground. The
proportions are her body's: ``scripts/build_skeleton.py`` reads them from her VRM into
``assets/canonical_skeleton.json`` (recording its SHA-256). Joints her VRM lacks are placed by a
measured rule recorded in the asset (``upperChest``: 44% of the way from chest to neck, as in a
reference VRM that has one).

Joint limits and ω_max are **placeholders** until M1 replaces them with dataset percentiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
from torch import Tensor

# (name, parent) in an order where every parent comes before its children.
JOINTS: tuple[tuple[str, str | None], ...] = (
    ("hips", None),
    ("spine", "hips"),
    ("chest", "spine"),
    ("upperChest", "chest"),
    ("neck", "upperChest"),
    ("head", "neck"),
    ("leftShoulder", "upperChest"),
    ("leftUpperArm", "leftShoulder"),
    ("leftLowerArm", "leftUpperArm"),
    ("leftHand", "leftLowerArm"),
    ("rightShoulder", "upperChest"),
    ("rightUpperArm", "rightShoulder"),
    ("rightLowerArm", "rightUpperArm"),
    ("rightHand", "rightLowerArm"),
    ("leftUpperLeg", "hips"),
    ("leftLowerLeg", "leftUpperLeg"),
    ("leftFoot", "leftLowerLeg"),
    ("leftToes", "leftFoot"),
    ("rightUpperLeg", "hips"),
    ("rightLowerLeg", "rightUpperLeg"),
    ("rightFoot", "rightLowerLeg"),
    ("rightToes", "rightFoot"),
)
NAMES: tuple[str, ...] = tuple(name for name, _ in JOINTS)
INDEX: dict[str, int] = {name: i for i, name in enumerate(NAMES)}
PARENTS: tuple[int, ...] = tuple(-1 if p is None else INDEX[p] for _, p in JOINTS)
J = len(NAMES)

# Each joint's mirror partner (itself for joints on the centre line).
MIRROR: tuple[int, ...] = tuple(
    INDEX[n.replace("left", "@").replace("right", "left").replace("@", "right")]
    if n.startswith(("left", "right")) else INDEX[n] for n in NAMES
)

CONTACT_JOINTS = ("leftFoot", "leftToes", "rightFoot", "rightToes")
# VRM 1.0 requires the others; an avatar may lack these (adapter.py folds them away).
OPTIONAL_JOINTS = frozenset({"chest", "upperChest", "neck", "leftShoulder", "rightShoulder",
                             "leftToes", "rightToes"})

# Placeholders (degrees from rest, and rad/s); M1 replaces them with dataset percentiles.
_PLACEHOLDER_MAX_ANGLE = {
    "spine": 60, "chest": 45, "upperChest": 45, "neck": 70, "head": 80,
    "Shoulder": 40, "UpperArm": 170, "LowerArm": 150, "Hand": 90,
    "UpperLeg": 130, "LowerLeg": 160, "Foot": 70, "Toes": 60,
}
_PLACEHOLDER_OMEGA_MAX = {"core": 10.0, "limb": 25.0}

ASSET = Path(__file__).with_name("assets") / "canonical_skeleton.json"


def placeholder_max_angle(name: str) -> float | None:
    """Degrees from rest; None for the root (its rotation is her facing)."""
    if name == "hips":
        return None
    for key, value in _PLACEHOLDER_MAX_ANGLE.items():
        if name == key or name.endswith(key):
            return float(value)
    raise KeyError(name)


def placeholder_omega_max(name: str) -> float:
    core = name in {"hips", "spine", "chest", "upperChest", "neck", "head"}
    return _PLACEHOLDER_OMEGA_MAX["core" if core else "limb"]


@dataclass(frozen=True)
class Skeleton:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    offsets: Tensor  # (J, 3) metres; offsets[0] is the hips' rest position
    max_angle_deg: tuple[float | None, ...]
    omega_max: Tensor  # (J,) rad/s
    source: dict[str, str]

    @property
    def rest_positions(self) -> Tensor:
        """(J, 3) world positions in the rest pose (identity rotations)."""
        out = torch.zeros_like(self.offsets)
        for j, parent in enumerate(self.parents):
            out[j] = self.offsets[j] if parent < 0 else out[parent] + self.offsets[j]
        return out


@cache
def load(dtype: torch.dtype = torch.float32) -> Skeleton:
    data = json.loads(ASSET.read_text("utf-8"))
    if tuple(data["names"]) != NAMES or tuple(data["parents"]) != PARENTS:
        raise ValueError("canonical_skeleton.json does not match skeleton.JOINTS; rebuild it")
    return Skeleton(
        names=NAMES,
        parents=PARENTS,
        offsets=torch.tensor(data["offsets"], dtype=dtype),
        max_angle_deg=tuple(data["max_angle_deg"]),
        omega_max=torch.tensor(data["omega_max"], dtype=dtype),
        source=data["source"],
    )
