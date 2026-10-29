"""ARDY's 27-joint Core skeleton → motion-lab's 22-joint canonical skeleton, at 60 Hz.

Both are T-poses with world-aligned joint frames and the same axes (+Y up, facing +Z, her left
+X; ARDY's ``CoreSkeleton27`` neutral joints show it, and :func:`check_rest` tests it). So each
canonical joint takes the **global** rotation of its ARDY joint, and its local rotation follows
from its canonical parent's. That folds ARDY's four spine joints into our three
(``spine ← Spine``, ``chest ← Spine2``, ``upperChest ← Spine3``) without losing the total bend,
and drops the hand-end and thumb joints.

The root is scaled by her standing hips height over ARDY's (``ROOT_SCALE``). 20 fps becomes
60 Hz by spherical interpolation of rotations and linear interpolation of the root. Contacts take
the nearest ARDY frame.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# canonical joint ← ARDY Core joint (motion-lab versions/v4/motionlab/skeleton.py order)
MAP = (
    ("hips", "Hips"), ("spine", "Spine"), ("chest", "Spine2"), ("upperChest", "Spine3"),
    ("neck", "Neck"), ("head", "Head"),
    ("leftShoulder", "LeftShoulder"), ("leftUpperArm", "LeftArm"),
    ("leftLowerArm", "LeftForeArm"), ("leftHand", "LeftHand"),
    ("rightShoulder", "RightShoulder"), ("rightUpperArm", "RightArm"),
    ("rightLowerArm", "RightForeArm"), ("rightHand", "RightHand"),
    ("leftUpperLeg", "LeftUpLeg"), ("leftLowerLeg", "LeftLeg"), ("leftFoot", "LeftFoot"),
    ("leftToes", "LeftToeBase"),
    ("rightUpperLeg", "RightUpLeg"), ("rightLowerLeg", "RightLeg"), ("rightFoot", "RightFoot"),
    ("rightToes", "RightToeBase"),
)
CANONICAL_PARENTS = (-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20)
HER_HIPS = 0.894626  # canonical rest hips height (canonical_skeleton.json)
# ARDY Core's standing hips height, measured from its own "a person stands still" generations
# (evaluate.py --calibrate, 2026-10-26: median 0.944 m over three 20 s seeds, 0.918 to 0.953);
# her hips over this scales ARDY's root travel and height.
ARDY_STANDING_HIPS = 0.944
ROOT_SCALE = HER_HIPS / ARDY_STANDING_HIPS
CONTACT_ORDER = (1, 0, 3, 2)  # ARDY [R foot, R toe, L foot, L toe] → ours [L foot, L toe, R...]


def ardy_index(names: list[str]) -> list[int]:
    return [names.index(ardy) for _, ardy in MAP]


def to_canonical(global_rot: np.ndarray, root: np.ndarray, contacts: np.ndarray,
                 names: list[str], contact_names: list[str] | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ARDY ``global_rot_mats (T, 27, 3, 3)``, ``root_positions (T, 3)``,
    ``foot_contacts (T, 4)`` → canonical local rotations ``(T, 22, 3, 3)``, root and contacts,
    still at ARDY's frame rate."""
    glob = global_rot[:, ardy_index(names)]
    local = np.empty_like(glob)
    for j, parent in enumerate(CANONICAL_PARENTS):
        local[:, j] = glob[:, j] if parent < 0 else \
            np.swapaxes(glob[:, parent], -1, -2) @ glob[:, j]
    order = CONTACT_ORDER if contact_names is None else \
        [contact_names.index(n) for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
    return local, root * ROOT_SCALE, contacts[:, list(order)] > 0.5


def resample(local: np.ndarray, root: np.ndarray, contacts: np.ndarray, fps: int,
             target: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample to ``target`` Hz (rotations by slerp, root linearly)."""
    frames = local.shape[0]
    t = np.arange(frames) / fps
    new_t = np.arange(int((frames - 1) * target / fps) + 1) / target
    out = np.empty((len(new_t), local.shape[1], 3, 3))
    for j in range(local.shape[1]):
        out[:, j] = Slerp(t, Rotation.from_matrix(local[:, j]))(new_t).as_matrix()
    root_out = np.stack([np.interp(new_t, t, root[:, a]) for a in range(3)], -1)
    nearest = np.clip(np.round(new_t * fps).astype(int), 0, frames - 1)
    return out, root_out, contacts[nearest]


def canonical_globals(local: np.ndarray) -> np.ndarray:
    """Canonical local rotations ``(T, 22, 3, 3)`` → global rotations."""
    glob = np.empty_like(local)
    for j, parent in enumerate(CANONICAL_PARENTS):
        glob[:, j] = local[:, j] if parent < 0 else glob[:, parent] @ local[:, j]
    return glob


def from_canonical(local: np.ndarray, root: np.ndarray, names: list[str],
                   parents: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The reverse of :func:`to_canonical`: canonical local rotations ``(T, 22, 3, 3)`` and root
    ``(T, 3)`` (her scale) → ARDY Core local rotations ``(T, 27, 3, 3)`` and root (ARDY's scale).

    Mapped joints take the canonical joint's global rotation. ARDY's extra spine joint (Spine1,
    between Spine and Spine2) takes halfway between their globals, so the bend is shared; the
    hand-end and thumb joints stay at rest relative to the hand.
    """
    glob_c = canonical_globals(local)
    frames = local.shape[0]
    glob = np.empty((frames, len(names), 3, 3))
    filled = np.zeros(len(names), dtype=bool)
    for c, (_, ardy) in enumerate(MAP):
        glob[:, names.index(ardy)] = glob_c[:, c]
        filled[names.index(ardy)] = True
    spine, spine1, spine2 = (names.index(n) for n in ("Spine", "Spine1", "Spine2"))
    for t in range(frames):
        pair = Rotation.from_matrix(np.stack((glob[t, spine], glob[t, spine2])))
        glob[t, spine1] = Slerp([0.0, 1.0], pair)(0.5).as_matrix()
    filled[spine1] = True
    for j, parent in enumerate(parents):  # anything else (hand ends, thumbs): rest under parent
        if not filled[j]:
            glob[:, j] = glob[:, parent]
    out = np.empty_like(glob)
    for j, parent in enumerate(parents):
        out[:, j] = glob[:, j] if parent < 0 else np.swapaxes(glob[:, parent], -1, -2) @ glob[:, j]
    return out, root / ROOT_SCALE


def check_rest(skeleton) -> float:
    """Largest gap (metres) between ARDY's FK with identity local rotations and its neutral
    joints: 0 means identity is its T-pose, so its joint frames are world-aligned at rest."""
    import torch

    identity = torch.eye(3).expand(1, skeleton.nbjoints, 3, 3).contiguous()
    out = skeleton.fk(identity, torch.zeros(1, 3))
    positions = out[1] if isinstance(out, (tuple, list)) else out
    return float((positions[0] - skeleton.neutral_joints).abs().max())
