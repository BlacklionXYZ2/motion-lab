"""Forward kinematics on the canonical skeleton (roadmap M0).

Local rotations are relative to world-aligned rest frames (``skeleton`` module), so:

- the hips' rotation is her world orientation, and her root position is the hips' position;
- ``W_j = W_parent · R_j`` and ``p_j = p_parent + W_parent · offset_j``.

With every rotation identity, FK reproduces the rest pose.
"""

from __future__ import annotations

import torch
from torch import Tensor

from motionlab.skeleton import Skeleton


def forward_kinematics(
    local_rotations: Tensor, root_position: Tensor, skel: Skeleton,
) -> tuple[Tensor, Tensor]:
    """``(..., J, 3, 3)`` local rotations and ``(..., 3)`` root position →
    world rotations ``(..., J, 3, 3)`` and world positions ``(..., J, 3)``."""
    offsets = skel.offsets.to(dtype=local_rotations.dtype, device=local_rotations.device)
    world_rot: list[Tensor] = []
    world_pos: list[Tensor] = []
    for j, parent in enumerate(skel.parents):
        r = local_rotations[..., j, :, :]
        if parent < 0:
            world_rot.append(r)
            world_pos.append(root_position)
        else:
            world_rot.append(world_rot[parent] @ r)
            world_pos.append(world_pos[parent] + (world_rot[parent] @ offsets[j].unsqueeze(-1))
                             .squeeze(-1))
    return torch.stack(world_rot, -3), torch.stack(world_pos, -2)
