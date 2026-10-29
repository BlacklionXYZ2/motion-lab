"""Shared test cases for the avatar map's C# twin (roadmap MA, A2).

Writes ``tests/fixtures/adapter_cases.json``: for avatars missing different optional joints, the
fold lists and the composed rotations for random canonical poses (motion-lab frame, row-major
3×3 matrices). The Unity environment's tests load it and must agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from motionlab import rotation, skeleton
from motionlab.adapter import AvatarMap

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "adapter_cases.json"
MISSING = [
    [],
    ["upperChest"],
    ["upperChest", "leftToes", "rightToes"],
    ["neck"],
    ["leftShoulder", "rightShoulder"],
    ["chest", "upperChest"],
    ["chest", "upperChest", "neck", "leftShoulder", "rightShoulder", "leftToes", "rightToes"],
]


def main() -> None:
    generator = torch.Generator().manual_seed(20261017)
    cases = []
    for missing in MISSING:
        present = {n for n in skeleton.NAMES if n not in missing}
        mapping = AvatarMap.for_joints(present)
        poses = []
        for _ in range(3):
            count = len(skeleton.NAMES)
            omega = torch.randn(count, 3, generator=generator, dtype=torch.float64) * 0.6
            local = rotation.so3_exp(omega)
            avatar = mapping.to_avatar(local)
            poses.append({
                "canonical": local.reshape(len(skeleton.NAMES), 9).tolist(),
                "avatar": avatar.reshape(len(mapping.joints), 9).tolist(),
            })
        cases.append({
            "missing": missing,
            "joints": list(mapping.joints),
            "compose": [[skeleton.NAMES[c] for c in chain] for chain in mapping.compose],
            "dropped": list(mapping.dropped),
            "poses": poses,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"frame": "motion-lab, row-major 3x3", "cases": cases}, indent=1))
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
