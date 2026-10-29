"""Compare typical joint angles across processed datasets (a retarget consistency check).

    .venv\\Scripts\\python scripts\\compare_datasets.py FOLDER [FOLDER ...]

People don't stand very differently from one capture studio to the next, so each joint's typical
angle from rest should be similar across datasets. A large gap (v1: collarbones about 40° in CMU
against 18 to 25° elsewhere) means the retarget copied a skeleton's construction into the
angles. Samples up to 60 clips per folder, every 10th frame.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import rotation, skeleton  # noqa: E402

JOINTS = ("spine", "chest", "neck", "head", "leftShoulder", "rightShoulder", "leftUpperArm",
          "leftLowerArm", "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes")


def main() -> None:
    folders = [Path(a) for a in sys.argv[1:]]
    rng = random.Random(0)
    print(f"{'median angle from rest':24}" + "".join(f"{f.name:>12}" for f in folders))
    table: dict[str, list[float]] = {j: [] for j in JOINTS}
    for folder in folders:
        files = sorted(folder.glob("*.pt"))
        angles: dict[str, list[torch.Tensor]] = {j: [] for j in JOINTS}
        for path in rng.sample(files, min(60, len(files))):
            r = rotation.rot6d_to_matrix(torch.load(path, weights_only=False)["rot6d"][::10]
                                         .double())
            for j in JOINTS:
                angles[j].append(torch.rad2deg(rotation.rotation_angle(r[:, skeleton.INDEX[j]])))
        for j in JOINTS:
            table[j].append(float(torch.cat(angles[j]).median()))
    for j in JOINTS:
        values = table[j]
        spread = max(values) - min(values)
        print(f"{j:24}" + "".join(f"{v:11.0f}°" for v in values) + f"   spread {spread:.0f}°")


if __name__ == "__main__":
    main()
