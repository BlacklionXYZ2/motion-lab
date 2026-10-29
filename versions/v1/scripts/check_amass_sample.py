"""Check the AMASS converter against the motion-capture markers in AMASS's sample file.

    .venv\\Scripts\\python scripts\\check_amass_sample.py [data\\amass\\sample\\amass_sample.npz]

The sample (from AMASS's repository) holds the same take as SMPL-H parameters and as labelled
marker positions. Converting the parameters onto the canonical skeleton and running FK should
point each limb the way the markers do:

- shins: knee centre (outer and inner markers) to the ankle marker;
- upper arms: shoulder centre (front and back) to the elbow centre;
- forearms: elbow centre to the wrist centre;
- head facing: back-of-head markers to front;
- left/right: right ankle to left ankle.

Markers sit on the skin, not at joint centres, so a few degrees are expected. Tens of degrees
would mean a wrong mapping or axis convention, and ~180° a left/right or facing flip. Also
reports the estimated root scale and how far the hips joint is from the waist markers' centre.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import fk, skeleton  # noqa: E402
from motionlab.datasets import amass  # noqa: E402

SAMPLE = LAB / "data" / "amass" / "sample" / "amass_sample.npz"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    with np.load(path) as data:
        labels = [label.decode() for label in data["marker_labels"]]
        markers = torch.from_numpy(data["marker_data"]).double() @ amass.WORLD.T
        poses, trans, fps = data["poses"], data["trans"], float(data["mocap_framerate"])
    local = amass.local_rotations(poses)
    root, scale = amass.estimate_root(local, torch.from_numpy(trans).double(), fps)
    world, positions = fk.forward_kinematics(local, root, skeleton.load(torch.float64))

    def m(*names: str) -> torch.Tensor:
        return torch.stack([markers[:, labels.index(n)] for n in names]).mean(0)

    def j(name: str) -> torch.Tensor:
        return positions[:, skeleton.INDEX[name]]

    def angle(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-9)
        degrees = torch.rad2deg(torch.arccos(cos.clamp(-1, 1)))
        return {"median_deg": round(float(degrees.median()), 1),
                "p95_deg": round(float(degrees.quantile(0.95)), 1)}

    checks = {}
    for side, s in (("left", "L"), ("right", "R")):
        checks[f"{side} shin"] = angle(j(f"{side}Foot") - j(f"{side}LowerLeg"),
                                       m(f"{s}ANK") - m(f"{s}KNE", f"{s}KNI"))
        checks[f"{side} upper arm"] = angle(j(f"{side}LowerArm") - j(f"{side}UpperArm"),
                                            m(f"{s}ELB", f"{s}IEL") - m(f"{s}FSH", f"{s}BSH"))
        checks[f"{side} forearm"] = angle(j(f"{side}Hand") - j(f"{side}LowerArm"),
                                          m(f"{s}OWR", f"{s}IWR") - m(f"{s}ELB", f"{s}IEL"))
    head_forward = world[:, skeleton.INDEX["head"], :, 2]
    checks["head facing"] = angle(head_forward, m("LFHD", "RFHD") - m("LBHD", "RBHD"))
    checks["left of right (ankles)"] = angle(j("leftFoot") - j("rightFoot"),
                                             m("LANK") - m("RANK"))
    waist = m("LFWT", "RFWT", "LBWT", "RBWT")
    offset = (j("hips") - waist)
    report = {
        "file": path.name, "frames": int(poses.shape[0]), "fps": fps,
        "estimated_root_scale": round(scale, 3),
        "hips_minus_waist_markers_m": [round(float(v), 3) for v in offset.median(0).values],
        "checks": checks,
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
