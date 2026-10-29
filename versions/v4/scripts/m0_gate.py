"""Measure every numeric M0 gate and write ``outputs/m0/gate.json`` (roadmap M0).

    .venv\\Scripts\\python scripts\\m0_gate.py

The tests check pass/fail; this records the measured values as evidence.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from motionlab import rotation as rot
from motionlab import skeleton
from motionlab.export import StepModel, export, parity
from motionlab.fk import forward_kinematics

ROOT = Path(__file__).resolve().parents[1]
GATES = {"rot6d_round_trip": 1e-5, "orthonormality_after_100k_steps": 1e-4,
         "onnx_parity": 1e-5, "fk_rest_pose": 1e-6}


def random_rotations(n: int, gen: torch.Generator, max_angle: float = math.pi) -> torch.Tensor:
    axis = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    return rot.so3_exp(axis * torch.rand(n, 1, generator=gen, dtype=torch.float64) * max_angle)


def main() -> None:
    gen = torch.Generator().manual_seed(11)
    results: dict[str, float] = {}
    r = random_rotations(100_000, gen).float()
    results["rot6d_round_trip"] = float((rot.rot6d_to_matrix(rot.matrix_to_rot6d(r)) - r)
                                        .abs().max())
    r = random_rotations(100_000, gen, math.pi - 0.01).float()
    results["exp_log_inverse_float32"] = float((rot.so3_exp(rot.so3_log(r)) - r).abs().max())

    started = time.perf_counter()
    state = torch.eye(3).repeat(256, 1, 1)
    omega = torch.randn(256, 3, generator=gen) * 3
    for _ in range(100_000):
        state = rot.integrate(state, omega, 1 / 60)
    results["orthonormality_after_100k_steps"] = float(rot.orthonormality_error(state).max())
    results["integration_seconds"] = round(time.perf_counter() - started, 1)

    skel = skeleton.load()
    eye = torch.eye(3).repeat(skeleton.J, 1, 1)
    _, positions = forward_kinematics(eye, skel.offsets[0], skel)
    results["fk_rest_pose"] = float((positions - skel.rest_positions).abs().max())

    torch.manual_seed(0)
    model = StepModel()
    path = export(model, ROOT / "outputs/m0/step_model.onnx")
    results["onnx_parity"] = parity(model, path, ticks=600)
    results["step_model_parameters"] = sum(p.numel() for p in model.parameters())
    results["onnx_bytes"] = path.stat().st_size

    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "results": results,
        "gates": {name: {"limit": limit, "value": results[name], "passed": results[name] < limit}
                  for name, limit in GATES.items()},
        "skeleton_source": skel.source,
    }
    target = ROOT / "outputs/m0/gate.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
