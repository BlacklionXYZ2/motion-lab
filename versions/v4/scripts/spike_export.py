"""Export the runtime-spike models (roadmap M0, runtime spike; sizes from M1's sizing sweep).

    .venv\\Scripts\\python scripts\\spike_export.py

For each size (widths 128, 256 and 512, one and two GRU layers) it writes to ``outputs/spike/``:
- ``step_<width>x<layers>.onnx``: the per-tick model with the GRU as cell arithmetic, which
  Unity's Sentis can run (it has no ``GRU`` operator);
- ``.weights.bin`` and ``.weights.json``: every parameter as little-endian float32, with names,
  shapes and offsets, for a hand-written C# GRU;
- ``.ref.bin``: 120 ticks of a PyTorch run with the hidden state fed back. Each tick holds the
  inputs (state, signal) then the outputs (rot6d, h'), all float32, so any runtime can be checked
  against it.

It then copies them into ``unity-spike/Assets`` (models, and StreamingAssets/spike). The weights
are random (seeded): the spike measures cost and parity, not motion quality. See
``docs/RUNTIME_SPIKE.md``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from motionlab import formats, motion_signal, skeleton
from motionlab.export import StepModel, export, operators, parity

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
OUT = ROOT / "outputs" / "spike"
SIZES = [(128, 1), (256, 1), (512, 1), (128, 2), (256, 2), (512, 2)]
TICKS = 120


def write_weights(model: StepModel, stem: Path) -> list[dict[str, object]]:
    entries, chunks, offset = [], [], 0
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().numpy().astype("<f4")
        entries.append({"name": name, "shape": list(values.shape), "offset": offset})
        chunks.append(values.ravel())
        offset += values.size
    stem.with_suffix(".weights.bin").write_bytes(np.concatenate(chunks).tobytes())
    stem.with_suffix(".weights.json").write_text(json.dumps(entries, indent=1) + "\n")
    return entries


def write_reference(model: StepModel, stem: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    h = model.initial_state()
    rows = []
    with torch.no_grad():
        for _ in range(TICKS):
            state = torch.randn(1, formats.STATE_SIZE, generator=generator) * 0.5
            signal = torch.rand(1, motion_signal.FEATURE_SIZE, generator=generator)
            rot6d, h = model(state, signal, h)
            rows.append(torch.cat((state[0], signal[0], rot6d[0], h.reshape(-1))).numpy())
    stem.with_suffix(".ref.bin").write_bytes(np.stack(rows).astype("<f4").tobytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = []
    for hidden, layers in SIZES:
        torch.manual_seed(hidden * 10 + layers)
        model = StepModel(hidden=hidden, layers=layers).eval()
        stem = OUT / f"step_{hidden}x{layers}"
        path = export(model, stem.with_suffix(".onnx"))
        assert "GRU" not in operators(path)
        write_weights(model, stem)
        write_reference(model, stem, seed=hidden + layers)
        models.append({
            "name": stem.name, "hidden": hidden, "layers": layers,
            "parameters": sum(p.numel() for p in model.parameters()),
            "onnx_bytes": path.stat().st_size,
            "onnx_parity": parity(model, path, ticks=TICKS),
        })
        print(models[-1])
    (OUT / "models.json").write_text(json.dumps({
        "state_size": formats.STATE_SIZE, "signal_size": motion_signal.FEATURE_SIZE,
        "rot6d_size": skeleton.J * 6, "ticks": TICKS, "models": models,
    }, indent=1) + "\n")
    copy_into_unity()


def copy_into_unity() -> None:
    """The Unity spike project imports the .onnx files as models and reads the rest at runtime."""
    unity = LAB / "unity-spike" / "Assets"
    if not unity.is_dir():
        return
    (unity / "Models").mkdir(exist_ok=True)
    (unity / "StreamingAssets" / "spike").mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.suffix == ".onnx":
            shutil.copy2(path, unity / "Models" / path.name)
        elif path.suffix in {".bin", ".json"} and path.name != "unity_results.json":
            shutil.copy2(path, unity / "StreamingAssets" / "spike" / path.name)
    print("copied into", unity)


if __name__ == "__main__":
    main()
