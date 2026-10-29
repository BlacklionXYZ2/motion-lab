"""ONNX export of a per-tick motion model, and a parity check (roadmap M0).

The runtime calls the network once per tick as a pure function ``f(state, h) → (output, h')``,
so the hidden state is an explicit input and output, never kept inside the graph. M0 proves the
path with :class:`StepModel`, a small stand-in with the shape of the M1 architecture: the motion
state and ``MotionSignal`` features go through encoders into a GRU, and a decoder emits 6D local
rotations.

Unity's inference package (Sentis, 2.6) does not support the ONNX ``GRU`` operator (checked
2026-10-12 against its supported-operators page). Training keeps ``nn.GRU``; the export writes
the same weights as the GRU cell's own arithmetic (:class:`CellStep`): matrix multiplies,
sigmoid and tanh, which every runtime has. One tick is one cell step, so nothing is lost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from motionlab import formats, motion_signal, skeleton


class StepModel(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 1) -> None:
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.pose_encoder = nn.Sequential(nn.Linear(formats.STATE_SIZE, 256), nn.GELU())
        self.signal_encoder = nn.Sequential(nn.Linear(motion_signal.FEATURE_SIZE, 64), nn.GELU())
        self.gru = nn.GRU(256 + 64, hidden, num_layers=layers, batch_first=True)
        self.decoder = nn.Linear(hidden, skeleton.J * 6)

    def encode(self, state: Tensor, signal: Tensor) -> Tensor:
        """The GRU's input for one tick. Any model exported here provides ``encode``, ``gru``,
        ``decoder``, ``layers`` and ``hidden``."""
        return torch.cat((self.pose_encoder(state), self.signal_encoder(signal)), -1)

    def forward(self, state: Tensor, signal: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """``state (B, STATE_SIZE)``, ``signal (B, FEATURE_SIZE)``, ``h (layers, B, hidden)``
        → ``rot6d (B, J·6)``, ``h' (layers, B, hidden)``."""
        x = self.encode(state, signal)
        out, h_next = self.gru(x.unsqueeze(1), h)
        return self.decoder(out[:, 0]), h_next

    def initial_state(self, batch: int = 1) -> Tensor:
        return torch.zeros(self.layers, batch, self.hidden)


class CellStep(nn.Module):
    """``model`` for one tick, with its GRU written as the cell equations PyTorch uses:

    ``r = σ(W_ir·x + b_ir + W_hr·h + b_hr)``, ``z = σ(W_iz·x + b_iz + W_hz·h + b_hz)``,
    ``n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))``, ``h' = (1 − z) ⊙ n + z ⊙ h``,
    with the weights read from ``model.gru`` (gate order r, z, n). Same inputs and outputs as
    :class:`StepModel`."""

    def __init__(self, model: StepModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, state: Tensor, signal: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        m = self.model
        x = m.encode(state, signal)
        layers = []
        for layer in range(m.layers):
            weight = {name: getattr(m.gru, f"{name}_l{layer}")
                      for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}
            h_prev = h[layer]
            i_r, i_z, i_n = (x @ weight["weight_ih"].T + weight["bias_ih"]).chunk(3, -1)
            h_r, h_z, h_n = (h_prev @ weight["weight_hh"].T + weight["bias_hh"]).chunk(3, -1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            x = (1 - z) * n + z * h_prev
            layers.append(x)
        return m.decoder(x), torch.stack(layers)


def example_inputs(model: StepModel, batch: int = 1, seed: int = 0) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, formats.STATE_SIZE, generator=generator),
            torch.randn(batch, motion_signal.FEATURE_SIZE, generator=generator),
            torch.randn(model.layers, batch, model.hidden, generator=generator) * 0.1)


def export(model: StepModel, path: Path, *, gru_operator: bool = False) -> Path:
    """Write one self-contained ONNX file. By default the GRU is exported as cell arithmetic
    (:class:`CellStep`) so Unity's Sentis can run it; ``gru_operator=True`` keeps ONNX's ``GRU``
    node instead (onnxruntime runs both)."""
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model if gru_operator else CellStep(model).eval(), example_inputs(model), str(path),
        dynamo=True, verbose=False,
        input_names=["state", "signal", "h"], output_names=["rot6d", "h_next"],
        external_data=False,  # one self-contained file for the Unity runtime
    )
    return path


def parity(model: StepModel, path: Path, *, ticks: int = 120, seed: int = 1) -> float:
    """Run PyTorch and onnxruntime side by side for ``ticks`` steps, feeding each one's hidden
    state back to itself; return the largest absolute difference seen in any output."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    state, signal, _ = example_inputs(model, seed=seed)
    h_torch = model.initial_state()
    h_onnx = h_torch.numpy()
    worst = 0.0
    with torch.no_grad():
        for tick in range(ticks):
            drift = torch.full_like(state, 0.01 * tick)
            out_t, h_torch = model(state + drift, signal, h_torch)
            out_o, h_onnx = session.run(None, {"state": (state + drift).numpy(),
                                               "signal": signal.numpy(), "h": h_onnx})
            worst = max(worst, float(np.abs(out_o - out_t.numpy()).max()),
                        float(np.abs(h_onnx - h_torch.numpy()).max()))
    return worst


def operators(path: Path) -> set[str]:
    """The ONNX operator types in an exported graph (to check a runtime supports all of them)."""
    import onnx

    return {node.op_type for node in onnx.load(str(path)).graph.node}
