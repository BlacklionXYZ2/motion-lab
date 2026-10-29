"""Run ARDY's denoiser and decoder through ONNX Runtime (DirectML on the owner's AMD GPU).

NVIDIA's ARDY swaps its networks for TensorRT engines with ``ardy/model/trt.py``
(``TRTCFGDenoiser``, ``TRTAutoencoder``). This mirrors that swap with ONNX Runtime sessions on
TREE Industries' ONNX graphs (``external/ardy/ARDY-Core-RP-20FPS-Horizon40-ONNX``), so the rest
of NVIDIA's code (the sampler, history handling, motion representation) runs unchanged.

The ONNX graphs keep their token and frame dimensions dynamic, so no padding is needed. Three
of NVIDIA's denoiser inputs (``future_len``, ``future_mask``, ``text_feat_pad_mask``) were
dropped from the graph at export as unused, and are dropped here too.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import nn

ONNX = Path(__file__).resolve().parent.parent / "external" / "ardy" / \
    "ARDY-Core-RP-20FPS-Horizon40-ONNX"
FOLDERS = {"fp16": "directml-mixed-fp16", "fp32": "fp32"}


def session(path: Path, provider: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if provider == "dml":
        # DirectML needs these off (ONNX Runtime's DirectML notes).
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), options, providers=providers)


MAX_TOKENS, FRAMES_PER_TOKEN = 64, 4  # model_contract.json
# Seconds spent inside each ONNX session and call counts, for probe.py's report.
TIMING: dict[str, float] = {"denoiser": 0.0, "denoiser_calls": 0, "decoder": 0.0,
                            "decoder_calls": 0}


def _np(x: torch.Tensor, dtype) -> np.ndarray:
    return x.detach().cpu().numpy().astype(dtype, copy=False)


def _pad(x: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad axis 1 up to ``size``."""
    extra = size - x.shape[1]
    if extra <= 0:
        return x
    widths = [(0, 0)] * x.ndim
    widths[1] = (0, extra)
    return np.pad(x, widths)


class OnnxCFGDenoiser(nn.Module):
    """Drop-in for NVIDIA's CFG-wrapped denoiser (same call signature as ``TRTCFGDenoiser``)."""

    def __init__(self, path: Path, provider: str, denoiser: nn.Module) -> None:
        super().__init__()
        self._session = session(path, provider)
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._denoiser = denoiser
        self.model = denoiser.model if hasattr(denoiser, "model") else denoiser

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._denoiser, name)

    def forward(self, cfg_weight_text, cfg_weight_cstr, token_seq_t, history_len,
                generation_len, future_len, history_mask, generation_mask, future_mask,
                history_token_mask, generation_token_mask, future_token_mask, text_feat,
                text_feat_pad_mask, timesteps, first_heading_angle=None, motion_mask=None,
                observed_motion=None) -> torch.Tensor:
        frames = history_mask.shape[1]
        dim = self.model.motion_rep.motion_rep_dim
        if first_heading_angle is None:
            first_heading_angle = torch.zeros(1)
        if motion_mask is None:
            motion_mask = torch.zeros(1, frames, dim)
        if observed_motion is None:
            observed_motion = torch.zeros(1, frames, dim)
        feed = {
            "cfg_weight_text": _np(cfg_weight_text, np.float32),
            "cfg_weight_cstr": _np(cfg_weight_cstr, np.float32),
            "x": _np(token_seq_t, np.float32),
            "history_len": _np(history_len, np.int64),
            "generation_len": _np(generation_len, np.int64),
            "history_mask": _np(history_mask, np.float32),
            "generation_mask": _np(generation_mask, np.float32),
            "history_token_mask": _np(history_token_mask, np.float32),
            "generation_token_mask": _np(generation_token_mask, np.float32),
            "future_token_mask": _np(future_token_mask, np.float32),
            "text_feat": _np(text_feat, np.float32),
            "timesteps": _np(timesteps, np.int64),
            "first_heading_angle": _np(first_heading_angle, np.float32),
            "motion_mask": _np(motion_mask, np.float32),
            "observed_motion": _np(observed_motion, np.float32),
        }
        missing = self._inputs - set(feed)
        if missing:
            raise RuntimeError(f"denoiser graph wants inputs this wrapper doesn't feed: {missing}")
        start = time.perf_counter()
        output = self._session.run(["output"], feed)[0]
        TIMING["denoiser"] += time.perf_counter() - start
        TIMING["denoiser_calls"] += 1
        return torch.from_numpy(output).to(token_seq_t.device)


class OnnxAutoencoder(nn.Module):
    """Wraps NVIDIA's autoencoder, routing ``detokenize`` to the ONNX decoder (like
    ``TRTAutoencoder``); encoding and requantisation stay in NVIDIA's code."""

    KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        "pose": ("root", "body"), "body": ("body",), "root": ("root",)}

    def __init__(self, path: Path, provider: str, autoencoder: nn.Module) -> None:
        super().__init__()
        self._session = session(path, provider)
        self._outputs = [o.name for o in self._session.get_outputs()]
        self._autoencoder = autoencoder
        self._keys = self.KEYS[autoencoder.decoder_output_feature_mode]

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._autoencoder, name)

    def detokenize(self, token_embeddings, external_cond=None, motion_pad_mask=None, **kwargs):
        # The decoder graph's attention reshapes were traced at 64 tokens (256 frames), so pad up
        # to that, as NVIDIA's TRTDecoder pads to its engine's size. The decoder is causal, so
        # padding after the real frames doesn't change them.
        tokens, frames = MAX_TOKENS, MAX_TOKENS * FRAMES_PER_TOKEN
        if token_embeddings.shape[1] > tokens:
            raise ValueError(f"{token_embeddings.shape[1]} tokens exceed the decoder's {tokens}")
        feed = {"latent_tokens": _pad(_np(token_embeddings, np.float32), tokens),
                "external_cond": _pad(_np(external_cond, np.float32), frames),
                "motion_pad_mask": _pad(_np(motion_pad_mask, np.float32), frames)}
        start = time.perf_counter()
        outputs = self._session.run(self._outputs, feed)
        TIMING["decoder"] += time.perf_counter() - start
        TIMING["decoder_calls"] += 1
        frames = motion_pad_mask.shape[1]
        return {key: torch.from_numpy(value[:, :frames]).to(token_embeddings.device)
                for key, value in zip(self._keys, outputs, strict=True)}


def attach(model, provider: str = "dml", precision: str = "fp16") -> None:
    """Swap ``model``'s denoiser and decoder for ONNX Runtime sessions, in place."""
    folder = ONNX / FOLDERS[precision]
    model.denoiser = OnnxCFGDenoiser(folder / "denoiser.onnx", provider, model.denoiser)
    model.set_autoencoder(OnnxAutoencoder(folder / "decoder.onnx", provider, model.autoencoder))
