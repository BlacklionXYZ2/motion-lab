"""Generate ARDY motion with NVIDIA's own code and our precomputed text features.

    .venv\\Scripts\\python probe.py "walk forward calmly" [--seconds 30] [--seed 0]
        [--backend torch-cpu|onnx-dml|onnx-cpu] [--precision fp16|fp32] [--steps 10]

Uses NVIDIA's ARDY package (``external/ardy/nv-tlabs-ardy``, Apache-2.0) with its original
checkpoint, with no text encoder attached: each prompt is encoded by ``text_table.py``'s INT4
encoder on the CPU first. With an ONNX backend, the denoiser and decoder calls go to TREE
Industries' ONNX graphs through ONNX Runtime (``onnx_backend.py``); everything else is NVIDIA's
code. Writes ``outputs/<name>.npz`` (local rotation matrices, root positions, joint positions and
foot contacts at 20 fps) and prints timing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent / "external" / "ardy"
sys.path.insert(0, str(EXTERNAL / "nv-tlabs-ardy"))
sys.path.insert(0, str(HERE))

import text_table  # noqa: E402
from ardy.model.load_model import load_model  # noqa: E402
from ardy.tools import seed_everything  # noqa: E402

CHECKPOINTS = EXTERNAL / "checkpoints"


def encode(prompts: list[str]) -> torch.Tensor:
    tok, run = text_table.tokenizer(), text_table.session("cpu")
    features = [run.run(None, text_table.prepare(tok, p))[0] for p in prompts]
    return torch.from_numpy(np.stack(features))  # (B, 1, 4096)


def load(backend: str, precision: str):
    model = load_model("core40", device="cpu", text_encoder=False,
                       checkpoints_dir=str(CHECKPOINTS))
    if backend.startswith("onnx"):
        import onnx_backend

        onnx_backend.attach(model, provider=backend.split("-")[1], precision=precision)
    return model


def default_history(model) -> int:
    """NVIDIA's default (``scripts/generate.py``): the longest history that, with the 2 s
    horizon, fits ARDY's 10 s training window, so 160 frames (8 s) for Core."""
    fps, horizon = int(model.motion_rep.fps), model.gen_horizon_len
    patch = model.num_frames_per_token
    window = (10 * fps // patch) * patch
    return ((window - horizon) // patch) * patch


def generate(model, text_feat: torch.Tensor, frames: int, steps: int,
             cfg: tuple[float, float] = (2.0, 2.0), history: int | None = None) -> dict:
    history = default_history(model) if history is None else history
    b = text_feat.shape[0]
    pad_mask = torch.ones(b, frames, dtype=torch.bool)
    with torch.no_grad():
        motion = model(None, frames, num_denoising_steps=steps, pad_mask=pad_mask,
                       first_heading_angle=torch.zeros(b), motion_mask=None,
                       observed_motion=None, cfg_weight=cfg, text_feat=text_feat,
                       text_pad_mask=torch.ones(b, 1, dtype=torch.bool),
                       crop_history_length=history, progress_bar=lambda x: x)
        return model.motion_rep.inverse(motion, is_normalized=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backend", default="torch-cpu",
                        choices=("torch-cpu", "onnx-dml", "onnx-cpu"))
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--name")
    args = parser.parse_args()
    started = time.time()
    model = load(args.backend, args.precision)
    loaded = time.time()
    text_feat = encode([args.prompt])
    encoded = time.time()
    seed_everything(args.seed)
    fps = int(model.motion_rep.fps)
    frames = int(args.seconds * fps) // 4 * 4
    out = generate(model, text_feat, frames, args.steps)
    done = time.time()
    windows = int(np.ceil(frames / model.gen_horizon_len))
    name = args.name or f"{args.backend}-{args.precision}-{args.seed}"
    path = HERE / "outputs" / f"{name}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez(path, prompt=args.prompt, fps=fps,
             **{k: v.numpy() for k, v in out.items() if isinstance(v, torch.Tensor)})
    print(f"{args.backend} {args.precision}: load {loaded - started:.1f} s, "
          f"text {encoded - loaded:.1f} s, "
          f"generate {frames} frames ({frames / fps:.0f} s of motion) in {done - encoded:.1f} s: "
          f"{(done - encoded) / windows * 1000:.0f} ms per 2 s window; "
          f"{frames / fps / (done - encoded):.1f}x real time → {path.name}")
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
    if args.backend.startswith("onnx"):
        import onnx_backend

        t = onnx_backend.TIMING
        print(f"inside ONNX: denoiser {t['denoiser']:.1f} s over {t['denoiser_calls']:.0f} calls "
              f"({t['denoiser'] / max(1, t['denoiser_calls']) * 1000:.0f} ms each), decoder "
              f"{t['decoder']:.1f} s over {t['decoder_calls']:.0f}; the rest (NVIDIA's code on the "
              f"CPU) {done - encoded - t['denoiser'] - t['decoder']:.1f} s")


if __name__ == "__main__":
    main()
