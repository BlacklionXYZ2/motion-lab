# Motion core: ARDY runs on the owner's GPU, with no text encoder at run time

Model name: **Opus 5.5**. Checkpoint **2026-10-26 19:05 UTC**. Status: **part 1 of the ARDY
pipeline done (running and measured); SORA integration not started**.

The owner: "Absolutely run NVIDIA's ARDY code. Although is there a way to get it running on my
GPU?" (permission scoped to NVIDIA's ARDY repository).

## Done (motion-lab `ardy-runtime/`; details and numbers in its README)

- **NVIDIA's code drives ONNX Runtime:** NVIDIA's code loads from `external/ardy/nv-tlabs-ardy`
  (read, not installed). `onnx_backend.py` swaps its denoiser and decoder for ONNX Runtime
  sessions, mirroring NVIDIA's TensorRT swap.
- **The decoder graph** was traced at 64 tokens, so its inputs are padded, as NVIDIA's
  TensorRT wrapper pads.
- **`text_table.py`** encodes phrases once with the 4-bit encoder on the CPU, so run time needs
  no text encoder. Its tokenisation reproduces the encoder's reference inputs exactly; cosine to
  full precision 0.977 to 0.982; about 1.1 s a phrase.
- **`probe.py`** generates motion and reports timing; **`compare.py`** compares runs.

## Measured (20 s walk prompt, 10 diffusion steps)

- **Speed per 2 s of motion:**
  - NVIDIA PyTorch on the CPU: 1.46 s;
  - ONNX fp32 on the CPU: 1.13 s;
  - DirectML fp32: 0.81 s;
  - **DirectML fp16: 0.67 s (3× real time)**.
  - On the GPU the time is the denoiser (66 ms × 10 calls); NVIDIA's CPU code takes 0.2 s.
- **GPU memory:** about 0.7 GB while generating (fp16).
- **Parity, ONNX fp32 against NVIDIA's original:**
  - first 2 s: mean 0.009°, largest 0.06°;
  - then the runs drift apart (5° by 18 s), as sampled runs do when each window feeds the
    next.
- **fp16 against fp32:** mean 0.18° in the first 2 s, diverging faster after. Whether fp16's
  motion is as good is still to be measured.

## Not done

- Retargeting to our skeleton and the 60 s evaluation;
- RAM measurement;
- fewer diffusion steps;
- every SORA part: the motion service, Unity, settings and GUI, HOW_TO_USE.

Nothing committed.
