# Runtime spike: running the motion model in Unity

Roadmap M0's runtime spike (`../SORA/docs/plans/MOTION-CORE-ROADMAP_2026-10-11.md`). Built and
measured by **Opus 5.5**, 2026-10-12, on the owner's PC (Intel i5-14600K, AMD RX 9070 XT,
Unity 6000.6.0f1). It never touched SORA's Unity project or install.

## Decision

**Run the motion model with Unity's Sentis (Unity Inference Engine 2.6.1) on the CPU.** Export
the GRU as cell arithmetic, because Sentis has no ONNX `GRU` operator.

- **Sentis has no `GRU` operator.** Its supported-operators page lists `GRU` as unsupported
  (`LSTM` is supported). `export.py` now writes the GRU as its own cell equations (`CellStep`):
  MatMul, Gemm, Sigmoid, Tanh, Mul, Add, Sub and Split. All of these are supported, and the
  result is exact, because one tick is one cell step. Training keeps `nn.GRU`.
- **The CPU beats the GPU at this size.** The GPU path pays for uploading inputs and reading
  results back every tick, which costs more than the arithmetic.
- **A hand-written C# GRU is correct but too slow under Mono,** which SORA's player uses. It
  runs plain loops without Burst: 0.9 to 10 ms per tick.
- **An ONNX Runtime plugin wasn't tried,** because Sentis works.
- **Runtime budget (decision 4):** one tick may use at most a tenth of a 60 Hz frame, 1.67 ms
  at the 99th percentile. Every size of the M1 sweep fits on Sentis CPU, up to 512 × 2 (3.0M
  parameters, p99 1.13 ms per whole tick). **Runtime doesn't constrain M1's model size;** data
  does.

## Results

### Built player, driving Alice at 60 Hz

`Builds/Spike/Spike.exe` uses SORA's player settings: Mono, .NET Standard, Direct3D 11, gamma
colour and a Development build. For each model and path it ran 60 warm-up frames, then 600
timed frames.

- **One tick** builds the motion state from her realised pose, runs the model, decodes the 6D
  output, folds `upperChest` into `chest` and writes her control-rig bones.
- **"Model"** is the model call alone. Times are in microseconds; the frame time is in
  milliseconds.

| Model | Parameters | Path | Tick mean | Tick p99 | Model mean | Frame mean / p99 |
|---|---|---|---|---|---|---|
| 128 × 1 | 0.24M | Sentis CPU | 250 | 444 | 232 | 16.70 / 16.70 |
| 128 × 1 | | Sentis GPU | 706 | 965 | 690 | 16.70 / 16.70 |
| 128 × 1 | | C# GRU | 876 | 954 | 860 | 16.70 / 16.70 |
| 256 × 1 | 0.53M | Sentis CPU | 294 | 539 | 276 | 16.70 / 16.70 |
| 256 × 1 | | Sentis GPU | 782 | 1,024 | 765 | 16.70 / 16.70 |
| 256 × 1 | | C# GRU | 1,870 | 1,951 | 1,853 | 16.70 / 16.70 |
| 512 × 1 | 1.40M | Sentis CPU | 399 | 707 | 382 | 16.70 / 16.70 |
| 512 × 1 | | Sentis GPU | 911 | 1,189 | 894 | 16.70 / 16.70 |
| 512 × 1 | | C# GRU | 4,880 | 5,028 | 4,862 | 16.70 / 16.70 |
| 128 × 2 | 0.34M | Sentis CPU | 304 | 484 | 287 | 16.70 / 16.70 |
| 256 × 2 | 0.93M | Sentis CPU | 400 | 814 | 383 | 16.70 / 16.70 |
| 512 × 2 | 2.98M | Sentis CPU | 680 | 1,128 | 663 | 16.70 / 16.70 |
| 512 × 2 | | Sentis GPU | 1,143 | 1,385 | 1,126 | 16.70 / 16.70 |
| 512 × 2 | | C# GRU | 10,438 | 10,995 | 10,414 | 16.70 / 16.70 |

- Every run held 60 Hz, and all outputs were finite.
- Decoding and applying the pose cost about 17 µs, the difference between a whole tick and the
  model call.
- Full rows are in `outputs/spike/player_results.json`, and frames from the 256 × 1 runs are in
  `outputs/spike/shots/`.

### In the editor: parity with PyTorch, and timing

`SpikeRunner.Run` ran each path for 120 ticks with the hidden state fed back, compared against
PyTorch's reference run, then timed 1,000 ticks including upload and readback.

- **Parity:** the largest difference from PyTorch on any path and size was 2.5e-7.
- **Sentis CPU:** 163 to 319 µs per tick.
- **Sentis GPU:** 371 to 900 µs.
- **C# GRU:** 0.86 to 10.4 ms.

Rows are in `outputs/spike/unity_results.json`.

## Limits

- **The weights are random (seeded).** The spike measures cost and correctness of the
  arithmetic, not motion. For display, each joint's rotation is scaled to 15% of its angle.
- **One machine, one run each,** with nothing else heavy running (SORA was closed).
- **Not measured:**
  - an IL2CPP or non-Development build, which would likely be faster;
  - inference running alongside SORA's own work, such as voice, the local model and capture.

  M8 measures those inside SORA.
- **The build ships DirectML** (Sentis's GPU acceleration library), which adds about 16 MB to
  the player. That's acceptable, but noted for M8.

## Reproduce

```powershell
.venv\Scripts\python versions\v2\scripts\spike_export.py   # models, weights, references → versions\v2\outputs\spike and unity-spike
& "$env:ProgramFiles\Unity\Hub\Editor\6000.6.0f1\Editor\Unity.exe" -batchmode -projectPath unity-spike -executeMethod SpikeRunner.Run -quit
& "$env:ProgramFiles\Unity\Hub\Editor\6000.6.0f1\Editor\Unity.exe" -batchmode -projectPath unity-spike -executeMethod SpikeBuild.Build -quit
unity-spike\Builds\Spike\Spike.exe -vrm ..\SORA\.local\avatars\Alice.vrm -out outputs\spike\player_results.json -shots outputs\spike\shots
```

The player opens a 540 × 720 window for about 3½ minutes.
