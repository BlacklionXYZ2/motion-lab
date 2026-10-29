# Motion core M0: runtime spike done; Sentis on the CPU, the GRU exported as cell arithmetic

Model name: **Opus 5.5**. Checkpoint **2026-10-15 09:00:00 +01:00**, Europe/London; measured on
2026-10-12. Status: **spike done; M0 still open** (the second-agent skeleton review is its gate).

The owner: "Could you continue working on the motion core?" M0's only item an agent could close
alone was the Unity runtime spike. M1 waits for the AMASS (with BABEL) and LAFAN1 licences.

## Changes

**`~/Desktop/motion-lab`** (uncommitted):
- **`export.py`:**
  - `CellStep` exports the GRU as its cell equations, because Unity's Sentis has no `GRU`
    operator; `gru_operator=True` keeps the native node;
  - `operators()` lists a graph's operator types.
- **Tests:** an exported graph has no `GRU` node and matches PyTorch, for one and two layers;
  the native-operator path still works. 35 tests in all.
- **New `scripts/spike_export.py`:** the six M1 sweep sizes, each with ONNX, raw weights and a
  120-tick reference, copied into the Unity project.
- **New `unity-spike/`,** a scratch Unity 6000.6 project:
  - Sentis 2.6.1 and UniVRM 0.131.2;
  - `SpikeRunner` (editor parity and timing);
  - `CsGru` (a hand-written GRU);
  - `SpikePlayer` and `SpikeBuild` (a player built with SORA's settings, driving Alice at
    60 Hz).
- **Docs:**
  - new `docs/RUNTIME_SPIKE.md`;
  - `docs/M0.md` updated;
  - a new README with the current status. The 28 September README moved to
    `docs/history/README_2026-09-28.md`.
- **`.gitignore`:** Unity's generated folders and the copied spike data.

**SORA:**
- the motion roadmap's M0 status;
- new evidence in
  [MOTION_M0-RUNTIME-SPIKE_Opus-5.5_2026-10-12](../../SORA/docs/verification/MOTION_M0-RUNTIME-SPIKE_Opus-5.5_2026-10-12.md).

## Verification

- **Component:**
  - motion-lab: 35 passed; ruff clean;
  - onnxruntime parity 5.4e-7 at all sizes;
  - Unity editor parity at or under 2.5e-7 on Sentis CPU, Sentis GPU and the C# GRU.
- **Native:** the built player drove Alice at 60 Hz.
  - Sentis CPU's whole tick averaged 0.25 to 0.68 ms, with p99 at or under 1.13 ms.
  - Frames held at 16.70 ms, and outputs were finite.
- **Not run:** IL2CPP; running inside SORA, under its own load.

## Limits and next

- **Random weights:** cost and correctness only.
- **The budget** (a tenth of a frame per tick) is this plan's proposal; the owner can change it.
- **Next:**
  - M0's skeleton review by a second agent, with the owner's go-ahead to start one;
  - M1 data pipelines (LAFAN1 BVH, AMASS and BABEL), which can be built and tested on
    synthetic data before the licences arrive.

HOW_TO_USE.md: no change to usage.
