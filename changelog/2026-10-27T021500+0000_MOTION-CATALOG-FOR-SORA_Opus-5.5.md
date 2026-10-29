# The motion catalogue and preview clips for SORA's Motion window

Model name: **Opus 5.5**. Checkpoint **2026-10-27 02:15 UTC**. Status: **done**.

SORA's new Motion window lists what motion-lab has. motion-lab is the source of truth, and SORA
validates what it reads (SORA changelog `2026-10-27T021500+0000_MOTION-MODEL-SETTINGS-AND-WINDOW`).

## Changes

- **`tools/motion_catalog.py`** writes `catalog/catalog.json` from what is on disk:
  - **ARDY** Core fp16 and fp32: ready when the runtime, ONNX networks, checkpoint and NVIDIA's
    code are present, with the real-time stream summaries' numbers;
  - **every `versions/vN`** with `runs/combined/best.pt`: sized by its parameter count, with its
    60 s evaluation, marked experimental. The roadmap's planned sizes are listed under the newest
    version;
  - **the recordings' reference numbers.**
  - It also converts ARDY's fp16 stream into a preview clip, scaled to her height.
- **`tools/version_preview.py`** runs one version's own code (nothing in the version folder
  changes) for a 20 s free rollout ("walk" intent from v4 on). It writes
  `catalog/previews/<version>-<size>.json`: joint positions at 10 fps in millimetres. Run for v1
  to v4.
- **`ardy-runtime/stream.py`:**
  - output names now carry precision, steps, history, cadence and seed (they overwrote each
    other);
  - writes a JSON summary beside the `.npz`.
  - Rerun at the real-time setting (4 steps, 4 frames of history, a replan every 8 frames):
    - **fp16:** median 314 ms per replan (max 357), never late;
    - **fp32:** median 304 ms, 2 of 151 late (max 493 ms).
- **`.gitignore`:** `catalog/` (generated).
- **Not listed:** v0, which has no code snapshot to run.

Nothing committed.
