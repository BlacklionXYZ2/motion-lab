# ARDY: retargeted, evaluated on our measures, and streamed the way SORA would drive it

Model name: **Opus 5.5**. Checkpoint **2026-10-26 21:45 UTC**. Status: **evaluated and streamed
in motion-lab; SORA integration not started**.

The owner: "continue with ARDY" (after moving the changelogs).

## Changes (`ardy-runtime/`; numbers in its README)

- **`retarget.py`:** ARDY Core 27 → canonical 22 at 60 Hz through global rotations. ARDY's
  identity pose equals its T-pose (to 4×10⁻⁸ m), and our rebuilt rotations match ARDY's own to
  0.06°. The root is scaled by 0.8946 / 0.944 (ARDY's standing hips, measured).
- **`evaluate.py`:** 12 prompts × 4 seeds × 60 s, scored with motion-lab v4's measures.
- **`stream.py`:** the streaming loop, after NVIDIA's demo: text plus a target velocity laid as
  root waypoints, with configurable history and replan cadence.
- **Experiments:** `history_test.py`, `path_test.py`, `wave_test.py`, `text_check.py`.
- **`probe.py`:** NVIDIA's default 8 s history (it was 2 s).

## Findings

- **Quality:** far cleaner than our versions.
  - 0 to 1.1% of joint-frames outside the data's limits (v4: 2.1%);
  - foot slide 0.005 to 0.03 m/s (v4: 0.10; recordings 0.08).
- **fp16 against fp32:** fp16 matches fp32 except for 30 to 60% more jitter in quiet poses, both
  far under the recordings' 449 m/s³.
- **A held prompt stops locomotion:** a held "walk" or "run" prompt stops, worst with 8 s of
  history.
- **A target velocity sustains it:**
  - walk 1.1 to 1.3 m/s (target 1.2) and run 2.2 to 2.5 (target 2.5, turning 90°) for 20 s
    each;
  - clean stops;
  - no teleports;
  - 0.06% outside the limits;
  - 0.65 s per 2 s step.
- **Mid-stream prompt changes** (stand → wave) take only with the demo's settings: 4 frames of
  history and replanning about every 0.4 s. Then the wave arrives after about 4 s (hand at
  1.79 m), and raised arms arrive at once.
- **Still to measure:** whether that cadence fits real time. 10 diffusion steps take about
  0.65 s on this GPU.
- **Text steers properly from a fresh start:** NVIDIA's PyTorch and the ONNX networks give the
  same arm motion to 0.01 rad/s.

## Not done

- Fewer diffusion steps;
- fp32 GPU memory;
- NVIDIA's foot-skate correction (needs a C++ toolchain; SORA's avatar layer pins feet anyway);
- any SORA part (the service, Unity, settings and GUI, HOW_TO_USE).

Nothing committed.
