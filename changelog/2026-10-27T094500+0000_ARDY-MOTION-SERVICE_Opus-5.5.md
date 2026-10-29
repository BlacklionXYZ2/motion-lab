# The ARDY motion service SORA will run, with the handover seed and the phrase table

Model name: **Opus 5.5**. Checkpoint **2026-10-27 09:45 UTC**. Status: **built and tested in
motion-lab; SORA's side is in progress**.

The owner, on the design:
- "The motion model should have the same restart button that the internal llm has. Then we can
  record motion data over the two to three seconds between when the button is pressed and ardy
  takes over to pass the data in."
- Then: "Go ahead and" (cut off; taken as go-ahead to build).

## Changes (`ardy-runtime/`)

- **`service.py`:** the motion service, JSON lines over stdin/stdout (protocol in its docstring).
  - It takes a seed (her recorded pose, canonical, any frame rate) and continues from it.
  - Signals carry a phrase-table key and a target velocity.
  - It streams canonical 60 Hz chunks of 24 frames, keeping about 0.45 s ahead of real time.
  - It runs at the real-time setting: 4 frames of history, a replan every 8 frames, 4 diffusion
    steps.
  - Library prints go to stderr, so stdout carries only the protocol (NVIDIA's model loading
    printed into it at first).
- **`phrases.py` → `assets/phrases.npz`:** our keywords as 686 keys and 392 prompts, encoded
  once (458 s on the CPU, 6.4 MB). An emotion's intensity becomes "slightly" or "very".
- **`retarget.from_canonical`:** canonical → ARDY Core, for the seed.
- **`stream.py`:** the target-velocity ease now runs on played time across replans.
- **`test_service.py`:** the end-to-end check, fast and `--realtime`.
- **Lint:** ruff fixes across `ardy-runtime/` and `tools/`.

## Measured

- The round trip canonical → ARDY → canonical is exact.
- The service is ready 7.2 s after starting.
- The stream continues from the seed (1.7 cm, 2.9°), with no jumps (largest step 2.4 cm a frame).
- Walking holds 1.19 m/s against a 1.2 target.
- Standing stops: net travel at most 0.009 m/s, over five runs.
- Real-time pacing: median 0.31 s ahead, worst 0.29 s late.
- **Two fixes the test found:**
  - **The ease:** restarting it each replan left her creeping at 0.15 to 0.26 m/s after
    "stand"; it now runs on played time.
  - **Stopping:** a null velocity sometimes let her walk on, so stopping sends zero.

Nothing committed.
