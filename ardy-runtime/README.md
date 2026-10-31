# ardy-runtime: NVIDIA's ARDY on the owner's PC (2026-10-26)

A separate environment (`.venv`) for running ARDY. It is separate from motion-lab's training
environment, and nothing here is part of SORA yet.

- **Environment:** CPU PyTorch 2.14.1 for NVIDIA's own code (its sampler, history handling and
  motion representation), and ONNX Runtime DirectML 1.24.4 for the networks on the AMD
  RX 9070 XT.
- **NVIDIA's code** is read from `../external/ardy/nv-tlabs-ardy` (Apache-2.0, commit
  `e706e27`), not installed. The owner approved running it on 2026-10-26.

| File | What |
|---|---|
| `text_table.py` | encodes motion phrases once, with the 4-bit Llama-3 LLM2Vec encoder on the CPU, so run time needs no text encoder. `--check` reproduces the encoder's reference inputs |
| `onnx_backend.py` | swaps ARDY's denoiser and decoder for ONNX Runtime sessions, mirroring NVIDIA's own TensorRT swap (`ardy/model/trt.py`) |
| `probe.py` | generates motion from a prompt with any backend, and reports timing |
| `compare.py` | compares two generations made from the same prompt and seed |
| `outputs/` | generated `.npz` files (git-ignored) |

## The motion service (2026-10-27): what SORA runs

| File | What |
|---|---|
| `service.py` | the service SORA's Core starts: JSON lines over stdin/stdout. In: a seed (her recorded pose), signals (a phrase key and a target velocity), stop. Out: ready, canonical 60 Hz chunks of 24 frames (0.4 s), errors. Nothing is generated before a seed or "start". stdout carries only the protocol |
| `phrases.py` → `assets/phrases.npz` | SORA's keywords (MotionSignal v2 intent, emotion, intensity) as 686 keys and 392 encoded prompts (6.4 MB), so no text encoder runs at run time |
| `retarget.from_canonical` | canonical → ARDY Core, for the seed (ARDY's extra spine joint takes half the bend; hand ends and thumbs at rest). Round trip exact (0.0000°, 0.000 mm) |
| `test_service.py` | the end-to-end check: a 2 s walking seed, walk at 1.2 m/s for 6 s, stand, stop; `--realtime` checks pacing |

**Measured** (fp16, 4 steps, RX 9070 XT):
- **Start-up:** ready 7.2 s after starting (2.1 s of it loading the model).
- **The seam:** the stream continues from the seed's last frame (1.7 cm, 2.9°).
- **No jumps:** the largest root step is 2.4 cm a frame.
- **Walking** 1.19 m/s against 1.2.
- **Stopping:** after "stand" with a zero velocity, net travel is 0.001 to 0.009 m/s (five runs).
- **Pacing (real time):** chunks arrive a median 0.31 s before they're due, at worst 0.29 s late,
  so Unity should buffer about half a second.

**Two fixes the test found:**
- **A clocked ease:** the target-velocity ease now runs on played time across replans. Restarting
  it from her measured velocity at every replan left her creeping on at 0.15 to 0.26 m/s.
- **Zero, not null, to stop:** standing sends a zero velocity, because text alone sometimes
  walked on.

## Stress test, idle anchoring and getting up (2026-10-30)

| File | What |
|---|---|
| `stress_service.py` | drives `MotionService` in-process chunk by chunk, as SORA does, through adversarial scenarios (turns, flicker, sit while travelling, extremes, odd seeds, a 2-minute idle, falls, lying starts), scored with motion-lab v5's stress measures; `--no-anchor`, `--retries`, `--only` |
| `idle_test.py` | the long-idle experiment: energy, turning, hips and limits per 10 s under variants |
| `guard.py` | sanity checks for one chunk (limits, hips band, head below hips, pops) |
| `standing_pose.py` → `assets/standing_pose.npz` | ARDY's own standing frame, the get-up target before she has stood still |

- **Found:**
  - a long idle came apart within 30 to 60 s under every variant tried;
  - after a fall she never got up.
- **Fixed in the service (on by default):**
  - **idle anchoring:** her own standing pose becomes a full-body keyframe 1 s and 2 s ahead
    once she has stood still for 1.6 s;
  - **getting up:** her last standing pose becomes a keyframe fixed 2.5 s ahead whenever she is
    down and asked to get up or stand about.
- **Results:**
  - over 2 minutes of idle, joints outside the limits went from 29% to 3.3% and the largest pop
    from 60° to 18° a frame;
  - she got up in 6 of 6 lying starts.
- Details in motion-lab's changelog `2026-10-30T085000+0000_ARDY-STRESS-IDLE-ANCHOR-AND-GET-UP`.

## Later files (evaluation and streaming)

| File | What |
|---|---|
| `retarget.py` | ARDY Core 27 → canonical 22 at 60 Hz, via global rotations; both are world-aligned T-poses (checked to 4×10⁻⁸ m); root scaled by 0.8946 / 0.944 |
| `evaluate.py` | 12 prompts × seeds × 60 s, scored with motion-lab v4's measures; `--calibrate` measures ARDY's standing hips |
| `stream.py` | the streaming loop SORA would use, mirroring NVIDIA's demo: text plus a target velocity laid as 2D root waypoints every 10 frames, eased in over 2 s |
| `history_test.py`, `path_test.py`, `wave_test.py`, `text_check.py` | the experiments below |

## Findings (2026-10-26)

**Quality on our measures** (`evaluate.py`; 12 prompts × 4 seeds × 60 s, GPU; reports in
`outputs/eval-onnx-dml-*.json`):
- **Standing, sitting, waving, looking, talking:** 0 to 0.3% of joint-frames outside the
  data's limits, foot slide 0.005 to 0.01 m/s.
- **Walking prompts:** 0.06 to 1.1% outside, slide 0.01 to 0.03 m/s.
- **For comparison:** v4 had 2.1% and 0.10 m/s; the recordings 0.4% and 0.08 m/s.
- **fp16 against fp32:** the same motion statistics, but fp16 has 30 to 60% more jitter in
  quiet poses (standing 37 against 25 m/s³; recordings 449). Speed: fp16 2.8× real time, fp32
  2.7×.

**A held prompt doesn't sustain locomotion** (`history_test.py`):
- with NVIDIA's default 8 s history, "walks forward" walks and then stops, and "runs" stops
  within 10 s;
- with 2 s history, walking keeps going (0.3 to 0.6 m/s); running still stops.
- ARDY treats a prompt as a clip that ends.

**A target velocity sustains it** (`stream.py`, the demo's method):
- **The scenario:** walk 20 s at 1.2 m/s, stand 10 s, wave 10 s, then run 20 s at 2.5 m/s
  turning 90°.
- **Results:**
  - walking 1.1 to 1.3 m/s throughout;
  - a clean stop (0.01 m/s);
  - running 2.2 to 2.5 m/s, turned as asked;
  - no root jump over 15 cm a frame;
  - 0.06% outside the limits, slide 0.058 m/s;
  - energy 0.8 to 1.8 rad/s while moving (recordings 1.04).
- **Latency:** median 0.65 s a 2 s step (max 0.83 s).
- **The offline version is unreliable:** handing the whole path to one offline call
  (`path_test.py`) caused impossible speeds and jumps in the first 20 s.

**Prompt changes need frequent replanning with little history** (`wave_test.py`, `text_check.py`):
- **Text steers from a fresh start:** the wave lifts the hand to 1.63 m, and PyTorch and ONNX
  agree.
- **After standing, a new prompt barely takes** with 8 s or 2 s of history, or when a whole 2 s
  is played before replanning.
- **With the demo's settings** (4 frames of history, about 0.4 s played between replans), the
  wave arrives after about 4 s (hand 1.79 m, arm 2.8 rad/s) and raised arms at once (1.73 m).
- **Cost:** a replan every 0.4 s needs generation under 0.4 s; 10 steps take about 0.65 s on
  this GPU.

**Diffusion steps at the demo's cadence** (`stream.py --history 4 --keep 8 --steps N`, the 60 s
scenario, fp16, one seed):

| Steps | Latency median / max | Steps late (budget 400 ms) | Walk (1.2) | Run (2.5) | Outside limits | Slide |
|---|---|---|---|---|---|---|
| 10 | 721 / 830 ms | 150 of 151 | 1.0–1.06 | 1.8–3.4 | 0.0% | 0.051 |
| **4** | **313 / 376 ms** | **0** | **1.04–1.18** | **2.0–2.7** | **0.12%** | **0.059** |
| 2 | 155 / 216 ms | 0 | ~0.5 (misses) | 1.0–2.0 | 1.6% | 0.058 |

- **4 steps is the real-time setting on this GPU,** with motion close to 10 steps'.
- **Waving after standing is unreliable across runs:**
  - strong in `wave_test.py`, with 10 steps and the same settings;
  - weak in this scenario (arm energy 0.17 to 0.26 at 4 steps; 0.05 at 10 steps).
  - NVIDIA's model card says motions don't always match the prompt. A hand target (ARDY's
    end-effector constraint) is the dependable way to get a specific gesture.

## Measured (2026-10-26, prompt "a person walks forward calmly", 20 s, seed 0)

- **Text encoding:** tokenisation reproduces all five reference records exactly. INT4 against
  full-precision embeddings: cosine 0.977 to 0.982. About 1.1 s a phrase on the CPU, no GPU.
- **Speed per 2 s window** (10 diffusion steps):

  | Backend | ms per window | × real time |
  |---|---|---|
  | NVIDIA PyTorch, CPU | 1,457 | 1.4 |
  | ONNX fp32, CPU | 1,128 | 1.8 |
  | ONNX fp32, DirectML GPU | 813 | 2.5 |
  | ONNX fp16, DirectML GPU | 674 | 3.0 |

  On the GPU the time is the denoiser, 66 ms a call × 10 calls a window; NVIDIA's CPU code takes
  0.2 s in 20 s of motion. The paper says 4 steps is acceptable for most uses; that's untested
  here.
- **GPU memory:** about 0.7 GB added while generating (fp16, no text encoder). RAM not yet
  measured: the Store Python launcher hides the real process.
- **Parity, ONNX fp32 against NVIDIA's PyTorch model** (same seed):
  - first 2 s: mean 0.009°, largest 0.06°, root identical;
  - then they drift apart: 1° by 10 s, 4.9° (root 32 cm) by 18 s.
  - That's expected: each window feeds the next, so a tiny float difference sends the sampler
    down a different but equally valid path. The conversion itself is faithful.
- **fp16 against fp32** (both DirectML): first 2 s, mean 0.18° (largest 0.93°), about 20× fp32's
  gap from NVIDIA. They diverge faster after that (the fp16 run walked a different route, ending
  5.3 m away).
- **Still open:** whether fp16's motion is as good as fp32's. That's a quality question (limits,
  foot slide, jitter), not a sameness one, and the evaluation retargeted to our skeleton will
  answer it.
- **The walk itself** (NVIDIA's model): about 0.5 m/s, 9 m in 20 s, hips steady at 0.91 to
  0.97 m, alternating foot contacts.
