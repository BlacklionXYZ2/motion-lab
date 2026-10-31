# Motion core v5: sampling as at run time, posture and tracking. A clean body, energy matched

Model name: **Opus 5.5**. Checkpoint **2026-10-30 15:28 UTC**. Status: **trained and
stress-tested**.

Follows `2026-10-30T133000+0000_MOTION-V5-FREE-RUNS-SAMPLING-AND-MEMORY_Opus-5.5.md`.

## Runs (GPU, side by side, from `latent-vel-free3`'s best; 3,000 steps each)

- **`latent-vel-free4`:**
  - free runs sampled as at run time: noise correlation 0.98, 4 s, every third step;
  - `posture` (mean hips within 8 cm of the recorded window's);
  - `jerk_ceiling` (2× the recorded window's);
  - free-run limits 4×.
- **`latent-vel-free5`:** the same plus **`track`**, following the velocity input. Each 0.5 s
  the achieved ground velocity is scored against the command, turned into the world by her own
  heading. Its validation tracking error fell from 1.30 to 0.39 m/s.

## Stress battery (16 test seeds, 5-minute runs; sampled at temperature 1, noise correlation 0.98)

| | recordings | v4 | free4 | free4, memory reset 2.5 s | **free5** | free5, memory reset |
|---|---|---|---|---|---|---|
| Seeds frozen, last 30 s | 0 | all | 0 | 0 | **0** | 0 |
| Energy over 5 min, rad/s | 1.03 | 0.04 → 0.02 | 1.7 → 1.85 | 0.95 → 1.04 | **0.82 → 0.95** | 0.74 → 0.76 |
| Outside the data's limits (walk) | 0.5% | 3.3% | 0.7% | 1.1% | **0.9%** | 0.5% |
| Hovering (idle) | 0.07% | 43% | 1.2% | 5.5% | **1.4%** | 1.5% |
| Legs crossed (walk) | 10% | 32% | 0.05% | 0.06% | **0.01%** | 0.1% |
| Wrist in torso | 0.08% | 2.8% | 0 | 0 | **0** | 0 |
| Hips at idle, mean m | 0.81 | 0.67 | 0.64 | 0.60 | **0.71** | 0.54 |
| Gliding (walk) | 30% | 90% | 71% | 72% | **71%** | 63% |
| Push of 1 rad: never recovered / seconds | – | 6% / 1.3 | 0 / 1.1 | 0 / 0.9 | **0 / 0.6** | 0 / 0.6 |
| Flicker: jerk against steady | – | 4.3× | 1.0× | 1.0× | **1.0×** | 1.0× |
| Mirror asymmetry, rad/s | – | 0.001 | **0.019** (a steady left turn) | 0.005 | **−0.003** | 0.009 |
| Steering, walk 1.2 / run 3 m/s, reached | – | – | 0.10 / 0.29 | 0.05 / 0.21 | 0.09 / 0.16 | 0.29 / 0.35 |
| Turning to a new heading | – | – | never | never | never | never |

**Reading:**
- **latent-vel-free5 without memory resets is the best of motion-lab's own models:**
  - it never freezes, and its energy matches the recordings for 5 minutes;
  - its body stays sane (limits, hovering, crossing and wrists all near the recordings or
    better);
  - it recovers from a 1 rad push in 0.6 s;
  - the mirror bias is gone.
- **Memory resets now hurt** (lower energy and hips). Free runs with the energy band keep the
  memory in range by themselves, so free5 runs without them.
- **Still wrong:**
  - **steering:** asked to walk at 1.2 m/s she reaches 0.09, and she never turns to face a new
    heading. In training the command is always the recording's own future velocity, so it never
    asks for something she isn't already doing;
  - **gliding:** 71% of moving frames, against the recordings' 30%;
  - **posture:** the hips sit 10 cm low (0.71 against 0.81 m).

## Next

- **Free runs with counterfactual commands:** half the free runs get a random constant
  velocity, 0 to 3 m/s in any direction. They are scored on tracking and the recording-free
  terms only, because the energy, posture and jerk references come from a recording the
  command no longer matches.
- **A recording-free skate term:** a foot joint within 3 cm of the floor mustn't slide.
- **Tracking weight 2.**

Not done: viewing on Alice; the Unity tests. Nothing committed.
