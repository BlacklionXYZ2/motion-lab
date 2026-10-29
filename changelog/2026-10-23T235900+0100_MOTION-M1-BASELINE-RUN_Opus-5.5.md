# Motion core M1: the first baseline training run

Model name: **Opus 5.5**. Checkpoint **2026-10-23 23:59:00 +01:00**, Europe/London. Status:
**trained; evaluated; one fault found**.

The owner approved the run ("Go ahead") and reported it finished.

## The run

- **What it was:** `configs/m1/baseline.json` in motion-lab: 828,643 parameters, LAFAN1 only,
  every signal neutral, 20,000 steps, 3.5 hours on the RX 9070 XT.
- **A stall:** the first launch stalled at step 550. The likely cause is its console output
  piped through a PowerShell filter in a minimised window; that wasn't confirmed. It was resumed
  from the step-500 save as a hidden process writing to files (`runs/baseline-out.txt`), and
  finished cleanly.
- **Validation** (subject 4, 32-tick rollouts): the best was at step 14,500, about 0.080 rad
  (4.6°) per joint and 2.5 cm. It drifted up slightly by the end (0.431 total at step 20,000).

## Evaluation (`scripts/evaluate_m1.py`, test subject, 16 seeds, 60 s free rollouts)

| | Baseline | Recordings |
|---|---|---|
| Stable | yes | yes |
| Joints beyond limits | 1.0% | 1.1% |
| Planted foot slide | 0.39 m/s | 0.11 m/s |
| Share of time planted | 16% | 55% |
| Jitter | 15,634 m/s³ | 499 m/s³ |
| Motion energy | 0.10 to 0.18 rad/s | 1.12 rad/s |
| Error at 0.5 / 1 / 2 s | 11.2° / 14.3° / 18.3°; 8.5 / 14.1 / 19.4 cm | |

**The jitter is the hips height, not the body.**
- Split up: hips height jerk is 19,427 m/s³, and the hips move about 2.3 cm between frames. The
  joints relative to the hips are at 377 m/s³, smoother than the recordings, and horizontal
  root motion is at 190.
- The model predicts the hips height afresh each tick, and no loss term penalises it changing
  between ticks.
- Mean hips height stays sensible (0.87 to 0.96 m per 10 s).
- The low motion energy and low planted share may partly follow from this. The model is also
  much calmer than the data (neutral signal, no intent).

## Proposed next (not done)

- Add a loss on the hips' vertical velocity against the recording.
- Continue from this model with `--init-from` for about 5,000 steps (about 55 minutes).
- Re-evaluate.

Nothing committed.
