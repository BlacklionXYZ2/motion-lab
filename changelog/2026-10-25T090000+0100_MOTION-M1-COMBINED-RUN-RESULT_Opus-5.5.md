# Motion core M1: the combined run, and a retargeting flaw it exposed

Model name: **Opus 5.5**. Checkpoint **2026-10-25 09:00:00 +01:00**, Europe/London. Status:
**trained and evaluated; a data flaw found; fix proposed, not made**.

## The run

`configs/m1/combined.json` (LAFAN1, 100STYLE and CMU; about 37 hours of data), 40,000 steps,
7.6 hours. It finished cleanly.
- **Validation** (32-tick rollouts on held-out performers and styles): 4.0° per joint and 2.25 cm
  at the end. The best total was at step 27,000.

## Evaluation (60 s free rollouts, the three test splits, 16 seeds)

| | Best (27k) | Final (40k) | LAFAN1 baseline | Recordings |
|---|---|---|---|---|
| Jitter | 47 m/s³ | 257 m/s³ | 15,634 m/s³ | 390 m/s³ |
| Planted foot slide | 0.16 m/s | 0.23 m/s | 0.39 m/s | 0.10 m/s |
| Share of time planted | 88% | 81% | 16% | 49% |
| Motion energy over 60 s | 0.09 → 0.03 rad/s | 0.09 → 0.05 rad/s | 0.10 to 0.18 rad/s | 1.12 rad/s |
| Joints beyond limits | 12% | 14% | 1.0% | 0.3% |
| Error at 0.5 / 1 / 2 s | 12.2° / 17.1° / 20.8° | 11.8° / 16.5° / 19.6° | 11.2° / 14.3° / 18.3° | |

(The two models' test sets differ: the combined model's includes CMU and 100STYLE.)

- **Fixed:** the hips buzz (jitter is now below the data's) and much of the foot slide.
- **Worse:** in free rollouts the model winds down to near stillness. Its joints, mostly the
  collarbones (48 to 92° against a 40° limit) and the toes, drift beyond their limits.

## The cause: the retarget copied each dataset's skeleton construction

Typical joint angles differ by dataset far more than real posture could explain:
- **collarbones:** about 25° (LAFAN1), 18° (100STYLE), 38 to 41° (CMU);
- **upper arms:** 79°, 82°, 118°;
- **spine:** 26°, 7°, 9°;
- **neck:** 22°, 6°, 31°.

The retarget made each canonical bone point exactly where the source bone points, at rest and
per frame. Each dataset places its joints differently: CMU's shoulder joint sits higher above
the collarbone, LAFAN1's spine joint further back. So its geometry, not only its motion, entered
the joint angles. The model sees three different "neutral" poses and drifts between them. The
"0.0° bone error" checks measured the wrong goal for the spine, collarbones and neck.

## Proposed fix (not done)

- **Retarget as rotation relative to each dataset's own T-pose.** A dataset's T-pose maps to her
  T-pose, and only motion away from it carries over. This is what UniVRM's retargeting and
  `AvatarFit` already do.
- **Drop the bone-direction matching** (both the calibration correction and the per-frame swing).
- **Then:** rebuild all three datasets, check that typical angles agree across them, and retrain.
- **Also considered:** a soft joint-limit loss (the roadmap asks for joint limits "from day one",
  and it isn't implemented).
- **Separate issue:** winding down to stillness is the usual regression to the average of a
  model with no input telling it what to do. Intent labels (BABEL, 100STYLE's styles) are the
  roadmap's answer.

Nothing committed.
