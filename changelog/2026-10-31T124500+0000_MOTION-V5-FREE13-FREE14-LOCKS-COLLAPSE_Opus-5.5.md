# Motion core v5: with every cheat locked out, the free-run fine-tunes collapse into stillness

Model name: **Opus 5.5**. Checkpoint **2026-10-31 12:45 UTC**. Status: **tested; a new
direction training**.

Follows `2026-10-31T084000+0000_MOTION-V5-HONEST-STEPPING-AND-GROUND-LOCK_Opus-5.5.md`.

## Results (run-time sampling, 16 seeds)

| | recordings | free7 (glider) | free12b | free13 | free14 (+ ground lock) |
|---|---|---|---|---|---|
| Steering: walk 1.2 / run 3 m/s, reached | – | 1.30 / 3.07 | 0.22 / 1.66 | 0.30 / 0.68 | **0.01 / 0.25** |
| 5 min walking at 1.2 m/s: speed reached | – | – | 0.27 | 0.30 → 0.13 | 0.06 → 0.01 |
| Carried feet (walk, 5 min) | 3.4% | 99.7% | 51% | 57% | **0** |
| Hovering (idle) | 0.07% | 0 | 93% | 96% | **0** |
| Legs crossed (walk) | 10% | 0 | 3% | **85%** | 1% |
| Hips at idle, m | 0.81 | 0.65 | 0.57 | 0.51 | **0.44** |
| Seeds frozen at 5 min | 0 | 0 | all | all | 94 to 100% |
| Outside the limits (walk) | 0.5% | 6% | 2.4% | 0.9% | 8.9% |

**free13:**
- **Its final "best" checkpoint is worse than its step-1,000 snapshot** (steering 0.78 / 2.32 m/s
  then, 0.30 / 0.68 now). Training validation improved throughout (counterfactual error
  0.70 → 0.55 m/s).
- **So validation's 4-second free runs don't predict longer runs.** `train_m1.py` can now keep a
  checkpoint at every validation (`"keep_checkpoints": true`) for the battery to choose from.

**free14 (the ground lock):**
- Hovering and carried feet are gone, but only because she barely moves.
- The ground lock lowers her body until a foot touches, and with her legs bent that is a deep
  crouch (hips 0.44 m).
- Its counterfactual tracking error stayed at 1.33 m/s for the whole run.

## Reading

- **Each lock removed a cheat:**
  - the foot lock removed gliding;
  - the flight limit removed skimming;
  - the ground lock removed hovering.
- **The free-run losses then drove each model into a collapsed stillness** (frozen, crouched,
  or legs crossed) rather than teach it to step.
- Those losses score plausibility and command-following, not *how* to move. With the cheats
  gone, standing still in some pose scores better than an unpractised attempt at walking.
- **Best of motion-lab's own models for each job:**
  - **free7:** lively, and steers precisely, but glides;
  - **free12b:** steps honestly, slowly;
  - **free4/free5:** a clean body at idle.

## Next (training)

**Learn to walk under the locks from the data first: `latent-vel-lock-sup`.**
- **Setup:** free4's weights (no gliding habit), the foot and ground locks on, and **only the
  supervised losses**: no free runs, no counterfactual commands.
- **Why:** the root now travels by the legs, and the recordings say exactly what the legs
  should do. The lock reproduces 85% of their travel (`scripts/lock_check.py`).
- **Free runs come back only once it walks.**
- **Checkpoints are kept at every validation.**

## Housekeeping

- **Lint:** `ruff` is clean on v5's code and on ardy-runtime's new and changed files.
  - In `stress.py` the joint-index name `I` became `JOINT`.
  - The deliberate import orders in `service.py` and `stress_service.py` are marked, because
    `guard` and `motionlab` need the path `stream.py` sets.
  - The ARDY service still runs (quick check).
- Tests: 88 pass.

Nothing committed.
