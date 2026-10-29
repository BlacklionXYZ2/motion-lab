# Motion core M1: pausing, resuming and starting from a trained model

Model name: **Opus 5.5**. Checkpoint **2026-10-23 20:30:00 +01:00**, Europe/London. Status:
**built and tested on short runs**.

The owner: "add save and resume then". This is so a training run (about 3.5 hours) can give the
GPU back when they need it.

## Changes (motion-lab `scripts/train_m1.py`, uncommitted)

- **`state.pt`** holds everything needed to carry on exactly: model, optimiser, learning-rate
  schedule, random-number states, step, best score and elapsed time.
  - It's written every 250 steps (`checkpoint_every`) and on a pause.
  - Writes are atomic: a temporary file, then a replace.
- **Pausing:** Ctrl+C, or a file named `STOP` in the run's folder (for runs with no window).
  The `STOP` file is renamed `STOP.handled`, not deleted.
- **`--resume runs/<name>`** continues with the run's own config.
- **`--init-from <checkpoint>`** starts a new run from a trained model's weights and normaliser.
  The model settings must match.
- **Overwriting:** a new run refuses to start in a folder that already has a run.
- **Docs:** motion-lab's `docs/M1.md`, "Pausing and resuming".

## Verification

- **Exactness:** a 60-step run paused after step 1 by a `STOP` file and resumed, against the same
  run straight through: identical losses at steps 30 and 60, and a largest weight difference of
  0.0.
- **A bug the test caught:** the saved state was loaded onto the GPU, which PyTorch rejects for
  random-number states. It now loads on the CPU.
- **`--init-from`:** starting from the 60-step model, validation at step 30 was 0.53, against 0.73
  for a fresh start.
- **The overwrite guard** refuses as intended.
- Ruff clean.
- **Not tested:** pausing with Ctrl+C (the same save path as the `STOP` file, triggered by an
  interrupt).

Nothing committed.
