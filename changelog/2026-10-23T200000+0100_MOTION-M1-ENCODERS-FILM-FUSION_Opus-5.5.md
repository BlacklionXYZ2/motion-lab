# Motion core M1: separate encoders, FiLM and fusion (plan part 1)

Model name: **Opus 5.5**. Checkpoint **2026-10-23 20:00:00 +01:00**, Europe/London. Status:
**built and component-tested; smoke-trained only**.

The owner approved part 1 of the M1 plan ("Include the fusion layer and FiLM, begin part 1!"):
- separate encoders per input;
- two-layer encoders and decoder;
- random signal neutralisation;
- a FiLM layer;
- a fusion layer.

The keyword vocabularies (part 3) wait until after the first training run.

## Changes (motion-lab, uncommitted)

- **`m1.py`, the model** (828,643 parameters; it was 533,067):
  - a two-layer pose encoder;
  - one two-layer encoder per `MotionSignal` group (intent, conversation, emotion, attention,
    audio, context). The groups are found from the contract's feature names, so changing a
    vocabulary only changes sizes;
  - FiLM, where the signals scale and shift the pose features (it starts at no effect);
  - a fusion network that feeds the GRU;
  - a two-layer decoder;
  - GELU activations throughout;
  - signal neutralisation, at 30% per group per window (`signal_dropout` in the config);
  - per-tick signals are accepted.
- **`export.py`:** the export calls each model's own `encode`, so any encoder design exports.
  M0's `StepModel` is unchanged in behaviour.
- **Features are selected by fixed matrices,** so the export uses no operation the runtime spike
  didn't already run in Unity.
- **Tests:** 2 new and 2 updated, 49 in total, all passing. They cover:
  - every signal feature belongs to exactly one group;
  - neutralisation is per window;
  - FiLM starts neutral;
  - each signal reaches the model;
  - ONNX parity, with no operations outside the spike's set.

## Verification

- **pytest:** 49 passed; ruff clean.
- **600-step smoke run against the previous design** (details in motion-lab's `docs/M1.md`):
  - smoother: jitter 1,720 against 6,464 m/s³; the recordings are 499;
  - less foot slide: 0.36 against 0.93 m/s; the recordings are 0.11;
  - slightly lower short-horizon error.
  - But in 60 s free rollouts its motion fades (0.13 → 0.07 rad/s) and the hips sink to a mean of
    0.43 m. That's not conclusive at 3% of a run.
- **Speed:** about 0.64 s per training step (a full run is about 3.5 hours).
- **Crash:** one attempt crashed natively in a GPU copy (exit code 9) after 100 steps. The rerun
  was clean; the cause is unknown (the GPU may have been in use).
- **Not run:** the baseline training run.

Nothing committed.
