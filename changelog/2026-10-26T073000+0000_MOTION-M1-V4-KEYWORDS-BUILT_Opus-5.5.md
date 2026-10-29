# Motion core M1: v4 built (keywords, hips above the support)

Model name: **Opus 5.5**. Checkpoint **2026-10-26 07:30 UTC**. Status: **built, tested and
smoke-tested; not trained**. The vocabularies wait for the owner's approval, as the agreed plan
(Part 3) requires before the keyword run.

The owner: "Go ahead, and is v3 the first to have the full suite of model sizes?" The answer was
no: every version so far trained only the 0.8M model, and the 17.5M config hasn't been run.

## What v4 is (motion-lab `versions/v4`, copied from v3)

- **MotionSignal v2, a draft** (`motionlab/motion_signal.py`;
  `motion-lab/specs/motion_signal.v2.schema.json`, beside the frozen v1):
  - **intent:** v1's ten plus run, fall and get_up, or null for unknown. Unknown is no longer
    the same as idle;
  - **emotion:** one of 12 keywords, or null, each with a fixed valence/arousal anchor;
  - **intensity:** low, medium or high, or null.
  - 44 features (v1: 26); the model is 828,979 parameters.
- **Labels from the data** (`motionlab/datasets/labels.py`, applied by `scripts/add_labels.py` to
  v4's own copy of the data). Minutes per intent:

  | walk | run | idle | look | climb | sit | lean | wave | stand | fall | get_up | point | present |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 702 | 380 | 167 | 16 | 9 | 4.1 | 4.0 | 2.7 | 2.3 | 2.1 | 2.0 | 0.5 | 0 |

  - 958 minutes are unknown.
  - Emotion comes only from 100STYLE's acted styles: sad 15, neutral 13, confident 12,
    cheerful 10, frustrated 9 minutes. The other seven emotions have none.
  - Clips with rare intents are sampled up to 21× as often.
- **Hips above the support:** the support height is the lowest planted foot's height above its
  rest height, held in the air and filtered (`outputs/labels-v4.txt`: median range within a
  clip 5.8 cm). The hips' training range is 0.0 to 1.06 m above the support, against v3's 0.06 to
  1.96 m above the floor.
- **Evaluation:** each seed is also run for 30 s under 12 keyword conditions, with root speed,
  energy, limits and hips reported, and the recordings' speed per intent for reference.

## Verified

- **Tests:** 66 passed (v3's 59, adapted to v2, plus 7 new). Ruff clean.
- **Smoke run:** 40 steps through the real training script; the evaluation ran every condition.
- **Not run:** the full training (about 10 h); the Unity environment's tests.

## Other

- motion-lab's `docs/M1.md` and README point at v4; v3's README says it's superseded.
- SORA's roadmap M1 status is updated in place.
- Nothing in SORA uses MotionSignal v2. It changes a frozen contract and needs the owner's
  approval first.

Nothing committed.
