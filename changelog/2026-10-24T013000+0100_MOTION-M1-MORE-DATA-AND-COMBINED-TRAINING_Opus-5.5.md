# Motion core M1: 100STYLE, CMU and BABEL; combined training set up

Model name: **Opus 5.5**. Checkpoint **2026-10-24 01:30:00 +01:00**, Europe/London. Status:
**data built and checked; combined training set up and smoke-tested; not run**.

The owner couldn't download from AMASS's server, so they asked for alternatives. They then had me
download 100STYLE, Bandai Namco and CMU, downloaded BABEL themselves, and said: "can you handle the
setup for training now?"

## Data (all git-ignored; listed in motion-lab's `data/README.md`)

| Source | Status | Size |
|---|---|---|
| 100STYLE | downloaded (Zenodo; CC BY 4.0 on the record) and built | 810 takes, 22.1 h |
| CMU Motion Capture Database | downloaded (mocap.cs.cmu.edu; "free for all uses") and built | 2,503 takes, 10.8 h (11 under 1 s skipped) |
| BABEL | downloaded by the owner; copied from their Downloads into `data/babel/` | labels attached to 1,902 CMU takes |
| Bandai Namco | downloaded (GitHub; CC BY-NC-ND 4.0) | not built: no T-pose anywhere, needs another calibration |

The total for training is about 37 hours with LAFAN1, against 4.5 before.

## Changes (motion-lab, uncommitted)

- **`datasets/retarget.py`:** a shared retarget for T-pose sources.
  - Profiles map canonical joints to source joints; unnamed joints fold into their neighbours.
  - The world axes are found from the rest skeleton.
  - A per-frame swing correction makes every bone point exactly where the source's does
    (without it, folding CMU's two neck bones lost up to 23°).
  - Root scale comes from standing height (100STYLE) or the rest skeleton (CMU, per performer).
  - The floor comes from planted ankles.
- **`datasets/style100.py`, `datasets/cmu.py`:** the profiles, an ASF/AMC reader, splits (by
  style; by performer), and CMU's 60 fps takes. BABEL's durations showed 512 takes are 60 fps,
  not 120.
- **`scripts/build_datasets.py`:** builds both, with a report per take.
- **`datasets/m1_data.py`:**
  - splits for the new sources;
  - optional system-memory storage, with rotations at half precision;
  - sampling weights per source.
- **`m1.py`:** a new loss on the hips' vertical velocity. It fixes the baseline's hips-height
  jitter of 19,400 m/s³; the data's is about 500.
- **`configs/m1/combined.json`:** LAFAN1, 100STYLE (weight 0.3) and CMU; 40,000 steps.
- **Tests:** 53, all passing. They include an ASF/AMC bend test, the sampler's storage and
  weights, and real CMU and 100STYLE takes retargeting with bone errors under 0.01°.

## Bugs found while building

- **Zero root scale:** CMU's root scale divided by a zero standing height. The test caught it
  before CMU built.
- **CMU's neck:** the 23° folding error, fixed by the swing correction.

## Verification

- **Builds:** worst bone-direction error 0.0° in both.
  - Planted ankles (median): 0.088 m for 100STYLE, 0.110 m for CMU; her rest height is 0.092 m.
  - Standing hips (median): 0.878 and 0.877 m; hers is 0.895 m.
- **Smoke run of the combined config** (300 steps):
  - 6.1M training frames and 1.2M validation frames load;
  - about 0.66 s per step;
  - the hips-velocity loss falls from 2.39 to 1.21.
  - Peak memory wasn't captured.
- **Not run:** the combined training (about 7.3 hours) and the Bandai Namco build.

Nothing committed.
