# Motion core: v4 training started; power logged per training run

Model name: **Opus 5.5**. Checkpoint **2026-10-26 07:40 UTC**. Status: **v4 is training; the
power logger runs alongside it**.

The owner: "Start the training", then "Can you record the power usage of my pc during each
training set? I want to tally it all up at the end."

## v4 training

`versions/v4/configs/combined.json` started at about 07:05 UTC as a hidden process (process
40380), logging to `outputs/train-combined.*.txt`. It needs about 10 h.

## Power logging (motion-lab `tools/`)

- **`power_log.py`** samples every 2 s until the training process exits, into the run's
  `power.csv`, then writes `power-summary.json`:
  - **CPU package power:** Intel RAPL through Windows' Energy Meter counter. It's the
    processor's own measurement.
  - **GPU power:** the AMD driver's power-management log (ADL), sensor 73. ADL doesn't name it.
    It moves in step with GPU activity (29 to 52 W during v4), so it's taken to be GPU power.
    **Not yet checked against HWiNFO.** HWiNFO needs administrator rights and a click on the
    owner's screen.
  - The summary gives kWh for the CPU and GPU, and an estimate at the wall: plus 50 W for the
    rest of the PC, at 0.9 power-supply efficiency.
- **`power_tally.py`** totals every run in `versions/*/runs` into `outputs/power-tally.md`.
  - Runs with a power log count as measured, scaled up when logging started late.
  - Earlier runs are estimated from their training time (44 W CPU and 40 W GPU, as measured on
    the 0.8M model).
  - Today's total is about 4.3 kWh at the wall, almost all estimated.
- **Started for v4 at 07:31**, about 26 minutes into the run; the tally scales for that.
- **Caveat:** readings include anything else running on the PC at the time.
- **Test:** a dummy-process run wrote four samples and a correct summary. Ruff clean.

Nothing committed.
