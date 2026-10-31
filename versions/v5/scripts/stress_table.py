"""Compare stress reports (``scripts/stress_m1.py``) side by side as a Markdown table.

    ..\\..\\.venv\\Scripts\\python scripts\\stress_table.py REPORT.json [REPORT.json ...]
        [--names NAME ...]

The first column is the recordings' value from the first report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt(value: object) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, list):
        if not value:
            return "–"
        return f"{_fmt(value[0])} → {_fmt(value[-1])}"
    return str(value)


ROWS = [
    ("Frozen seeds, last 30 s of 5 min (unknown)", "long_runs/unknown/frozen_seed_share_last_30s",
     "recordings/frozen_seed_share_last_30s"),
    ("Frozen seeds, last 30 s of 5 min (walk)", "long_runs/walk/frozen_seed_share_last_30s", None),
    ("Energy over 5 min, rad/s (unknown)", "long_runs/unknown/energy_by_30s",
     "recordings/energy_mean_rad_s"),
    ("Energy over 5 min, rad/s (walk)", "long_runs/walk/energy_by_30s", None),
    ("Root speed over 5 min, m/s (walk)", "long_runs/walk/root_speed_by_30s",
     "recordings/root_speed_mean_m_s"),
    ("Root speed over 5 min, m/s (run)", "long_runs/run/root_speed_by_30s", None),
    ("Gliding, share of moving frames (walk)", "long_runs/walk/glide_share_when_moving",
     "recordings/glide_share_when_moving"),
    ("Carried feet (no foot planted), moving frames (walk)",
     "long_runs/walk/carried_share_when_moving", "recordings/carried_share_when_moving"),
    ("Steering: walk, carried feet", "steering/walk 1.2 m\\s ahead/carried_share_when_moving",
     None),
    ("5 min at 0 m/s: frozen seeds", "long_runs/idle at 0 m\\s/frozen_seed_share_last_30s", None),
    ("5 min at 0 m/s: energy, rad/s", "long_runs/idle at 0 m\\s/energy_by_30s", None),
    ("5 min walking at 1.2 m/s: speed", "long_runs/walk at 1.2 m\\s/root_speed_by_30s", None),
    ("5 min walking at 1.2 m/s: energy", "long_runs/walk at 1.2 m\\s/energy_by_30s", None),
    ("5 min walking at 1.2 m/s: carried feet",
     "long_runs/walk at 1.2 m\\s/carried_share_when_moving", None),
    ("5 min walking at 1.2 m/s: outside limits",
     "long_runs/walk at 1.2 m\\s/outside_limits_share", None),
    ("Hovering, every foot over 15 cm (idle)", "long_runs/idle/airborne_share",
     "recordings/airborne_share"),
    ("Legs crossed (walk)", "long_runs/walk/legs_crossed_share", "recordings/legs_crossed_share"),
    ("Wrist in torso (walk)", "long_runs/walk/hand_in_torso_share",
     "recordings/hand_in_torso_share"),
    ("Root speed p99, m/s (walk)", "long_runs/walk/root_speed_p99_m_s",
     "recordings/root_speed_p99_m_s"),
    ("Fastest root frame, m/s (walk)", "long_runs/walk/root_speed_max_m_s",
     "recordings/root_speed_max_m_s"),
    ("Root frames over 5 m/s (walk)", "long_runs/walk/root_spike_share",
     "recordings/root_spike_share"),
    ("Toes in the floor (walk)", "long_runs/walk/toe_in_floor_share",
     "recordings/toe_in_floor_share"),
    ("Turning, rad/s (walk)", "long_runs/walk/turn_rate_abs_rad_s",
     "recordings/turn_rate_abs_rad_s"),
    ("Outside the data's limits (walk, 5 min)", "long_runs/walk/outside_limits_share",
     "recordings/outside_limits_share"),
    ("Hips, mean m (idle)", "long_runs/idle/hips_m", "recordings/hips_m"),
    ("Flicker every tick: jerk against steady",
     "flicker/walk\\idle every tick/jerk_ratio_vs_steady", None),
    ("Push 1 rad: never recovered", "pushes/1.0 rad/never_recovered_share", None),
    ("Push 1 rad: seconds to recover", "pushes/1.0 rad/seconds_to_recover_median", None),
    ("Mirror asymmetry, rad/s (walk)", "mirror_bias/walk/asymmetry_rad_s", None),
    ("Steering: walk 1.2 m/s, achieved", "steering/walk 1.2 m\\s ahead/speed_m_s_after_3s", None),
    ("Steering: run 3 m/s, achieved", "steering/run 3 m\\s ahead/speed_m_s_after_3s", None),
    ("Steering: stand, residual m/s", "steering/stand (0 m\\s)/speed_m_s_after_3s", None),
    ("Steering: turn 90°, s to face", "steering/turn 90 degrees and walk/"
     "seconds_to_face_within_20deg_median", None),
    ("Steering: turn 180°, s to face", "steering/turn 180 degrees and walk/"
     "seconds_to_face_within_20deg_median", None),
    ("Steering: walk, gliding", "steering/walk 1.2 m\\s ahead/glide_share_when_moving", None),
]


def lookup(report: dict, path: str) -> object:
    # "\\" stands for "/" inside a key (keys such as "walk/idle every tick" and "1.2 m/s")
    parts = [p.replace("\\", "/") for p in path.split("/")]
    node: object = report
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--names", nargs="*", help="column names (default: file names)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console's cp1252 has no "→"
    reports = [json.loads(p.read_text("utf-8")) for p in args.reports]
    names = args.names or [p.stem.removeprefix("stress-") for p in args.reports]
    print("| | recordings | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for label, path, ref in ROWS:
        values = [lookup(r, path) for r in reports]
        if all(v is None for v in values):
            continue
        reference = lookup(reports[0], ref) if ref else None
        if isinstance(reference, list) and len(reference) == 3:  # hips: the mean
            reference = reference[0]
        cells = []
        for v in values:
            if isinstance(v, list) and len(v) == 3 and "hips" in path:
                v = v[0]
            cells.append(_fmt(v))
        print(f"| {label} | {_fmt(reference)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
