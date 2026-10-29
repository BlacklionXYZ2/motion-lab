"""Tally the energy of every training run in ``versions/*/runs`` (owner, 2026-10-26).

    .venv\\Scripts\\python tools\\power_tally.py [--cpu-watts 44 --gpu-watts 40]

A run with ``power-summary.json`` (logged by ``tools/power_log.py``) counts as **measured**,
scaled up to the run's full training time when the logger started late. Any other run is
**estimated**: its training time (the last ``seconds`` in its ``log.jsonl``) times typical CPU
and GPU power, by default what the logger measured on v4's 0.8M model. A run of a larger model
draws more, so its estimate is low. Wall energy adds ``--other-watts`` and divides by
``--psu-efficiency``, as ``power_log.py`` does. Writes ``outputs/power-tally.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def training_seconds(run: Path) -> float:
    log = run / "log.jsonl"
    if not log.exists():
        return 0.0
    last = 0.0
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = max(last, float(json.loads(line).get("seconds", 0.0)))
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cpu-watts", type=float, default=44.0)
    parser.add_argument("--gpu-watts", type=float, default=40.0)
    parser.add_argument("--other-watts", type=float, default=50.0)
    parser.add_argument("--psu-efficiency", type=float, default=0.9)
    args = parser.parse_args()
    rows, total = [], 0.0
    for run in sorted(LAB.glob("versions/*/runs/*")):
        seconds = training_seconds(run)
        if seconds <= 0:
            continue
        hours = seconds / 3600
        summary_path = run / "power-summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text("utf-8"))
            logged = summary["hours"] or hours
            scale = max(1.0, hours / logged)
            cpu, gpu = summary["cpu_kwh"] * scale, summary["gpu_kwh"] * scale
            kind = "measured" + (f" ({logged:.1f} of {hours:.1f} h logged)" if scale > 1.01
                                 else "")
        else:
            cpu, gpu = args.cpu_watts * hours / 1000, args.gpu_watts * hours / 1000
            kind = "estimated"
        wall = (cpu + gpu + args.other_watts * hours / 1000) / args.psu_efficiency
        total += wall
        rows.append(f"| {run.parent.parent.name}/{run.name} | {hours:.2f} | {cpu:.3f} | "
                    f"{gpu:.3f} | {wall:.3f} | {kind} |")
    text = "\n".join([
        "# Training energy tally",
        "",
        f"Estimates use CPU {args.cpu_watts:g} W and GPU {args.gpu_watts:g} W (measured on v4's "
        f"0.8M model); wall energy adds {args.other_watts:g} W for the rest of the PC and divides "
        f"by a {args.psu_efficiency:g} power-supply efficiency. Readings are the PC's power during "
        "each run, including anything else running at the time.",
        "",
        "| Run | Hours | CPU kWh | GPU kWh | Wall kWh (est.) | How |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        f"**Total at the wall: about {total:.2f} kWh.**",
        "",
    ])
    out = LAB / "outputs" / "power-tally.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
