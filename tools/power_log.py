"""Log the PC's power while a training run lasts, then total its energy (owner, 2026-10-26).

    .venv\\Scripts\\python tools\\power_log.py --pid <training process id> --out <run folder>
    .venv\\Scripts\\python tools\\power_log.py --summarise <run folder>\\power.csv

Every ``--every`` seconds (default 2) it appends to ``<out>/power.csv``:

- **CPU package power:** Windows' Energy Meter counter (Intel RAPL, ``rapl_package0_pkg``), read
  through PDH. It's the processor's own measurement.
- **GPU power:** the AMD driver's power-management log (``atiadlxx.dll``,
  ``ADL2_New_QueryPMLogData_Get``), sensor 73. ADL doesn't name it; on this RX 9070 XT it moves in
  step with GPU activity (29 to 52 W while v4 trained), so it's taken to be the GPU's power.
  It isn't checked against HWiNFO yet.
- GPU activity and edge temperature, for context.

When the process with ``--pid`` exits, it writes ``<out>/power-summary.json``: hours, kWh for the
CPU and GPU, and an **estimate** for the whole PC at the wall. That adds ``--other-watts``
(default 50) for the motherboard, memory, drives and fans, and divides by ``--psu-efficiency``
(default 0.9). Only a plug-in power meter measures the wall exactly. Both readings include
anything else running on the PC at the time (SORA's language model, a game), so they're the
PC's power during the run, not the training's alone.
"""

from __future__ import annotations

import argparse
import csv
import ctypes as C
import itertools
import json
import os
import time
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path

GPU_POWER, GPU_ACTIVITY, GPU_EDGE = 73, 19, 8


class _Sensor(C.Structure):
    _fields_ = [("supported", C.c_int), ("value", C.c_int)]


class _LogOut(C.Structure):
    _fields_ = [("size", C.c_int), ("sensors", _Sensor * 256)]


class Gpu:
    def __init__(self) -> None:
        self.adl = C.WinDLL("atiadlxx.dll")
        self._buffers: list[C.Array[C.c_char]] = []

        @C.WINFUNCTYPE(C.c_void_p, C.c_int)
        def alloc(size: int) -> int:
            buffer = C.create_string_buffer(size)
            self._buffers.append(buffer)
            return C.addressof(buffer)

        self._alloc = alloc
        self.context = C.c_void_p()
        if self.adl.ADL2_Main_Control_Create(alloc, 1, C.byref(self.context)) != 0:
            raise RuntimeError("ADL didn't start")

    def read(self) -> tuple[float, float, float]:
        out = _LogOut()
        out.size = C.sizeof(_LogOut)
        if self.adl.ADL2_New_QueryPMLogData_Get(self.context, 0, C.byref(out)) != 0:
            return float("nan"), float("nan"), float("nan")
        return (float(out.sensors[GPU_POWER].value), float(out.sensors[GPU_ACTIVITY].value),
                float(out.sensors[GPU_EDGE].value))


class _Formatted(C.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", C.c_double)]


class Cpu:
    """CPU package power (watts) through PDH."""

    def __init__(self) -> None:
        self.pdh = C.WinDLL("pdh.dll")
        self.query = C.c_void_p()
        self.counter = C.c_void_p()
        if self.pdh.PdhOpenQueryW(None, None, C.byref(self.query)) != 0:
            raise RuntimeError("PDH query failed")
        path = r"\Energy Meter(rapl_package0_pkg)\Power"
        if self.pdh.PdhAddEnglishCounterW(self.query, path, None, C.byref(self.counter)) != 0:
            raise RuntimeError(f"no counter {path}")
        self.pdh.PdhCollectQueryData(self.query)

    def read(self) -> float:
        self.pdh.PdhCollectQueryData(self.query)
        value = _Formatted()
        if self.pdh.PdhGetFormattedCounterValue(self.counter, 0x200, None, C.byref(value)) != 0:
            return float("nan")
        return value.value / 1000.0  # milliwatts


def alive(pid: int) -> bool:
    handle = C.windll.kernel32.OpenProcess(0x1000, False, pid)  # query limited information
    if not handle:
        return False
    code = wintypes.DWORD()
    C.windll.kernel32.GetExitCodeProcess(handle, C.byref(code))
    C.windll.kernel32.CloseHandle(handle)
    return code.value == 259  # STILL_ACTIVE


def summarise(path: Path, other_watts: float, psu_efficiency: float) -> dict[str, object]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    energy = {"cpu": 0.0, "gpu": 0.0}
    seconds = 0.0
    for previous, row in itertools.pairwise(rows):
        dt = float(row["unix"]) - float(previous["unix"])
        if not 0 < dt < 60:  # a gap (the logger stopped): don't bridge it
            continue
        seconds += dt
        for key in energy:
            value = float(row[f"{key}_w"])
            if value == value:  # not NaN
                energy[key] += value * dt
    hours = seconds / 3600
    cpu, gpu = energy["cpu"] / 3.6e6, energy["gpu"] / 3.6e6
    wall = (cpu + gpu + other_watts * hours / 1000) / psu_efficiency
    summary = {
        "log": str(path), "samples": len(rows), "hours": round(hours, 3),
        "cpu_kwh": round(cpu, 4), "gpu_kwh": round(gpu, 4),
        "cpu_mean_w": round(energy["cpu"] / seconds, 1) if seconds else None,
        "gpu_mean_w": round(energy["gpu"] / seconds, 1) if seconds else None,
        "estimated_wall_kwh": round(wall, 4),
        "estimate_assumes": {"other_watts": other_watts, "psu_efficiency": psu_efficiency},
    }
    path.with_name("power-summary.json").write_text(json.dumps(summary, indent=1),
                                                     encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--every", type=float, default=2.0)
    parser.add_argument("--summarise", type=Path, metavar="POWER_CSV")
    parser.add_argument("--other-watts", type=float, default=50.0)
    parser.add_argument("--psu-efficiency", type=float, default=0.9)
    args = parser.parse_args()
    if args.summarise:
        print(json.dumps(summarise(args.summarise, args.other_watts, args.psu_efficiency),
                         indent=1))
        return
    if not (args.pid and args.out):
        parser.error("give --pid and --out, or --summarise")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "power.csv"
    new = not path.exists()
    cpu, gpu = Cpu(), Gpu()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["utc", "unix", "cpu_w", "gpu_w", "gpu_activity_pct", "gpu_edge_c"])
        while alive(args.pid):
            gpu_w, activity, edge = gpu.read()
            now = time.time()
            writer.writerow([datetime.now(UTC).isoformat(timespec="seconds"), round(now, 1),
                             round(cpu.read(), 1), gpu_w, activity, edge])
            handle.flush()
            os.fsync(handle.fileno())
            time.sleep(args.every)
    print(json.dumps(summarise(path, args.other_watts, args.psu_efficiency), indent=1))


if __name__ == "__main__":
    main()
