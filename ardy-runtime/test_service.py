"""End-to-end check of service.py, the way Core will drive it (2026-10-27).

    .venv\\Scripts\\python test_service.py [--realtime]

1. **Round trip:** canonical → ARDY Core → canonical leaves every mapped joint where it was
   (to rounding) except the folded spine, whose total bend is kept.
2. **The service, as a subprocess, over JSON lines:**
   - seed it with 2 s of her walking (taken from ARDY's own walk, converted to canonical
     60 Hz);
   - ask it to walk at 1.2 m/s for 6 s, then to stand for 4 s;
   - stop it.
3. **Checks:**
   - the first streamed frame continues from the seed's last;
   - no frame-to-frame jump anywhere;
   - walking reaches its speed and standing stops;
   - real-time pacing keeps up (``--realtime``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "external" / "ardy" / "nv-tlabs-ardy"))

import retarget  # noqa: E402

SOURCE = HERE / "outputs" / "stream-fp16-s4-h4-k8-cfg2-seed0.npz"


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.swapaxes(a, -1, -2) @ b
    return np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def canonical_walk() -> tuple[np.ndarray, np.ndarray]:
    """2 s of walking, canonical, 60 Hz (from 4 s into the stream scenario)."""
    data = np.load(SOURCE)
    names = [str(n) for n in data["joint_names"]]
    contacts = [n for n in names if n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
    s = slice(80, 120)  # 4 to 6 s at 20 fps
    local, root, c = retarget.to_canonical(data["global_rot_mats"][0, s],
                                           data["root_positions"][0, s],
                                           data["foot_contacts"][0, s], names, contacts)
    local, root, _ = retarget.resample(local, root, c, int(data["fps"]))
    return local, root


def round_trip() -> None:
    data = np.load(SOURCE)
    names = [str(n) for n in data["joint_names"]]
    parents = [int(p) for p in data["joint_parents"]]
    local, root = canonical_walk()
    ardy_local, ardy_root = retarget.from_canonical(local, root, names, parents)
    ardy_glob = np.empty_like(ardy_local)
    for j, p in enumerate(parents):
        ardy_glob[:, j] = ardy_local[:, j] if p < 0 else ardy_glob[:, p] @ ardy_local[:, j]
    back, back_root, _ = retarget.to_canonical(ardy_glob, ardy_root, np.zeros((len(local), 4)),
                                               names)
    worst = angle_deg(local, back).max()
    gap_mm = np.abs(back_root - root).max() * 1000
    print(f"round trip: worst joint {worst:.4f}°, root {gap_mm:.3f} mm")
    assert worst < 0.01 and np.abs(back_root - root).max() < 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    round_trip()
    local, root = canonical_walk()
    quats = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(len(local), -1, 4)
    seed = {"type": "seed", "fps": 60, "frames": [
        {"root": root[i].tolist(), "rot": quats[i].tolist()} for i in range(len(local))]}
    command = [sys.executable, str(HERE / "service.py")] + ([] if args.realtime else ["--fast"])
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(message: dict) -> None:
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def receive() -> dict:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("the service ended")
        return json.loads(line)

    ready = receive()
    print(f"ready after {time.perf_counter() - started:.1f} s: {ready}")
    assert ready["type"] == "ready", ready
    send({"type": "signal", "key": "walk|-|-", "velocity": [0.0, 1.2]})
    send(seed)
    chunks, arrivals = [], []
    stream_start = time.perf_counter()
    standing = 0
    while standing < 10:  # until 4 s of motion generated after "stand" took effect
        message = receive()
        if message["type"] != "chunk":
            print("message:", message)
            continue
        chunks.append(message)
        arrivals.append(time.perf_counter() - stream_start)
        standing += "stands relaxed" in message["prompt"]
        if len(chunks) == 15:  # after 6 s, stand (fast mode may already be further ahead)
            # Zero, not null: text alone doesn't reliably stop her (one fast run walked on at
            # about 1 m/s); a stationary path does. Unity always sends her intended velocity.
            send({"type": "signal", "key": "idle|-|-", "velocity": [0.0, 0.0]})
    send({"type": "stop"})
    while (message := proc.stdout.readline()) and json.loads(message)["type"] != "stopped":
        pass  # drain chunks already on their way, or the pipe blocks the service
    proc.wait(timeout=20)
    frames = [f for c in chunks for f in c["frames"]]
    rots = np.stack([Rotation.from_quat(f["rot"]).as_matrix() for f in frames])
    roots = np.asarray([f["root"] for f in frames])
    seam_root = np.linalg.norm(roots[0] - root[-1]) * 100
    seam_rot = angle_deg(rots[0], local[-1]).max()
    steps = np.linalg.norm(np.diff(roots, axis=0), axis=1)
    turn = angle_deg(rots[:-1], rots[1:]).max(axis=1)
    speed = steps * 60
    walk = float(speed[60:360].mean())  # 1 to 6 s
    stand = float(speed[-120:].mean())  # the last 2 s, which are 2 to 4 s into standing
    # Net travel over those 2 s: swaying on the spot moves the root back and forth without
    # going anywhere, which speed alone can't tell from walking on.
    drift = float(np.linalg.norm(roots[-1, [0, 2]] - roots[-120, [0, 2]]) / 2)
    print(f"standing: root speed {stand:.3f} m/s, net travel {drift:.3f} m/s")
    print(f"{len(chunks)} chunks, {len(frames)} frames; seam: root {seam_root:.1f} cm, "
          f"worst joint {seam_rot:.1f}°")
    print(f"largest root step {steps.max() * 100:.2f} cm/frame, largest joint turn "
          f"{turn.max():.1f}°/frame; walk {walk:.2f} m/s (target 1.2), stand {stand:.3f} m/s")
    print(f"prompts: {chunks[0]['prompt']!r} → {chunks[-1]['prompt']!r}")
    if args.realtime:
        late = [a - c["t0"] for a, c in zip(arrivals, chunks, strict=True)]
        print(f"arrival minus due time: median {np.median(late):+.2f} s, worst {max(late):+.2f} s "
              "(negative means ahead of when Unity plays it)")
    assert seam_root < 5 and seam_rot < 20, "the stream doesn't continue from the seed"
    assert steps.max() < 0.1, "a root jump"
    assert 0.8 < walk < 1.6 and drift < 0.1, "walking speed off, or she doesn't stop"
    print("ok")


if __name__ == "__main__":
    main()
