"""ARDY motion service: keywords and a target velocity in, canonical 60 Hz poses out.

    .venv\\Scripts\\python service.py [--precision fp16] [--steps 4] [--fast]

SORA's Core starts this as a helper process and talks JSON lines over stdin and stdout (one
object per line). Everything it sends and receives is in motion-lab's **canonical** terms: the
22-joint skeleton, her proportions, right-handed (+Y up, facing +Z, her left +X), local
rotations as quaternions ``[x, y, z, w]``. Unity converts to and from its own frame.

**In** (stdin):
- ``{"type": "seed", "fps": 60, "frames": [{"root": [x, y, z], "rot": [[x, y, z, w] × 22]}, …]}``:
  her actual recent motion (recorded during the handover). Streaming starts from it, so the
  model continues where she is. A seed restarts the stream.
- ``{"type": "start"}``: start streaming without history (she starts standing). Nothing is
  generated before a seed or a start, so a signal sent first applies from the first frame.
- ``{"type": "signal", "key": "walk|cheerful|high", "velocity": [vx, vz] | null}``: what she is
  doing (a phrase-table key; unknown keys fall back to "-|-|-") and, for locomotion, the
  velocity to travel at (m/s, canonical frame). It applies from the next replan (at most 0.4 s).
- ``{"type": "stop"}``.

**Out** (stdout):
- ``{"type": "ready", "model": …, "load_s": …}`` once loaded;
- ``{"type": "chunk", "seq": n, "t0": s, "fps": 60, "frames": [{"root", "rot", "contacts"}…]}``:
  24 frames (0.4 s) at a time, ``t0`` being stream time (s) of the first frame;
- ``{"type": "error", "message": …}`` and ``{"type": "stopped"}``.

It runs at the real-time setting measured on the owner's GPU: a replan every 8 frames (0.4 s),
4 frames of history, 4 diffusion steps by default. It keeps about one chunk ahead of real time.
``--fast`` skips the pacing (tests and evaluation).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# stdout carries the protocol only: libraries that print (NVIDIA's model loading prints its
# settings) write to stderr instead.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr

import probe  # noqa: E402
import retarget  # noqa: E402
from stream import Streamer  # noqa: E402

PHRASES = HERE / "assets" / "phrases.npz"
KEEP = 8  # ARDY frames played between replans (0.4 s at 20 fps)
HISTORY = 4  # ARDY frames of history per replan (NVIDIA's demo default)
LEAD_S = 0.45  # how far ahead of real time to stay
MAX_MOTION = 64  # ARDY frames kept as context (longer than any history used)
OUT_FPS = 60


class PhraseTable:
    def __init__(self, path: Path = PHRASES) -> None:
        data = np.load(path)
        pairs = zip(data["keys"], data["key_to_prompt"], strict=True)
        self.index = {str(k): int(i) for k, i in pairs}
        self.prompts = [str(p) for p in data["prompts"]]
        self.features = torch.from_numpy(data["features"])

    def feature(self, key: str) -> tuple[torch.Tensor, str]:
        i = self.index.get(key, self.index["-|-|-"])
        return self.features[i].view(1, 1, -1), self.prompts[i]


class MotionService:
    def __init__(self, precision: str, steps: int) -> None:
        started = time.time()
        self.model = probe.load("onnx-dml", precision)
        self.rep = self.model.motion_rep
        self.names = list(self.rep.skeleton.bone_order_names)
        self.parents = [int(p) for p in self.rep.skeleton.joint_parents]
        self.contact_names = list(self.rep.skeleton.foot_joint_names)
        self.fps = int(self.rep.fps)
        self.phrases = PhraseTable()
        self.streamer = Streamer(self.model, HISTORY, steps)
        self.text, self.prompt = self.phrases.feature("-|-|-")
        self.velocity: tuple[float, float] | None = None
        self.seq = 0
        self.t = 0.0  # stream time of the next frame to send
        self.last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # last canonical frame
        self.load_s = time.time() - started

    # ---------- input ----------

    def seed(self, message: dict) -> None:
        frames = message.get("frames") or []
        fps = int(message.get("fps", OUT_FPS))
        if len(frames) < 2:
            raise ValueError("a seed needs at least two frames")
        rot = np.stack([Rotation.from_quat(f["rot"]).as_matrix() for f in frames])  # (T, 22, 3, 3)
        root = np.asarray([f["root"] for f in frames], dtype=np.float64)
        step = max(1, round(fps / self.fps))
        rot, root = rot[::step], root[::step]  # to ARDY's 20 fps
        local, ardy_root = retarget.from_canonical(rot, root, self.names, self.parents)
        usable = len(local) // self.streamer.patch * self.streamer.patch
        local, ardy_root = local[-usable:], ardy_root[-usable:]
        explicit = self.rep(local_joint_rots=torch.tensor(local[None], dtype=torch.float32),
                            root_positions=torch.tensor(ardy_root[None], dtype=torch.float32),
                            to_normalize=False)
        self.streamer.motion = self.rep.normalize(explicit)
        last_c = (rot[-1:], root[-1:], np.zeros((1, 4), dtype=bool))
        self.last = last_c
        self.seq, self.t = 0, 0.0

    def signal(self, message: dict) -> None:
        self.text, self.prompt = self.phrases.feature(str(message.get("key", "-|-|-")))
        velocity = message.get("velocity")
        self.velocity = None if velocity is None else (
            float(velocity[0]) / retarget.ROOT_SCALE, float(velocity[1]) / retarget.ROOT_SCALE)

    # ---------- output ----------

    def next_chunk(self) -> dict:
        """Generate the next 0.4 s and return it as canonical 60 Hz frames."""
        self.streamer.step(self.text, self.velocity, KEEP)
        if self.streamer.motion.shape[1] > MAX_MOTION:
            self.streamer.motion = self.streamer.motion[:, -MAX_MOTION:]
        out = self.rep.inverse(self.streamer.motion[:, -KEEP:], is_normalized=True)
        local, root, contacts = retarget.to_canonical(
            out["global_rot_mats"][0].numpy(), out["root_positions"][0].numpy(),
            out["foot_contacts"][0].numpy(), self.names, self.contact_names)
        if self.last is None:  # no seed: the first frame stands in for "before"
            self.last = (local[:1], root[:1], contacts[:1])
        prev_local, prev_root, prev_contacts = self.last
        both = (np.concatenate((prev_local, local)), np.concatenate((prev_root, root)),
                np.concatenate((prev_contacts, contacts)))
        fine_local, fine_root, fine_contacts = retarget.resample(*both, self.fps, OUT_FPS)
        fine_local, fine_root, fine_contacts = fine_local[1:], fine_root[1:], fine_contacts[1:]
        self.last = (local[-1:], root[-1:], contacts[-1:])
        quats = Rotation.from_matrix(fine_local.reshape(-1, 3, 3)).as_quat().reshape(
            len(fine_local), -1, 4)
        chunk = {"type": "chunk", "seq": self.seq, "t0": round(self.t, 4), "fps": OUT_FPS,
                 "prompt": self.prompt,
                 "frames": [{"root": [round(float(v), 4) for v in fine_root[i]],
                             "rot": np.round(quats[i], 5).tolist(),
                             "contacts": [bool(c) for c in fine_contacts[i]]}
                            for i in range(len(fine_local))]}
        self.seq += 1
        self.t += len(fine_local) / OUT_FPS
        return chunk


def emit(message: dict) -> None:
    PROTOCOL.write(json.dumps(message, separators=(",", ":")) + "\n")
    PROTOCOL.flush()


def read_lines(inbox: queue.Queue) -> None:
    for line in sys.stdin:
        line = line.strip()
        if line:
            inbox.put(line)
    inbox.put(None)  # stdin closed: Core has gone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fast", action="store_true", help="no real-time pacing")
    args = parser.parse_args()
    try:
        service = MotionService(args.precision, args.steps)
    except Exception as exc:
        emit({"type": "error", "message": f"load failed: {type(exc).__name__}: {exc}"[:500]})
        return
    emit({"type": "ready", "model": f"ARDY Core {args.precision}", "steps": args.steps,
          "load_s": round(service.load_s, 2)})
    inbox: queue.Queue = queue.Queue()
    threading.Thread(target=read_lines, args=(inbox,), daemon=True).start()
    clock = None  # wall time at stream time 0
    streaming = False  # nothing is generated before a seed (or "start" without history)
    while True:
        try:
            while True:
                line = inbox.get_nowait()
                if line is None:
                    emit({"type": "stopped"})
                    return
                message = json.loads(line)
                kind = message.get("type")
                if kind == "stop":
                    emit({"type": "stopped"})
                    return
                if kind == "seed":
                    service.seed(message)
                    clock, streaming = None, True
                elif kind == "start":
                    clock, streaming = None, True
                elif kind == "signal":
                    service.signal(message)
                else:
                    emit({"type": "error", "message": f"unknown message type {kind!r}"})
        except queue.Empty:
            pass
        except (ValueError, KeyError, TypeError) as exc:
            emit({"type": "error", "message": f"bad message: {exc}"[:500]})
            continue
        if not streaming:
            time.sleep(0.01)
            continue
        if clock is None:
            clock = time.perf_counter()
        ahead = service.t - (time.perf_counter() - clock)
        if args.fast or ahead < LEAD_S:
            emit(service.next_chunk())
        else:
            time.sleep(min(0.02, ahead - LEAD_S + 0.001))


if __name__ == "__main__":
    main()
