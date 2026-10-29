"""Add v4's per-frame labels and support height to this version's processed clips.

    ..\\..\\.venv\\Scripts\\python scripts\\add_labels.py

For every clip in ``data/processed/{lafan1,100style,cmu}`` (v4's own copy of v3's data), adds
``intent``, ``emotion`` and ``intensity`` (int8 per frame, −1 unknown) and ``support`` (metres per
frame), as ``motionlab/datasets/labels.py`` describes, and writes the clip back in place. The
motion itself is untouched. Running it again recomputes the same values. Prints and writes
``outputs/labels-v4.txt``: minutes of data per intent, emotion and intensity.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parents[1]  # motion-lab: raw downloads (data/), specs and the Unity projects are shared
sys.path.insert(0, str(ROOT))

from motionlab import motion_signal  # noqa: E402
from motionlab.datasets import labels  # noqa: E402
from motionlab.formats import FPS  # noqa: E402

SOURCES = ("lafan1", "100style", "cmu")


def main() -> None:
    minutes: dict[str, Counter[str]] = {"intent": Counter(), "emotion": Counter(),
                                        "intensity": Counter()}
    by_source: dict[str, Counter[str]] = {}
    support_range: list[float] = []
    for source in SOURCES:
        files = sorted((ROOT / "data" / "processed" / source).glob("*.pt"))
        for n, path in enumerate(files, 1):
            data = torch.load(path, weights_only=False)
            support = labels.support_height(data["rot6d"], data["root_position"],
                                            data["contacts"])
            intent = labels.intent_labels(data, support)
            emotion, intensity = labels.emotion_labels(data)
            data.update(support=support.float(), intent=intent.to(torch.int8),
                        emotion=emotion.to(torch.int8), intensity=intensity.to(torch.int8))
            partial = path.with_suffix(".partial")
            torch.save(data, partial)
            os.replace(partial, path)
            for key, values, names in (("intent", intent, motion_signal.INTENTS),
                                       ("emotion", emotion, motion_signal.EMOTIONS),
                                       ("intensity", intensity, motion_signal.INTENSITIES)):
                for index, count in zip(*torch.unique(values, return_counts=True), strict=True):
                    name = "unknown" if int(index) < 0 else names[int(index)]
                    minutes[key][name] += int(count) / FPS / 60
                    if key == "intent":
                        by_source.setdefault(source, Counter())[name] += int(count) / FPS / 60
            support_range.append(float(support.max() - support.min()))
            if n % 250 == 0:
                print(f"{source}: {n} of {len(files)}", flush=True)
    lines = []
    for key, counter in minutes.items():
        lines.append(f"{key} (minutes): " + ", ".join(f"{name} {value:.1f}"
                                                      for name, value in counter.most_common()))
    for source, counter in by_source.items():
        lines.append(f"intent in {source} (minutes): "
                     + ", ".join(f"{name} {value:.1f}" for name, value in counter.most_common()))
    ranges = torch.tensor(support_range)
    lines.append(f"support height range within a clip: median {float(ranges.median()):.3f} m, "
                 f"90th percentile {float(torch.quantile(ranges, 0.9)):.3f} m, "
                 f"max {float(ranges.max()):.3f} m")
    report = "\n".join(lines) + "\n"
    print(report)
    (ROOT / "outputs" / "labels-v4.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
