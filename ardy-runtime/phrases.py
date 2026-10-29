"""SORA's motion keywords → ARDY prompts, encoded once into a phrase table.

    .venv\\Scripts\\python phrases.py            # writes assets/phrases.npz (about 10 MB)
    .venv\\Scripts\\python phrases.py --list     # prints every prompt without encoding

The keywords are MotionSignal v2's (motion-lab ``versions/v4/motionlab/motion_signal.py``, the
owner's choice for now, 2026-10-27): an intent, an emotion and an intensity, each possibly
unknown. Each combination becomes one plain, physical sentence (ARDY's model card advises
neutral, physical wording), for example "a person walks forward very happily".

Intensity only shades an emotion, so it is ignored without one. The table holds every
combination: 14 intents (with unknown) × (1 + 12 emotions × 4 intensities) = 686 prompts. The
service looks a prompt up by its key, ``intent|emotion|intensity``, with "-" for unknown.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "phrases.npz"

INTENTS = {
    None: "stands naturally",
    "idle": "stands relaxed, shifting their weight",
    "wave": "waves with their right hand",
    "sit": "sits down on a chair and stays seated",
    "lean": "leans against a wall",
    "climb": "climbs up onto a ledge",
    "stand": "stands up",
    "walk": "walks forward",
    "point": "points at something with their right hand",
    "look": "looks around",
    "present": "gestures with open hands while explaining",
    "run": "runs forward",
    "fall": "falls to the ground",
    "get_up": "gets up from the ground",
}
EMOTIONS = {
    "neutral": "", "calm": "calmly", "warm": "warmly", "cheerful": "happily",
    "excited": "excitedly", "confident": "confidently", "focused": "attentively",
    "surprised": "in surprise", "concerned": "with a worried look", "embarrassed": "shyly",
    "frustrated": "in frustration", "sad": "sadly",
}
INTENSITIES = {None: "", "low": "slightly", "medium": "", "high": "very"}


def key(intent: str | None, emotion: str | None, intensity: str | None) -> str:
    return "|".join(part or "-" for part in (intent, emotion, intensity if emotion else None))


def prompt(intent: str | None, emotion: str | None, intensity: str | None) -> str:
    manner = EMOTIONS.get(emotion or "", "")
    if manner and emotion and intensity and INTENSITIES[intensity]:
        # "very sadly", "slightly in frustration" reads badly; adverbs take the intensity word.
        manner = f"{INTENSITIES[intensity]} {manner}" if manner.endswith("ly") else manner
    return " ".join(part for part in ("a person", INTENTS[intent], manner) if part)


def table() -> dict[str, str]:
    entries: dict[str, str] = {}
    for intent in INTENTS:
        entries[key(intent, None, None)] = prompt(intent, None, None)
        for emotion in EMOTIONS:
            for intensity in INTENSITIES:
                entries[key(intent, emotion, intensity)] = prompt(intent, emotion, intensity)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    entries = table()
    if args.list:
        for name, text in entries.items():
            print(f"{name:32} {text}")
        print(f"{len(entries)} keys, {len(set(entries.values()))} distinct prompts")
        return
    sys.path.insert(0, str(HERE))
    import text_table

    prompts = sorted(set(entries.values()))
    tok, run = text_table.tokenizer(), text_table.session("cpu")
    started = time.time()
    features = []
    for i, text in enumerate(prompts, 1):
        features.append(run.run(None, text_table.prepare(tok, text))[0][0])
        if i % 50 == 0:
            print(f"{i} of {len(prompts)} ({time.time() - started:.0f} s)", flush=True)
    index = {text: i for i, text in enumerate(prompts)}
    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, keys=np.asarray(list(entries)), prompts=np.asarray(prompts),
             key_to_prompt=np.asarray([index[entries[k]] for k in entries]),
             features=np.stack(features).astype(np.float32))
    print(f"{OUT.relative_to(HERE)}: {len(entries)} keys, {len(prompts)} prompts, "
          f"{np.stack(features).nbytes / 1e6:.1f} MB, {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
