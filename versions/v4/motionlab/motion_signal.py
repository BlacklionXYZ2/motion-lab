"""``MotionSignal`` v2 (DRAFT, 2026-10-26): the motion core's only input from SORA.

v1 (frozen at M0, ``specs/motion_signal.v1.schema.json``) gave emotion as two floats, valence and
arousal, which a small language model can't set reliably. The owner asked for keywords instead
(2026-10-23): the brain picks words from fixed lists, and the words' meanings are learned.
v2, drafted in ``specs/motion_signal.v2.schema.json`` and **not yet approved**, changes:

- **intent:** v1's ten keywords plus ``run``, ``fall`` and ``get_up`` (the datasets have them and
  SORA needs them), and ``None`` for *unknown*. In v1 the neutral value was ``idle``, so "idle"
  and "unknown" were the same thing to the model; now ``idle`` is a real intent (standing about,
  nothing in particular) and unknown is all zeros.
- **emotion:** a keyword (``EMOTIONS``) or ``None`` for unknown, plus an **intensity** (``low``,
  ``medium``, ``high``, or ``None``). Each keyword also feeds its fixed valence and arousal anchor
  (``EMOTION_ANCHORS``), so related emotions start close together and an emotion with little
  data borrows from its neighbours.
- conversation, attention, audio and context are unchanged from v1.

:func:`features` turns one signal into the fixed vector the model's encoders read;
:func:`label_features` does the same for batches of label indices (training data), with ``-1``
meaning unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import torch
from torch import Tensor

VERSION = 2
INTENTS = ("idle", "wave", "sit", "lean", "climb", "stand", "walk", "point", "look", "present",
           "run", "fall", "get_up")
CONVERSATIONS = ("idle", "listening", "thinking", "speaking", "waiting_for_agent")
INTENSITIES = ("low", "medium", "high")
# valence (−1 … 1) and arousal (0 … 1): starting points from the circumplex model of affect
# (Russell 1980), for the owner to adjust. They are fixed inputs, not trained.
EMOTION_ANCHORS: dict[str, tuple[float, float]] = {
    "neutral": (0.0, 0.3),
    "calm": (0.3, 0.15),
    "warm": (0.5, 0.35),
    "cheerful": (0.7, 0.6),
    "excited": (0.7, 0.85),
    "confident": (0.5, 0.55),
    "focused": (0.1, 0.5),
    "surprised": (0.1, 0.85),
    "concerned": (-0.3, 0.5),
    "embarrassed": (-0.3, 0.6),
    "frustrated": (-0.6, 0.7),
    "sad": (-0.6, 0.2),
}
EMOTIONS = tuple(EMOTION_ANCHORS)


@dataclass(frozen=True)
class Attention:
    target: str | None = None  # a Core-issued reference
    salience: float = 0.0  # 0 … 1


@dataclass(frozen=True)
class Audio:
    rms: float = 0.0  # 0 … 1, her speech loudness this frame
    voiced: bool = False
    pitch: float | None = None  # 0 … 1, log-pitch within her voice's range; None when unvoiced
    onset: float = 0.0  # 0 … 1, onset strength (syllable and stress attacks)


@dataclass(frozen=True)
class Context:
    monitor_id: str = ""
    support_surface_id: str | None = None  # the taskbar, a window top or an edge she is on


@dataclass(frozen=True)
class MotionSignal:
    intent: str | None = None  # None: unknown
    intent_target: str | None = None
    conversation: str = "idle"
    emotion: str | None = None  # None: unknown
    intensity: str | None = None  # how strongly the emotion shows; None: unknown
    attention: Attention = field(default_factory=Attention)
    audio: Audio | None = None  # None while she is silent
    context: Context = field(default_factory=Context)
    at: str = ""  # UTC timestamp, ISO 8601

    def __post_init__(self) -> None:
        if self.intent is not None and self.intent not in INTENTS:
            raise ValueError(f"unknown intent {self.intent!r}")
        if self.conversation not in CONVERSATIONS:
            raise ValueError(f"unknown conversation state {self.conversation!r}")
        if self.emotion is not None and self.emotion not in EMOTIONS:
            raise ValueError(f"unknown emotion {self.emotion!r}")
        if self.intensity is not None and self.intensity not in INTENSITIES:
            raise ValueError(f"unknown intensity {self.intensity!r}")
        _range("salience", self.attention.salience, 0, 1)
        if self.audio is not None:
            _range("rms", self.audio.rms, 0, 1)
            _range("onset", self.audio.onset, 0, 1)
            if self.audio.pitch is not None:
                _range("pitch", self.audio.pitch, 0, 1)


def _range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{name} {value} outside {low}…{high}")


def neutral() -> MotionSignal:
    """Everything unknown: what every encoder sees until its data exists, and what training's
    label neutralisation substitutes (roadmap decision 2)."""
    return MotionSignal()


FEATURE_NAMES: tuple[str, ...] = (
    *(f"intent.{i}" for i in INTENTS),
    *(f"conversation.{c}" for c in CONVERSATIONS),
    "intent.has_target",
    *(f"emotion.{e}" for e in EMOTIONS),
    "emotion.valence", "emotion.arousal",
    *(f"emotion.intensity.{i}" for i in INTENSITIES),
    "attention.salience", "attention.has_target",
    "audio.rms", "audio.voiced", "audio.pitch", "audio.has_pitch", "audio.onset",
    "context.on_surface",
)
FEATURE_SIZE = len(FEATURE_NAMES)
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def features(signal: MotionSignal) -> Tensor:
    """The fixed ``(FEATURE_SIZE,)`` vector for one tick."""
    vector = torch.zeros(FEATURE_SIZE)
    if signal.intent is not None:
        vector[INDEX[f"intent.{signal.intent}"]] = 1
    vector[INDEX[f"conversation.{signal.conversation}"]] = 1
    vector[INDEX["intent.has_target"]] = float(signal.intent_target is not None)
    if signal.emotion is not None:
        vector[INDEX[f"emotion.{signal.emotion}"]] = 1
        valence, arousal = EMOTION_ANCHORS[signal.emotion]
        vector[INDEX["emotion.valence"]] = valence
        vector[INDEX["emotion.arousal"]] = arousal
    if signal.intensity is not None:
        vector[INDEX[f"emotion.intensity.{signal.intensity}"]] = 1
    audio = signal.audio or Audio()
    for name, value in (("attention.salience", signal.attention.salience),
                        ("attention.has_target", float(signal.attention.target is not None)),
                        ("audio.rms", audio.rms), ("audio.voiced", float(audio.voiced)),
                        ("audio.pitch", audio.pitch or 0.0),
                        ("audio.has_pitch", float(audio.pitch is not None)),
                        ("audio.onset", audio.onset),
                        ("context.on_surface",
                         float(signal.context.support_surface_id is not None))):
        vector[INDEX[name]] = value
    return vector


@cache
def _tables() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Each table's last row is "unknown" (zeros), so index −1 selects it."""
    base = features(neutral())
    intent = torch.zeros(len(INTENTS) + 1, FEATURE_SIZE)
    for i, name in enumerate(INTENTS):
        intent[i] = features(MotionSignal(intent=name)) - base
    emotion = torch.zeros(len(EMOTIONS) + 1, FEATURE_SIZE)
    for i, name in enumerate(EMOTIONS):
        emotion[i] = features(MotionSignal(emotion=name)) - base
    intensity = torch.zeros(len(INTENSITIES) + 1, FEATURE_SIZE)
    for i, name in enumerate(INTENSITIES):
        intensity[i] = features(MotionSignal(intensity=name)) - base
    return base, intent, emotion, intensity


def label_features(intent: Tensor, emotion: Tensor, intensity: Tensor) -> Tensor:
    """Integer label tensors of one shape (−1 = unknown) → ``(*shape, FEATURE_SIZE)``, with
    every other input neutral."""
    base, intents, emotions, intensities = (t.to(intent.device) for t in _tables())
    return (base + intents[intent.long()] + emotions[emotion.long()]
            + intensities[intensity.long()])
