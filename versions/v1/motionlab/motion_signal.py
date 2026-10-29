"""``MotionSignal`` v1: the motion core's only input from SORA, frozen for motion-lab (M0).

It follows SORA's motion interface design (2026-10-05) with the roadmap's defaults:

- **audio (decision 8):** her own speech envelope as ``rms``, ``voiced``, ``pitch`` and ``onset``,
  computed from her TTS output. The owner's voice isn't in v1.
- **emotion (decision 7):** valence and arousal, with no label set.

SORA builds the same contract right after V1 (the M8 prerequisite) and must match
``specs/motion_signal.v1.schema.json``; a change needs a version bump agreed with the owner.

:func:`features` turns one signal into the fixed-size vector the model's encoders read. Geometry
(direction and distance to an attention target, support-surface height) is computed by the
motion layer from Core's resolved targets in M4; it is a separate input, not part of this vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

VERSION = 1
INTENTS = ("idle", "wave", "sit", "lean", "climb", "stand", "walk", "point", "look", "present")
CONVERSATIONS = ("idle", "listening", "thinking", "speaking", "waiting_for_agent")
NEUTRAL_AROUSAL = 0.3


@dataclass(frozen=True)
class Emotion:
    valence: float = 0.0  # −1 … 1
    arousal: float = NEUTRAL_AROUSAL  # 0 … 1


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
    intent: str = "idle"
    intent_target: str | None = None
    conversation: str = "idle"
    emotion: Emotion = field(default_factory=Emotion)
    attention: Attention = field(default_factory=Attention)
    audio: Audio | None = None  # None while she is silent
    context: Context = field(default_factory=Context)
    at: str = ""  # UTC timestamp, ISO 8601

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"unknown intent {self.intent!r}")
        if self.conversation not in CONVERSATIONS:
            raise ValueError(f"unknown conversation state {self.conversation!r}")
        _range("valence", self.emotion.valence, -1, 1)
        _range("arousal", self.emotion.arousal, 0, 1)
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
    """What every encoder sees until its data exists (roadmap decision 2)."""
    return MotionSignal()


FEATURE_NAMES: tuple[str, ...] = (
    *(f"intent.{i}" for i in INTENTS),
    *(f"conversation.{c}" for c in CONVERSATIONS),
    "intent.has_target",
    "emotion.valence", "emotion.arousal",
    "attention.salience", "attention.has_target",
    "audio.rms", "audio.voiced", "audio.pitch", "audio.has_pitch", "audio.onset",
    "context.on_surface",
)
FEATURE_SIZE = len(FEATURE_NAMES)


def features(signal: MotionSignal) -> Tensor:
    """The fixed ``(FEATURE_SIZE,)`` vector for one tick."""
    vector = torch.zeros(FEATURE_SIZE)
    vector[INTENTS.index(signal.intent)] = 1
    vector[len(INTENTS) + CONVERSATIONS.index(signal.conversation)] = 1
    base = len(INTENTS) + len(CONVERSATIONS)
    audio = signal.audio or Audio()
    vector[base:] = torch.tensor([
        float(signal.intent_target is not None),
        signal.emotion.valence, signal.emotion.arousal,
        signal.attention.salience, float(signal.attention.target is not None),
        audio.rms, float(audio.voiced), audio.pitch or 0.0, float(audio.pitch is not None),
        audio.onset,
        float(signal.context.support_surface_id is not None),
    ])
    return vector
