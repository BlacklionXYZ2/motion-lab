# M3 — Audio / prosody

**Status:** brief — needs a dataset choice.
**Lane:** make/train (owner).

## Goal
Speech-correlated head/body/gesture timing. Explicitly not lip-sync — mouth animation stays with the existing specialist system.

## What you build / train
- **Causal audio encoder** (TCN or GRU) on streaming log-mel frames; keeps its own internal state at the audio feature rate; the controller reads the latest latent each tick.
- **Dataset** — a speech+gesture motion corpus (candidate: BEAT-class). Check licences first — several are research-only; decide what that means for this project before downloading.
- **Training** — add the audio latent to fusion; retrain from the M1 control recipe; ablation runs (audio on/off) to prove it is the audio doing the work.

## Gate
Held-out clips show speech-correlated body/head timing that survives the 60 s stability battery; no regression versus the control.

## Open decisions
Dataset choice (licence first); feature rate and encoder size.
