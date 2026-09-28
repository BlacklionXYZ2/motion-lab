# M5 — Annotation factory + emotion/style

**Status:** brief.
**Lane:** make/train (owner).

## Goal
Build the labelling tooling, edit the datasets, and train emotion/style conditioning. This is where the "datasets don't contain all the data I'm looking for" problem gets solved.

## What you build
- **Clip scrubber/segmenter** — scrub through mocap clips, mark intent segments and boundaries, review/override heuristic labels.
- **Emotion/style vectors** — continuous dimensions (valence, arousal, confidence, curiosity, energy, …); heuristic pre-labels (e.g. energy from velocity statistics) reviewed by hand; adopt style labels from style-labelled corpora where available.
- **Export** — labelled windows into the training format.

## What you train
- Emotion/style conditioning added to fusion; ablation: the same intent under different emotion must differ measurably and stay stable.

## Gate
Measurable, stable style modulation; label coverage documented.

## Open decisions
Label dimensions and scale; how much manual labelling is acceptable; which source corpora contribute emotion/style.
