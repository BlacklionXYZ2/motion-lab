"""Encode motion phrases into ARDY text features once, so run time needs no text encoder.

    .venv\\Scripts\\python text_table.py --check     # reproduce the encoder's reference inputs
    .venv\\Scripts\\python text_table.py phrases.txt out.npz [--provider cpu|dml]

ARDY's denoiser reads one pooled 4,096-value text feature per prompt. The encoder that makes it
(Llama 3 8B with LLM2Vec, here TREE Industries' INT4 ONNX in ``external/ardy``) is about 4.7 GB,
so SORA encodes its fixed phrase table here, ahead of time, and loads only the table (16 KB a
phrase as float32) at run time.

Input format, as LLM2Vec prepares it (``ardy/model/llm2vec/llm2vec.py``) and as the encoder's
reference records show: the Llama 3 user header around the prompt, end-of-turn after it, left
padding to 64 tokens with the end-of-turn token, attention over the whole formatted prompt, and
mean pooling over the prompt's own tokens plus end-of-turn.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENCODER = HERE.parent / "external" / "ardy" / "Llama-3-ARDY-Text-Encoder-ONNX"
LENGTH = 64
PAD = 128009  # <|eot_id|>
HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"


def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(ENCODER / "tokenizer" / "tokenizer.json"))


def prepare(tok, text: str) -> dict[str, np.ndarray]:
    """``input_ids``, ``attention_mask`` and ``embed_mask``, each ``(1, 64)`` int64."""
    full = tok.encode(HEADER + text.strip() + "<|eot_id|>").ids
    header = tok.encode(HEADER).ids
    if len(full) > LENGTH:
        raise ValueError(f"prompt too long ({len(full)} tokens, at most {LENGTH}): {text!r}")
    pad = LENGTH - len(full)
    ids = [PAD] * pad + full
    attention = [0] * pad + [1] * len(full)
    embed = [0] * (pad + len(header)) + [1] * (len(full) - len(header))
    return {name: np.asarray([values], dtype=np.int64)
            for name, values in (("input_ids", ids), ("attention_mask", attention),
                                 ("embed_mask", embed))}


def session(provider: str):
    import onnxruntime as ort

    providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if provider == "dml" \
        else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(ENCODER / "onnx" / "text_encoder_int4.onnx"),
                                providers=providers)


def check(provider: str) -> None:
    """Our tokenisation against the encoder's five reference records, then the INT4 output
    against their full-precision embeddings."""
    records = json.loads((ENCODER / "validation" / "text_encoder_reference.json")
                         .read_text("utf-8"))["records"]
    tok = tokenizer()
    for record in records:
        ours = prepare(tok, record["prompt"])
        for key in ("input_ids", "attention_mask", "embed_mask"):
            if ours[key][0].tolist() != record[key]:
                sys.exit(f"{key} differs for {record['prompt']!r}")
    print(f"tokenisation matches all {len(records)} reference records")
    run = session(provider)
    for record in records:
        start = time.time()
        out = run.run(None, prepare(tok, record["prompt"]))[0][0]
        ref = np.asarray(record["embedding"], dtype=np.float32)
        cos = float(out @ ref / np.linalg.norm(out) / np.linalg.norm(ref))
        print(f"{record['prompt']!r}: cosine {cos:.4f}, {time.time() - start:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("phrases", type=Path, nargs="?")
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--provider", choices=("cpu", "dml"), default="cpu")
    args = parser.parse_args()
    if args.check:
        check(args.provider)
        return
    if not (args.phrases and args.out):
        parser.error("give a phrase file and an output path, or --check")
    phrases = [line.strip() for line in args.phrases.read_text("utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
    tok, run = tokenizer(), session(args.provider)
    features = np.stack([run.run(None, prepare(tok, p))[0][0] for p in phrases])
    np.savez(args.out, phrases=np.asarray(phrases), features=features.astype(np.float32))
    print(f"{len(phrases)} phrases → {args.out} ({features.nbytes / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
