# external: downloaded third-party models

Git-ignored except this file. Nothing here is our work. It is kept for evaluation, unmodified.

## ardy/ (downloaded 2026-10-26 at the owner's request)

NVIDIA's ARDY real-time motion model, as converted to ONNX by a third party, TREE Industries on
Hugging Face. **This is not an NVIDIA release.** Its conversion is checked against NVIDIA's
original before we rely on it.

| Folder | From | Licence | What |
|---|---|---|---|
| `ARDY-Core-RP-20FPS-Horizon40-ONNX/` | huggingface.co/TREEIndustries/ARDY-Core-RP-20FPS-Horizon40-ONNX | NVIDIA Open Model License (weights); see its LICENSE, NOTICE, THIRD_PARTY_NOTICES | encoder, denoiser and decoder graphs. `fp32/` is the exact reference; `directml-mixed-fp16/` is the runtime build. 27-joint Core skeleton, 20 fps, 40-frame horizon |
| `Llama-3-ARDY-Text-Encoder-ONNX/` | huggingface.co/TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX | Meta Llama 3 Community License and Acceptable Use Policy (credit: "Built with Meta Llama 3"); LLM2Vec adapters MIT | prompt → 4096-value text feature; INT4 weights, 64-token input |

Each folder's own `SHA256SUMS.txt` was checked after download (2026-10-26):
- **Model files:** every one matches, in both repos.
- **The text encoder's `README.md`:** doesn't match its listed hash. It's the Hugging Face model
  card (plain text, 4,431 bytes), presumably edited after the checksum list was written. It isn't
  a model file.
- **The converter's own fp16 against fp32 check** (`VALIDATION.md`, RTX 4080 Laptop, DirectML):
  mean absolute differences 0.004 to 0.014 per graph, maximum 0.12 to 0.32. Those are
  normalised features, one pass each. Not yet checked over long rollouts.

The original model is at huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40 (NVIDIA Open Model
Agreement), with code at github.com/nv-tlabs/ardy (Apache-2.0). It's described in the paper
arXiv 2607.08741 (SIGGRAPH 2026).
