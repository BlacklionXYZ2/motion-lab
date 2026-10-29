# Motion core: ARDY downloaded, its runtime environment set up (paused for a permission)

Model name: **Opus 5.5**. Checkpoint **2026-10-26 18:20 UTC**. Status: **set up; paused before
running any third-party code**.

The owner asked:
- to download the ONNX conversion of NVIDIA's ARDY;
- then: "Could you build the motion model settings now? ... set up the pipeline to be compatible
  with ARDY? Is there a way to skip out on the text encoder ... I still want to minimise the
  memory footprint."

## Done (all in motion-lab; nothing in SORA's code yet)

- **`external/ardy/`** (git-ignored; `external/README.md` records sources, licences and checks):
  - TREE Industries' ONNX conversion of ARDY Core Horizon40, fp32 and mixed fp16;
  - the 4-bit Llama-3 LLM2Vec text encoder ONNX;
  - NVIDIA's original checkpoint (`checkpoints/ARDY-Core-RP-20FPS-Horizon40`, ungated);
  - NVIDIA's reference code, cloned at `e706e27` (Apache-2.0), read but not run.
  - Every model file matches its published SHA-256. One text-encoder README (a model card)
    differs.
- **`ardy-runtime/.venv`:** an isolated environment with CPU PyTorch 2.14.1, ONNX Runtime
  DirectML 1.24.4 (the DirectML provider is available), NumPy 1.26.4 and ARDY's pure-Python
  dependencies. motion-lab's training environment is untouched.
- **`.gitignore`:** ignores `external/*` except its README.

## Findings

- **What the ONNX files are:** only the networks (history encoder, denoiser, decoder). The
  streaming loop, sampler, motion representation and retargeting are in NVIDIA's Python
  (`ardy/model/ardy_model.py` and the rest); the converter's own runtime isn't published.
- **The text encoder can be skipped at run time:** the denoiser takes one pooled 4,096-value
  text feature per prompt (`text_feat` `[1, 1, 4096]`), so a fixed phrase table can be encoded
  once.
- **The Core skeleton** (`model_contract.json`) has 27 joints: Hips, Spine to Spine3, Neck,
  Head, the shoulders, arms, forearms and hands, plus HandEnd and Thumb1 for each hand, then the
  legs, feet and toe bases. 20 fps.

## Paused

Loading NVIDIA's own code was blocked by the permission system (running code from an external
repository). It waits for the owner's decision.

Nothing committed.
