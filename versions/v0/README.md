# v0: experiments before v1 (2026-10-23)

The first M1 training runs, kept as the owner asked. **No code snapshot:** the code that produced
these runs was changed in place before the versions were separated (2026-10-25), so v0 holds what
they produced, not what produced them.

| Run (`runs/`) | What it was | Closest code |
|---|---|---|
| `baseline` | **The first real training run:** LAFAN1 only, every signal neutral, 20,000 steps (3.5 h), 828,643 parameters. Best at step 14,500 | v1's code without the hips-velocity loss, the system-memory storage and the per-source weights. Same architecture (separate encoders, FiLM, fusion, GRU 1×256) |
| `smoke-2026-10-23`, `smoke2-…`, `smoke3-…` | 300 to 600-step tests of the first architecture (533,067 parameters: one shared signal encoder, one-layer encoder and decoder). smoke3 added absolute hips height | earlier than v1: the one-layer design |
| `smoke4-fusion-2026-10-23` | 600-step test of the separate-encoder, FiLM and fusion design | v1's architecture |
| `loadcheck-…`, `init-test`, `resume-test-straight`, `resume-test-paused` | Tests of data loading, `--init-from` and exact pause and resume | v1's training script |

- `configs/baseline.json`: the baseline's config. Its data path (`data/processed/lafan1`) is now
  v1's LAFAN1 data: `versions/v1/data/processed/lafan1`, the same conversion.
- `outputs/`:
  - the baseline's and the smoke runs' 60 s evaluations (`eval-*.json`);
  - the resume test's logs;
  - `lafan1_alignment_probe.pt`, an early calibration probe.

**Baseline results** (60 s free rollouts on LAFAN1's test performer):
- it was stable, with joints within limits (1.0%);
- the error at 0.5 s was 11° per joint and 8.5 cm;
- but its hips height buzzed about 2.3 cm per frame (jitter 15,634 m/s³, against the data's 499),
  and its motion was much calmer than the data's.

v1 added the hips-velocity loss that fixed the buzz. Details are in the changelog
[`2026-10-23T235900+0100_MOTION-M1-BASELINE-RUN_Opus-5.5.md`](../../changelog/2026-10-23T235900+0100_MOTION-M1-BASELINE-RUN_Opus-5.5.md).
