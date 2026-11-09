# Analysis plan: DM selector F vs Hermes BM25 with GPT-6.1 Sol (written before any run)

Written: 2026-11-08T07:22Z, before the first run of this batch. The earlier 2-task Sol check (results/sol_check)
used tasks chosen because they favour F; it is not part of this test.

## Design
Identical to results/final_franken except the agent model and the number of repeats: GPT-6.1 Sol
(openai/gpt-6.1-sol, pinned to provider OpenAI, default sampling), 37 tasks (k8s_crashloop excluded), 5 repeats, BM25 and
F queued back to back per (task, repeat) slot: 185 slots, 370 runs. Selector F unchanged (harness/selectors_real.py
dm_franken). Cost per run now includes cache-write tokens at $2.50/M (fix verified against the key's charge).
The batch runs to completion regardless of interim numbers; provider-error runs with no tokens are rerun.
Power (simulated from measured variance): about 98% for a 12.5% token reduction at Sol-like noise (SD 0.30), 80% at
DeepSeek-like noise (SD 0.55).

## Primary analysis (the verdict, for Sol)
Same as results/final_franken: per-task mean of log(tokens_F / tokens_BM25) over its slots (37 values), one-sample t-test,
two-sided, alpha 0.05, plus task-level bootstrap 95% interval (10,000 resamples, seed 1). "Fewer tokens" only if p < 0.05
and the interval excludes 1.

## Secondary
Cost per run including the DM fee and cache writes (paired ratio, same method); pass rate and McNemar; searches; LLM
calls; wall time.

## Comparable DeepSeek table (descriptive; DeepSeek's own verdict remains the full 15-repeat result)
From results/final_franken, for each of the 37 tasks draw 5 of the 15 repeats at random with
random.Random(20261108).sample(range(15), 5), drawn task by task in alphabetical task order; both setups of a drawn slot
are kept. The same metrics are computed on that subsample and shown next to Sol.

## Interim looks log
- 2026-11-08T07:36Z: look 1 at 103 runs (user request). Commitment: the batch runs to all 185 slots regardless.
- 2026-11-08T07:44Z: look 2 at 184 runs (user request). Commitment unchanged.
- 2026-11-08T07:59Z: look 3 at 321 runs (user request). Commitment unchanged.
