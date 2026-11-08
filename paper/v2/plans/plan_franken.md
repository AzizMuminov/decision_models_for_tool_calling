# Analysis plan: combined DM selector F vs Hermes BM25 (written before any run)

Written: 2026-11-08T02:21Z, before the first run of this batch.

## Selector F (fixed before the run; chosen in an offline selector study and checked on held-out queries)
Microsoft Decision 1. Fixed service cards (GitHub lists branches/commits/files; Git is the local copy only; GitLab is
legacy hosting); services until 90% of the probability, at most 3; a 'none' option (empty answer, no BM25 fallback, when
P(none) >= 0.4); a service the query names is always included; per search the 5 tools that best serve the query with
descriptions, plus as names up to 3 prerequisites ('which tool would an expert engineer call first') and the services'
most-used tools (leave-one-task-out lists from the paper's main run), at most 15 tools; tools already shown in the run are
not repeated. Code: harness/selectors_real.py dm_franken; condition hermes_dmfrankenbridge.

## Design
Same as results/final_rec510: Hermes Agent 0.19.0, task set v2, 1,076 tools, schema validation, deny list,
connected-services note; DeepSeek V4.1 Flash pinned to DeepInfra fp8, temperature 0.6, seed 7; 37 tasks (k8s_crashloop
excluded for both setups, as before); 15 repeats; BM25 and F queued back to back per (task, repeat) slot: 555 slots.
The batch runs to completion regardless of interim numbers. Runs that end in provider errors with no recorded tokens
are rerun, as before; nothing else is excluded.

## Primary analysis (the verdict)
- Measure: total tokens per run. Per slot log(tokens_F / tokens_BM25); per task the mean over its slots (37 values).
- Test: one-sample t-test of the 37 task means against 0, two-sided, alpha 0.05; task-level bootstrap 95% interval
  (10,000 resamples, seed 1). Verdict "fewer tokens" only if p < 0.05 and the interval excludes 1.

## Secondary (reported, not part of the verdict)
- Pass rate per setup and McNemar test on discordant slots; failures listed per task (in particular gh_open_pr).
- Cost per run including the DM fee; searches; LLM calls; wall time; slot-level tests.
- Comparison with the earlier recommendations design (results/final_rec510) is descriptive only (different batches).

## Interim looks log
- 2026-11-08T02:34Z: container restart interrupted the batch at 61 runs; resumed with the same command (finished runs kept, interrupted ones rerun). No design or analysis change.
- 2026-11-08T02:35Z: look 1 at 62 runs (user request). Commitment: the batch runs to all 555 slots regardless.
- 2026-11-08T02:45Z: look 2 at 101 runs (user request). Commitment unchanged.
- 2026-11-08T03:11Z: look 3 at 225 runs (user request). Commitment unchanged.
- 2026-11-08T04:56Z: look 4 at 661 runs (user request). Commitment unchanged.
- 2026-11-08T06:35Z: look 5 at 1082 runs (user request). Commitment unchanged.
