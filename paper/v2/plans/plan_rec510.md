# Analysis plan: DM recommendations 5+10 vs Hermes BM25 (written before the final analysis)

Written: 2026-11-07T01:52Z, with 254 of 1,110 planned runs finished. Interim looks so far: ratio and
p-values were viewed at 103 slots (0.89x, p about 0.02). From now on only progress counts are viewed until the end.

## Design (fixed)
- Hermes Agent 0.19.0, task set v2, 1,076 tools, schema validation, deny list, connected-services note.
- DeepSeek V4.1 Flash pinned to DeepInfra fp8, temperature 0.6, seed 7.
- 37 tasks (k8s_crashloop excluded for both setups before the final analysis: its grading rule forbids a reasonable
  read-only exec), 15 repeats, two setups per (task, repeat) slot queued back to back: 555 planned slots.
- Usage lists for the recommendations are leave-one-task-out (built from the paper's main run, excluding the task).

## Primary analysis (the verdict)
- Measure: total tokens per run (input + cached input + output over all LLM calls).
- Per slot: log(tokens_recommendations / tokens_BM25). Per task: mean over its slots (37 values).
- Primary test: one-sample t-test of the 37 task means against 0, two-sided, alpha 0.05.
- Interval: task-level bootstrap (resample tasks, 10,000 resamples, seed 1) of the geometric-mean ratio, 95%.
- Verdict: "fewer tokens" only if the test gives p < 0.05 AND the interval excludes 1; otherwise "not established".
- Only complete slots (both setups finished) count; no other exclusions after this point.

## Secondary (reported, not used for the verdict)
- Cost per run including the DM fee (paired ratio, same method); pass rate per setup; searches; LLM calls; wall time.
- Slot-level paired t-test and Wilcoxon (they overstate the evidence because repeats of a task are correlated).

## Interim looks log
- 2026-11-07T02:13Z: look 2 at 347 runs (user request). Commitment: the batch runs to all 555 slots regardless; the verdict uses only the final analysis.
- 2026-11-07T03:27Z: look 3 at 669 runs (user request). Commitment unchanged.
- 2026-11-07T05:11Z: container restart interrupted the batch at 1076 runs; resumed with the same command (finished runs kept, interrupted ones rerun). No design or analysis change.
- 2026-11-07T05:25Z: 2 runs (prom_threshold r14 and sentry_fix r14, DM recommendations) ended with HTTP 429 rate-limit errors from the provider and recorded no tokens; treated as unfinished like the restart-interrupted runs and rerun before any final analysis. Their BM25 partners are kept.
