# Decision Models as Universal Tool Selectors

Code, data and paper for using a decision model (Microsoft Decision 1, via OpenRouter's Decisions API) as an agent's
tool selector. The selector chooses MCP services first, with a "none" option, and then tools. It is compared with
the built-in BM25 tool search of Hermes Agent on 37 coding tasks over the real tool definitions of 30 MCP servers, and
with published retrievers on the ToolRet benchmark.

Paper: [`paper/dm_tool_selectors.pdf`](paper/dm_tool_selectors.pdf). Authors: Aziz Muminov, Alexandra Ilina.

## Layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX source, PDF, bibliography, figures and `make_figures.py`, which rebuilds the figures from `results/` |
| `harness/` | The experiment harness: tool catalog harvester and snapshots (`real_catalog*/`), simulated MCP server (`mcp_server_real.py`), selectors (`selectors_real.py`, selector F is `dm_franken`), Hermes Agent runner (`run_hermes.py`), task sets (`tasks_real_v2/` is the one in the paper), offline selector study (`selector_study.py`) and analysis scripts |
| `harness/hermes_patch/tool_search.py` | Hermes Agent 0.19.0's `tool_search` module with the ranker slot patched to call the selector's `/rank` endpoint |
| `dmtc/` | Client for decision-model APIs |
| `results/final_franken/` | DeepSeek V4.1 Flash, 37 tasks × 15 repeats: analysis plan, final analysis, one row per run, and each run's search and tool-call log |
| `results/final_franken_sol/` | GPT-6.1 Sol, 37 tasks × 5 repeats, same files, plus the seeded 5-of-15 DeepSeek subsample |
| `results/selector_study/` | Offline selector study: the variants of Table 2 and the held-out check of Figure 3 |
| `results/retrieval_real/` | Offline retrieval of Table 3 |
| `benchmarks/toolret/` | ToolRet run of Table 4: plan, scripts, summaries and per-query rankings |
| `configs/`, `scripts/`, `tests/` | Earlier benchmark configurations, helper scripts and tests |

## Reproducing

Requirements: Python 3.11+, `pip install -r requirements.txt`, and an OpenRouter key in `OPENROUTER_API_KEY` for any
step that calls the decision model or an agent model.

- **Statistics and figures from the logged runs**, no API calls:
  `python harness/paper_stats_5rep.py` and `python paper/make_figures.py`.
  Each `FINAL_ANALYSIS.txt` holds the analysis exactly as specified in that batch's `ANALYSIS_PLAN.md`.
- **Agent runs** need Hermes Agent 0.19.0 with `harness/hermes_patch/tool_search.py` installed over its
  `tools/tool_search.py`, and the sandbox from `harness/sandbox_setup.sh`. Example, one task and both selectors:
  `python harness/run_hermes.py --tasks gh_open_pr --conditions hermes_default,hermes_dmfrankenbridge --models deepseek --tasks-dir tasks_real_v2 --extra --validate --deny --preamble --pin-provider default --temperature 0.6 --out results/example`.
- **Selector study:** `python harness/selector_study.py`, add `--heldout` for the held-out check.
- **ToolRet:** download `mangopy/ToolRet-Queries` and `mangopy/ToolRet-Tools` from Hugging Face, then run
  `benchmarks/toolret/bm25_repro.py`, followed by `full_run.py` with and without `NOINST=1`; see
  `benchmarks/toolret/ANALYSIS_PLAN.md`.

## Not included

Raw run directories (agent logs, session databases, repository snapshots), decision-model response caches, tool
embeddings and BM25 indexes are left out for size: about 7 GB. They can be regenerated with the scripts above.
