# decision_models_for_tool_calling

Code, benchmark and paper for **"Can Decision Models Replace Tool Calling for Coding Agents?"**
(Aziz Muminov, Alexandra Ilyina). Paper: [`paper/coding_agents.pdf`](paper/coding_agents.pdf).

Coding agents carry large tool catalogs, and every request re-sends every tool definition. We test whether a
*decision model* (DM), a model that returns a probability over caller-supplied options without generating text,
can take the catalog's place. The agent gets one **universal tool**: `find_tool(intent)` (a DM ranks the whole
catalog against the stated intent) and `call_tool(name, arguments)`.

In a sandboxed coding-agent benchmark (opencode, 30 Python tasks with hidden tests, a fictional 100/300-tool
internal catalog, two LLMs, 1,260 runs), the universal tool with a one-line list of services matched loading
every definition (98–99% vs. 97–99% on tasks that need the catalog) while reading 81% (100 tools) and 92–93%
(300 tools) less context per run.

## Layout

```
harness/                 the coding-agent benchmark (main experiment)
  catalog.py             fictional "Quillmark" catalog: 12 services / 100 tools (+20 services / 300 tools)
  mcp_server.py          serves the catalog to the agent: native, filtered, or universal (find_tool/call_tool)
  runner.py              runs opencode on each task x condition x model x repeat, inside the sandbox, and scores it
  llm_proxy.py           local proxy that holds the API key and admits only the experiment's models
  sandbox_setup.sh       one-time OS sandbox setup (unprivileged user, root-only experiment files)
  build_tasks_v2.py      generates the 30 tasks in tasks_v2/ (repo, hidden tests, world file, reference solution)
  validate_tasks_v2.py   checks every task: hidden tests fail on the original code, pass with the reference
  analyze_v2.py          report for the 100-tool runs; analyze_sweep.py for 100 vs 300 tools
  tasks/, build_tasks.py, analyze.py   the 5-task pilot
dmtc/, configs/, tests/  earlier tool-selection studies (BFCL padded, When2Call, SynthHolster, SWE replay)
scripts/                 analyses for those studies and the session replay (meta_*.py)
paper/                   LaTeX source, figures and figure scripts
  coding_agents.tex      multi-file source (uses shared/ and figures/)
  coding_agents_full.tex single self-contained source (figures embedded as PGF); regenerate with flatten_single_file.py
```

## Running the benchmark

Requirements: Linux, root (for the sandbox user), Python 3.11+, Node.js, an OpenRouter API key.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt mcp uvicorn starlette
(cd harness && npm ci)                       # opencode 1.19.4
sudo bash harness/sandbox_setup.sh           # creates user `agentbox`, /opt/agentbox, /srv/agentbox/work
.venv/bin/python harness/validate_tasks_v2.py
```

opencode installs plugins and fetches its model list on first start. The runner gives each run a copy of
`/opt/agentbox/xdg-template/{config,cache}` so the sandboxed agent never needs the network: start opencode
once as a normal user and copy `~/.config/opencode` and `~/.cache/opencode` there.

Put `OPENROUTER_API_KEY=...` in `.or_env` (git-ignored; keep it root-only), then:

```bash
set -a; source .or_env; set +a
.venv/bin/python harness/runner.py --models deepseek,glm --repeats 3 --workers 6 --budget 10 --out results/harness_v2
.venv/bin/python harness/runner.py --conditions native,universal,universal_index --models deepseek,glm \
    --repeats 2 --catalog-size 300 --workers 6 --budget 10 --out results/harness_v2_300
.venv/bin/python harness/analyze_v2.py results/harness_v2
.venv/bin/python harness/analyze_sweep.py results/harness_v2 results/harness_v2_300
```

The runner resumes from `rows.jsonl` and stops at `--budget` (USD). The full 100-tool grid (900 runs) cost
about $9 and the 300-tool sweep (360 runs) about $10 on OpenRouter.

**Security note.** Run agents only inside the sandbox. In an earlier pilot that relied on opencode's own
permissions, the agent read the hidden tests and answer files through absolute paths (paper, App. A).

## Building the paper

```bash
cd paper && pdflatex coding_agents_full.tex && pdflatex coding_agents_full.tex
```

The figure scripts (`make_figures*.py`) read run records from `results/`, which are not in this repository.
