# Decision Models as Universal Tool Selectors — paper sketch (v2)

Authors: Aziz Muminov, Alexandra Ilina. Status: sketch for drafting; numbers are final unless marked.

## Abstract (draft)

Agents increasingly work through many tools: a coding agent connected to a company's MCP servers can face hundreds or
thousands of tool definitions. Loading them all is ruinously expensive (about 182k tokens per request for 613 tools,
8–9× the cost per run of searching), so shipped harnesses search instead, mostly with BM25 (Codex, Anthropic's tool
search, Hermes Agent). Lexical search is cheap but imprecise, and every extra search is a full agent turn. We propose
decision models (DMs), which return a probability per option in one forward pass, as general-purpose tool selectors.
Our selector, built on Microsoft Decision 1, chooses the service, then the tools, with a "none" option and usage-based
recommendations, and needs about three DM calls per search. On the real tool definitions of 30 MCP servers (1,076 tools)
and 37 coding tasks with simulated back ends, it ranks far better than BM25 offline (right tool first 0.92 vs 0.54).
Inside a shipped agent, Hermes Agent, in pre-planned, same-time paired tests it used 12.6% fewer tokens with DeepSeek
V4.1 Flash (555 pairs, p = 0.0002) and 9.7% fewer with GPT-6.1 Sol (185 pairs, p = 0.004), with 35–60% fewer searches
and equal pass rates. Its per-search fee made runs 20% more expensive with the inexpensive model and cost-neutral with
the frontier model; savings grow with the agent model's price and how much it searches.

## 1. Introduction
- Tool explosion: MCP servers expose 10s–100s of tools each; agents connect to many. Tool choice is now a bottleneck.
- Option A, dump everything: context cost explodes (opencode, 613 tools: largest prompt ~182k tokens vs 12–16k with
  search; 8–9× cost per run at the same pass rate).
- Option B, search (what ships): Table 1. BM25 dominates; it is imprecise, so agents search repeatedly; each search is a
  full LLM turn re-reading the whole context — costly with frontier models.
- Decision models: probabilities over named options from one forward pass, no generated text, cheap and fast. Thesis:
  DMs as a drop-in, harness-agnostic tool selector.
- Contributions: (1) the DM selector design and what makes it work; (2) a test suite on real MCP tool definitions;
  (3) a measurement protocol for token differences in agents (drift, pairing, clustering, written plans);
  (4) results across two agent models.

## 2. Background: how harnesses select tools; decision models
- Table 1 (keep, update): native loading, Claude Code names + keyword search, BM25 (Codex, Anthropic, Hermes), Copilot
  virtual tools, embeddings/LLM routers (Goose, removed), DM (this paper).
- Decision models: Microsoft Decision 1 (post-trained from Qwen3.5-9B, $0.04/M input tokens, up to 255 options per
  question, multiple-choice or yes/no questions). Brief note: TypeSafe Jev matched it offline (right tool first 0.875
  each at 1,076 tools), twice as fast (0.27 vs 0.57 s) but ~45% more per search; we use Microsoft Decision 1.

## 3. The DM tool selector (selector F)
- Figure (update the old flow figure): query → service question → tool questions → answer.
- Service step: one multiple-choice question over service cards (+ "none"); services until 90% of the probability
  (max 3); a service named in the query is always included. Card quality matters (GitHub/Git/GitLab fix).
- Tool step, per chosen service (parallel): "which tool best serves the request" → 5 tools with descriptions;
  "which tool would an expert call first" → up to 3 prerequisites as names; the service's most-used tools as names;
  at most 15 tools; nothing repeated within a run.
- Why this shape (from the analyses): agents work service by service and use ~3–4 tools per service, most of them for
  later steps; descriptions are expensive and names cheap; the DM's probabilities are too peaked to size lists, so the
  list size is fixed and filled by usage and prerequisites.
- Cost: ~3 DM calls per search in 2 sequential rounds (~1.1 s), ~$0.30 per 1,000 searches.

## 4. Experimental setup
- Catalog: real tools/list output of 30 MCP servers; 613 (12 popular servers) and 1,076 tools with decoys
  (GitLab, CloudWatch, EKS). Simulated back ends (rules per task, organisation-wide facts).
- Tasks: 38 symptom-first coding tasks (single, multi-service, absent-information, controls), hidden tests + required
  writes + no destructive calls. One task (k8s_crashloop) excluded from the main tests for both setups (its grading rule
  forbids a reasonable read-only exec).
- Harnesses: Hermes Agent 0.19.0 (main; its own tool-search bridge, ranker swapped behind it), opencode 1.19.4
  (supporting). Sandbox isolation (App.).
- Agent models: DeepSeek V4.1 Flash (pinned provider, T=0.6), GPT-6.1 Sol (pinned to OpenAI).
- Baseline: Hermes's built-in BM25, unmodified.

## 5. Measuring token differences in agents (methods; place before results)
- Run-to-run noise: one slot can swing ~1.6× (SD of per-slot log ratio 0.55 DeepSeek, 0.30 Sol).
- Drift: an A/A test (BM25 vs BM25 a few days apart) differed by ~10% — batches run at different times are not
  comparable.
- Protocol: same-time pairs per (task, repeat) slot; per-task mean of log ratios; one-sample t-test over the 37 tasks;
  task-level bootstrap interval; written analysis plan before each batch; interim looks logged; runs to completion.
- Power: 15 repeats for DeepSeek (98% for a 12.5% effect), 5 for Sol (≈98% at its noise).

## 6. Results
6.1 Offline: selector design study (13 variants + held-out check). Table: right tool first / top 5 / top 15,
   coverage of agents' next tools, named-service accuracy, cost. Findings: card quality and service nucleus matter
   most; "none" is free (95% correct, no false "none"); prompt wording matters little; yes/no alone and sliding
   windows are worse or 11–17× costlier; held-out (500 agent searches from another batch): F best coverage
   (0.845 / all-covered 0.79) vs BM25 0.806 / 0.754 and top-5 0.79 vs 0.71.
6.2 Offline: DM vs shipped mechanisms and an LLM router (keep from v1): right tool first at 613 / 1,076 tools —
   DM 0.93 / 0.88, LLM router 0.86 / 0.82 at 3× the cost and 2.4× the latency, BM25 0.54 / 0.49, embeddings
   0.51 / 0.47, Claude Code style 0.48 / 0.45.
6.3 Main experiment, DeepSeek, 15 repeats (Table): tokens 0.874× [0.82, 0.93], p = 0.0002; searches 11.1 → 7.1;
   LLM calls 0.92×; pass 550/555 each; cost 1.20× (DM fee); wall time equal. Figure: per-slot token ratios.
6.4 Second agent model, GPT-6.1 Sol, 5 repeats (Table, side by side with a pre-specified random 5-of-15 DeepSeek
   subsample): tokens 0.903× [0.84, 0.96], p = 0.004; searches 5.5 → 2.2; LLM calls 0.88×; pass 172 vs 173;
   cost 0.99× (break-even). DeepSeek at 5 repeats: 0.915× [0.84, 1.01], p = 0.07 (noisier model, as the power
   estimate predicted).
6.5 Failure analysis: the Git/GitHub/GitLab confusion (earlier selector failed gh_open_pr 3/15; queries naming one
   service: DM picks it 97.9%, errors almost all within the Git family, caused by service cards); fixed cards +
   named-service rule → 15/15. Remaining failures are task-hard for both rankers (prom_threshold, loki_errors).
6.6 Dumping all tools (opencode, 613 tools, keep from v1): every condition passes; native loading's largest prompt
   ~182k vs 12–16k; cost per run $0.072 vs $0.008–0.009.
6.7 Cost by agent model: measured DeepSeek +20%, Sol ±0%; projection table for other prices (mean token mix).

## 7. Discussion
- Why fewer searches matter more than cheaper searches (each search = a turn re-reading the context).
- When the DM pays off: expensive agents, many tools, search-heavy work; when not: cheap agents, efficient agents.
- "Universal": one API call, no training, works with any harness and catalog; limits below.
- What the DM is good and bad at: comparing options (yes) vs judging tools in isolation (no); depends on cards.

## 8. Threats to validity
- Simulated services; coding tasks only; two agent models; one harness for the main tests.
- Selector designed in exploratory runs (a separate long task) and an offline study before the tests; held-out check.
- Usage lists from the same suite (leave-one-task-out).
- Exclusion of k8s_crashloop; interim looks logged; container restarts and rate-limit reruns logged.

## 9. Related work (keep, update): tool retrieval (ToolLLM, AnyTool, RAG-MCP, MCP-Zero, ToolRet), MCP benchmarks,
   vendor reports (Stacklok, Arcade), decision models (AA Decision Index).

## 10. Conclusion

## Appendices
A. Sandbox hardening. B. Task set and catalog tables. C. Analysis plans (verbatim) and interim-look logs.
D. Earlier designs: the recommendations design (one service, 5 + 10 names) under the same protocol: 0.91×
   [0.85, 0.97], p = 0.009, 555 pairs — a second, independent confirmation. E. Six-ranker comparison from the first
   version (3 repeats, not same-time): random ranking 1.57× BM25 (ranking matters); other differences within the
   drift we later measured. F. Next-tool prediction in one real session (DM selects, LLM calls).

## Dropped from v1
Old DM + none headline (not same-time; within drift), DM v1/v3 naming, the long-horizon runs (exploratory),
the per-run cost projection table based on interim data.
