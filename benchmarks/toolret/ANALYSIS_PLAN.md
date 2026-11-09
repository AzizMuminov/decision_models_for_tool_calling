# ToolRet full run: analysis plan (written before the run)

Benchmark: ToolRet (Shi et al., Findings of ACL 2025), original documents (HF mangopy/ToolRet-Queries and -Tools), all
7,961 queries, all 44,453 tools. Two settings, as in the paper: with instruction (Table 5; also Tool-REX's ToolRet rows)
and without instruction (Table 4; query alone, an empty query becomes "NONE").
Candidates: union of BM25 top 50 (bm25s, English stopwords, full corpus) and text-embedding-3-small top 50 (tool text =
first 1,000 characters of the documentation), interleaved, 100 tools; query text as in the setting.
Methods, fixed on the 385-query pilots (stratified 11 per task, seed 7): A = Microsoft Decision 1, one multiple-choice call
over the 100 candidates, plain card (350 characters), prompt "A retrieval system is ranking tools for a user's task. Which
tool is most relevant to it?"; E = same prompt, full documentation (1,500 characters), 4 parallel calls of 25, top 6 of each
to a final call. With instruction: A and E. Without instruction: A only (budget).
Metric and averaging as the paper: NDCG@10, Recall@10, Completeness@10 per task; mean of tasks within each category; headline
= mean of the three categories. Reported on all 7,961 queries and on the 7,576 not in the pilot. Failed DM calls are retried
5 times; still-failed queries keep the first-stage order and are counted. Published rows are quoted, not re-run.
Check before the DM: our BM25 must reproduce the paper's BM25s (with 36.46, without 22.32) within 0.5.
