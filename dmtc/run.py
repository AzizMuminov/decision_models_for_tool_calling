"""Runs the ablation grid defined in a YAML config.

    python -m dmtc.run configs/ablation.yaml                # everything in the config
    python -m dmtc.run configs/ablation.yaml --only dm_gate --limit 20

Each (benchmark, category, condition, llm, dm) cell is written to its own JSONL file; rerunning
skips examples already present, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from dmtc.backends.dm import make_dm
from dmtc.backends.llm import OutOfCredits, make_llm
from dmtc.data import BFCL_CATEGORIES, Example, iter_jsonl, load, subsample, subsample_groups
from dmtc.eval.bfcl_checker import score_bfcl, tool_selection_correct
from dmtc.pipeline import Pipeline, when2call_label

NO_LLM_CONDITIONS = {"dm_full"}
NO_DM_CONDITIONS = {"llm_native", "oracle_select"}


def score(ex: Example, pred: dict[str, Any], label: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"called": bool(pred["calls"])}
    out["call_decision_correct"] = out["called"] == ex.should_call
    if ex.benchmark == "bfcl":
        out.update(score_bfcl(ex, pred["calls"]))
        out["selection_correct"] = tool_selection_correct(ex, pred["selected_tools"])
        # Same question for every condition, judged from the calls actually emitted (LLM or DM):
        # are the called function names exactly the gold ones (as a multiset)?
        if ex.ground_truth:
            gold = sorted(next(iter(g)) for g in ex.ground_truth)
            out["call_names_correct"] = sorted(next(iter(c)) for c in pred["calls"]) == gold
    else:
        out["pred_label"] = label
        out["correct"] = label == ex.label
    return out


def cells(cfg: dict[str, Any], only: set[str] | None):
    for block in cfg["runs"]:
        llms = block.get("llm", [None])
        dms = block.get("dm", [None])
        for cond, llm_name, dm_name in itertools.product(block["conditions"], llms, dms):
            if only and cond not in only:
                continue
            llm_name = None if cond in NO_LLM_CONDITIONS else llm_name
            dm_name = None if cond in NO_DM_CONDITIONS else dm_name
            yield cond, llm_name, dm_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*", help="run only these conditions")
    ap.add_argument("--limit", type=int, help="override limit_per_category")
    ap.add_argument("--out", help="override output directory")
    ap.add_argument("--force", action="store_true", help="resume even if the config changed")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out or cfg.get("output_dir", "results/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = out_dir / "config.yaml"
    if saved.exists() and not args.force:
        old = yaml.safe_load(saved.read_text())
        # Rows from different pipeline/model settings must not be mixed in one cell file.
        for k in ("seed", "pipeline", "llms", "dms", "judge", "when2call_path"):
            if old.get(k) != cfg.get(k):
                raise SystemExit(f"{saved}: '{k}' differs from {args.config}; use a new output_dir or --force")
    saved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    limit = args.limit if args.limit is not None else cfg.get("limit_per_category")
    seed = cfg.get("seed", 0)

    datasets: dict[tuple[str, str], list[Example]] = {}
    for bench, cats in cfg["benchmarks"].items():
        # a list uses limit_per_category for each; a mapping gives each category its own size
        cats = cats if isinstance(cats, dict) else {c: None for c in cats}
        for cat, cat_limit in cats.items():
            n = limit if args.limit is not None or cat_limit is None else cat_limit
            if bench == "bfcl":
                assert cat in BFCL_CATEGORIES, cat
                datasets[(bench, cat)] = subsample(load(bench, cat, cfg), n, seed)
            else:
                exs = load(bench, None, cfg)
                # keep each BFCL item's counterfactual variants together (paired gate analysis)
                datasets[(bench, "all")] = subsample_groups(exs, n, seed)

    llm_cache: dict[str, Any] = {}
    dm_cache: dict[str, Any] = {}

    def get_llm(name):
        if name is None:
            return None
        if name not in llm_cache:
            llm_cache[name] = make_llm(cfg["llms"][name])
        return llm_cache[name]

    def get_dm(name):
        if name is None:
            return None
        if name not in dm_cache:
            dm_cache[name] = make_dm(cfg["dms"][name])
        return dm_cache[name]

    judge = get_llm(cfg.get("judge"))
    # Spend guard: provider-reported cost of every row already in out_dir, plus this run's.
    budget = cfg.get("budget_usd")
    spend = {"usd": sum(r.get("llm_cost_usd", 0) + r.get("dm_cost_usd", 0) + r.get("judge_cost_usd", 0)
                        for p in out_dir.glob("*.jsonl") for r in iter_jsonl(p))}
    if budget is not None:
        print(f"spent so far in {out_dir}: ${spend['usd']:.4f} of ${budget:.2f} budget")
    # All (cell, example) pairs go through one shared pool, so small cells don't serialise the run.
    lock = threading.Lock()
    tasks: list[tuple[tuple, Pipeline, Example]] = []
    paths: dict[tuple, Path] = {}
    for cond, llm_name, dm_name in cells(cfg, set(args.only) if args.only else None):
        for (bench, cat), exs in datasets.items():
            key = (bench, cat, cond, llm_name, dm_name)
            if key in paths:
                continue
            paths[key] = out_dir / f"{bench}__{cat}__{cond}__{llm_name or '-'}__{dm_name or '-'}.jsonl"
            done = {r["id"] for r in iter_jsonl(paths[key])}
            pipe = Pipeline(cond, get_llm(llm_name), get_dm(dm_name), cfg.get("pipeline", {}))
            tasks += [(key, pipe, e) for e in exs if e.id not in done]

    stop = {"why": None}

    def work(key: tuple, pipe: Pipeline, ex: Example) -> dict[str, Any] | None:
        if stop["why"] or (budget is not None and spend["usd"] >= budget):
            return None
        bench, _, cond, llm_name, dm_name = key
        try:
            pred = pipe.run(ex)
            label, used_judge, judge_cost = (None, False, 0.0)
            if ex.benchmark == "when2call":
                label, used_judge, judge_cost = when2call_label(pred, judge, ex)
        except OutOfCredits as e:
            stop["why"] = f"out of credits: {e}"
            return None
        pd = pred.to_dict()
        with lock:
            spend["usd"] += pred.llm_cost_usd + pred.dm_cost_usd + judge_cost
        return {
            "id": ex.id, "benchmark": bench, "category": ex.category, "condition": cond,
            "llm": llm_name, "dm": dm_name, "should_call": ex.should_call, "gold_label": ex.label,
            "judge_used": used_judge, "judge_cost_usd": judge_cost, "source_id": ex.meta.get("source_id"),
            **pd, **score(ex, pd, label),
        }

    files = {k: open(p, "a") for k, p in paths.items()}
    skipped = 0
    try:
        with ThreadPoolExecutor(cfg.get("concurrency", 4)) as pool:
            futs = {pool.submit(work, *t): t[0] for t in tasks}
            bar = tqdm(as_completed(futs), total=len(futs), desc="ablation")
            for fut in bar:
                row = fut.result()
                if row is None:
                    skipped += 1
                    continue
                with lock:
                    f = files[futs[fut]]
                    f.write(json.dumps(row, default=str) + "\n")
                    f.flush()
                bar.set_postfix(spend=f"${spend['usd']:.2f}")
    finally:
        for f in files.values():
            f.close()

    for key, path in paths.items():
        rows = list(iter_jsonl(path))
        if rows:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"{path.stem}: n={len(rows)} acc={acc:.3f}")
    print(f"total spend in {out_dir}: ${spend['usd']:.4f}")
    if stop["why"]:
        raise SystemExit(f"stopped, {skipped} tasks not run: {stop['why']}; rerun to resume")
    if skipped:
        raise SystemExit(f"budget_usd={budget} reached; {skipped} tasks not run (raise the budget and rerun to resume)")


if __name__ == "__main__":
    main()
