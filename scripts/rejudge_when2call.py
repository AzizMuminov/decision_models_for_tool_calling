"""Re-labels Study 1's judged When2Call text replies with a judge that is never evaluated (GPT-6.1 Sol).

Study 1 used DeepSeek, which is also a system under test, to map text replies onto
direct / request_for_info / cannot_answer. This re-runs the same prompt with the neutral judge so the
two label sets can be compared (agreement, Cohen's kappa, and the per-label table under each judge).

    source ../.or_env && .venv/bin/python scripts/rejudge_when2call.py results/openrouter_v1 results/review
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmtc.data import load_when2call  # noqa: E402
from dmtc.pipeline import WHEN2CALL_LABELS  # noqa: E402

MODEL = "openai/gpt-6.1-sol"


def prompt(user: str, reply: str) -> str:
    # verbatim copy of the Study 1 prompt in dmtc/pipeline.py:when2call_label
    return (
        "Classify the assistant reply to the user's request into exactly one category.\n"
        f"Categories: {', '.join(l for l in WHEN2CALL_LABELS if l != 'tool_call')}\n"
        "- direct: answers the question directly\n"
        "- request_for_info: asks the user for missing information\n"
        "- cannot_answer: says it cannot help or lacks the capability\n\n"
        f"User request: {user}\nAssistant reply: {reply}\n\nReply with the category name only."
    )


def main() -> None:
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(res / "config.yaml"))
    users = {ex.id: ex.user_text for ex in load_when2call(Path(cfg["when2call_path"]))}
    todo: dict[str, tuple[str, str]] = {}
    for p in sorted(res.glob("when2call__*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if r["judge_used"]:
                k = hashlib.sha1(f"{r['id']}\n{r['text']}".encode()).hexdigest()
                todo[k] = (users[r["id"]], r["text"])
    cache_p = out / "when2call_rejudge_cache.jsonl"
    cache = {json.loads(l)["key"]: json.loads(l) for l in open(cache_p)} if cache_p.exists() else {}
    keys = [k for k in todo if k not in cache]
    print(f"{len(todo)} unique replies, {len(keys)} to judge", flush=True)
    cli = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    def one(k: str) -> dict:
        user, reply = todo[k]
        for attempt in range(8):
            try:
                r = cli.chat.completions.create(model=MODEL, max_tokens=400, extra_body={"reasoning": {"effort": "low"}},
                                                messages=[{"role": "user", "content": prompt(user, reply)}])
                txt = (r.choices[0].message.content or "").lower()
                lab = next((x for x in ("request_for_info", "cannot_answer", "direct") if x in txt), "direct")
                return {"key": k, "label": lab, "raw": txt[:80], "cost_usd": float(getattr(r.usage, "cost", 0) or 0)}
            except Exception as e:  # noqa: BLE001
                if getattr(e, "status_code", None) == 403 or attempt == 7:
                    raise
                time.sleep(min(10 * (attempt + 1), 60))
        raise RuntimeError("unreachable")

    spent = 0.0
    with open(cache_p, "a") as f, ThreadPoolExecutor(16) as pool:
        for i, row in enumerate(pool.map(one, keys), 1):
            f.write(json.dumps(row) + "\n")
            spent += row["cost_usd"]
            if i % 200 == 0:
                f.flush()
                print(f"{i}/{len(keys)} ${spent:.3f}", flush=True)
    print(f"done, ${spent:.3f}")


if __name__ == "__main__":
    main()
