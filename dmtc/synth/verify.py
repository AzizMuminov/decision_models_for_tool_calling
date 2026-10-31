"""Blind item-level verification of SynthHolster gold labels by a model that is NOT evaluated.

String checks cannot catch an item whose meaning drifted from its spec (e.g. the text ends up asking
for a sibling tool). The reviewer sees the conversation and the item's tool family (plus "none of
these tools fits", "need more information from the customer/user", "answer directly, no tool") without
the gold, and must pick one. An item is kept only if the pick equals the gold.

Trade-off, reported with the results: this keeps items an LLM can resolve, which may favour LLM-style
reading; it removes wrong or ambiguous golds, which matters more.

    python -m dmtc.synth.verify --data data/synth
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dmtc.synth.build import CANARY

REVIEWER = "openai/gpt-6.1-sol"
PROMPT = """You are checking a test item for a tool-using assistant. Read the conversation and decide the single best
next action of the assistant, using ONLY the options below.

Conversation:
{conv}

Options:
{options}

Answer with the option letter only."""

SPECIAL = {
    "cannot_answer": "None of the listed tools fits what is asked; the assistant cannot do it with these tools.",
    "request_for_info": "A tool fits, but a required detail (such as an id) is missing, so the assistant must ask for it first.",
    "direct": "No tool is needed; the assistant should just answer from general knowledge.",
}


def main() -> None:
    import openai

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    a = ap.parse_args()
    d = Path(a.data)
    cat = json.loads((d / "catalog.json").read_text())
    V = {v["name"]: v for dom in cat["domains"] for f in dom["families"] for v in f["variants"]}
    items = [json.loads(l) for l in open(d / "items.jsonl")][1:]
    cli = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    def check(it):
        fam = [t for t in it["family_tools"] if t != it.get("removed_tool")]
        opts = [(f"tool:{t}", f"Call {t}: {V[t]['description']}") for t in fam]
        opts += [(k, v) for k, v in SPECIAL.items() if not (k == "request_for_info" and it["domain"] == "CodingAgent")]
        random.Random(it["id"]).shuffle(opts)
        letters = "ABCDEFGHIJ"
        conv = "\n".join(f"{m['role']}: {m['content']}" for m in it["messages"])
        listing = "\n".join(f"{letters[i]}. {txt}" for i, (_, txt) in enumerate(opts))
        gold = f"tool:{it['gold_tool']}" if it["label"] == "tool_call" else it["label"]
        for i in range(5):
            try:
                r = cli.chat.completions.create(model=REVIEWER, max_tokens=600, extra_body={"reasoning": {"effort": "low"}},
                                                messages=[{"role": "user", "content": PROMPT.format(conv=conv, options=listing)}])
                m = re.search(r"\b([A-J])\b", (r.choices[0].message.content or "").strip())
                pick = opts[letters.index(m.group(1))][0] if m and letters.index(m.group(1)) < len(opts) else None
                return pick, gold, float(getattr(r.usage, "cost", 0) or 0)
            except Exception:  # noqa: BLE001
                time.sleep(3 * (i + 1))
        return None, gold, 0.0

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(check, items))
    kept, dropped = [], Counter()
    for it, (pick, gold, _) in zip(items, res):
        it["verify"] = {"reviewer": REVIEWER, "pick": pick}
        if pick == gold:
            kept.append(it)
        else:
            dropped[(it["domain"] == "CodingAgent" and "coding" or "business", it["slice"])] += 1
    for k, it in enumerate(kept):
        it["id"] = f"sh{k:04d}"
    (d / "items_unverified.jsonl").write_text(json.dumps({"canary": CANARY}) + "\n" + "\n".join(json.dumps(i) for i in items) + "\n")
    (d / "items.jsonl").write_text(json.dumps({"canary": CANARY}) + "\n" + "\n".join(json.dumps(i) for i in kept) + "\n")
    print(f"verified: kept {len(kept)}/{len(items)}; cost ${sum(c for *_, c in res):.4f}")
    print("dropped by (domain, slice):", dict(dropped))
    print("kept by (domain, slice):", dict(Counter((i["domain"] == "CodingAgent" and "coding" or "business", i["slice"]) for i in kept)))


if __name__ == "__main__":
    main()
