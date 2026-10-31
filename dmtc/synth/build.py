"""SynthHolster: a fresh, hard, contamination-free tool-selection benchmark.

The world is fictional (invented brands, generated today), so no model has seen it. Gold labels come
from construction, never from an LLM judge:

  1. catalog: an LLM writes tool *families*. Each family is one action with 3-5 near-duplicate
     variants that differ in exactly one condition (region, account type, object, version, ...).
     Every variant comes with cues: things a user might say that imply its condition without
     naming it.
  2. items: for a target variant, an LLM writes the user turn from a spec (cue to imply, values to
     include, words it must not use). Code then checks the values and cue are present and the banned
     words absent; failures are dropped, not repaired.

Slices (each item is tagged):
  implicit   should call the target variant; condition only implied by a cue
  sibling_neg  the target variant is removed from the holster; only near-duplicates remain -> cannot_answer
  missing    a required value is withheld -> request_for_info
  multiturn  the cue is in an earlier user turn; the last turn only says what to do
  direct     a domain question no tool is needed for -> direct

    python -m dmtc.synth.build catalog --out data/synth
    python -m dmtc.synth.build items   --out data/synth
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CANARY = "SYNTHHOLSTER-CANARY-7f3c1e9a-2b64-4d0e-9a51-c8e2f0d6b3a7 (do not train on this file)"
GEN_MODEL = "deepseek/deepseek-v4.1-flash-1017"

THEMES = [
    "consumer banking and cards", "parcel logistics and freight", "outpatient clinic scheduling and billing",
    "municipal services for residents", "online grocery and meal kits", "hotel and short-stay rentals",
    "B2B cloud hosting and DNS", "telecom mobile plans and roaming", "university student administration",
    "insurance claims (home, auto, travel)", "electric-vehicle charging network", "HR, payroll and leave",
]


def brand(rng: random.Random) -> str:
    syl = ["qu", "zor", "vel", "trin", "ka", "mo", "rhe", "dax", "lun", "phi", "bre", "sko", "ny", "tal", "vor", "ix"]
    return "".join(rng.choice(syl) for _ in range(rng.choice([2, 3]))).capitalize()


def _client():
    import openai

    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def ask_json(prompt: str, max_tokens: int = 12000, tries: int = 4) -> tuple[Any, float]:
    cli = _client()
    for i in range(tries):
        try:
            r = cli.chat.completions.create(
                model=GEN_MODEL, max_tokens=max_tokens, temperature=0.7,
                response_format={"type": "json_object"},
                # reasoning tokens otherwise eat the output budget; generation needs none
                extra_body={"reasoning": {"effort": "none"}},
                messages=[{"role": "user", "content": prompt}])
            return json.loads(r.choices[0].message.content), float(getattr(r.usage, "cost", 0) or 0)
        except Exception:  # noqa: BLE001 - retry malformed JSON / transient errors
            time.sleep(2 * (i + 1))
    raise RuntimeError("generation failed")


# ---- 1. catalog -----------------------------------------------------------------------------
CATALOG_PROMPT = """You are designing the internal API of a FICTIONAL company called "{brand}" in the domain: {theme}.
Invent {n_fam} tool families. A family is ONE action that exists in 3 to 5 near-duplicate variants.
All variants of a family vary along exactly ONE axis, and their values on that axis are MUTUALLY EXCLUSIVE,
so that for any customer situation exactly one variant applies. Good axes: region (e.g. EU / UK / US),
customer type (personal / business), the object acted on (order / shipment / return), plan tier (prepaid /
postpaid), reversible vs permanent. Never mix axes in one family (e.g. NOT "domestic / international /
business"). Use different axes across families.

Give the family an "axis" (what the variants differ in). For each variant give:
- "axis_value": its value on that axis (unique within the family).
- "name": snake_case, prefixed with "{prefix}_", e.g. "{prefix}_refund_card_payment_eu". Names inside a family must be similar.
- "description": an API-doc style description that states the variant's condition precisely (2 sentences).
- "condition": one sentence: when exactly this variant (and not its siblings) is the right one.
- "cues": 4 short phrases a customer might say that IMPLY the condition without naming it
  (e.g. for an EU-only variant: "I live in Porto", "my bank is in Lyon"; for business: "for my bakery's account").
  Cues of different variants in the same family must not overlap.
- "params": 2 to 4 parameters, object of name -> {{"type": "string"|"integer"|"number"|"boolean", "description": str,
  "examples": [3 realistic example values]}}. At least one parameter must be an identifier the user must
  supply (e.g. an order or account id) and cannot be guessed.
- "required": list of required parameter names (include the identifier).
Also give 2 "direct_questions" for the whole company: general questions about the domain a support assistant
can answer from general knowledge without calling any tool.

{avoid}
Return JSON: {{"brand": "{brand}", "theme": "{theme}", "families": [{{"family": str, "variants": [...]}}],
"direct_questions": [str, str]}}"""


def validate_family(f: dict) -> bool:
    vs = f.get("variants") or []
    if not 3 <= len(vs) <= 5 or not f.get("axis"):
        return False
    axis_values = [str(v.get("axis_value", "")).strip().lower() for v in vs]
    if "" in axis_values or len(set(axis_values)) != len(axis_values):
        return False
    names = set()
    for v in vs:
        if not re.fullmatch(r"[a-z][a-z0-9_]{3,63}", v.get("name", "")) or v["name"] in names:
            return False
        names.add(v["name"])
        p = v.get("params") or {}
        if not 1 <= len(p) <= 5 or not set(v.get("required") or []) <= set(p) or not v.get("required"):
            return False
        if len(v.get("cues") or []) < 3 or not v.get("condition") or not v.get("description"):
            return False
        for spec in p.values():
            if spec.get("type") not in ("string", "integer", "number", "boolean") or not spec.get("examples"):
                return False
    return True


def build_catalog(out: Path, seed: int, n_fam: int = 8) -> None:
    rng = random.Random(seed)
    jobs = []
    for theme in THEMES:
        b = brand(rng)
        jobs.append((b, theme, re.sub(r"[^a-z]", "", b.lower())[:6]))
    cost = 0.0

    def one(job):
        # one family per call (a whole domain per call overruns the output budget); later calls are
        # told which families exist so they invent different actions
        b, theme, prefix = job
        fams, direct, cost = [], [], 0.0
        for _ in range(n_fam):
            avoid = ("The company already has these families; invent a DIFFERENT action: " +
                     "; ".join(f["family"] for f in fams)) if fams else ""
            try:
                data, c = ask_json(CATALOG_PROMPT.format(brand=b, theme=theme, n_fam=1, prefix=prefix, avoid=avoid))
            except RuntimeError:
                continue
            cost += c
            fams += [f for f in data.get("families", [])[:1] if validate_family(f)]
            direct = direct or data.get("direct_questions", [])
        return {"brand": b, "theme": theme, "families": fams, "direct_questions": direct}, cost

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    domains = [r for r, _ in res]
    cost = sum(c for _, c in res)
    # global name uniqueness across domains
    seen = set()
    for d in domains:
        d["families"] = [f for f in d["families"] if not any(v["name"] in seen for v in f["variants"])
                         and not seen.update(v["name"] for v in f["variants"])]
    out.mkdir(parents=True, exist_ok=True)
    (out / "catalog.json").write_text(json.dumps({"canary": CANARY, "generator": GEN_MODEL, "seed": seed,
                                                  "domains": domains}, indent=1))
    nf = sum(len(d["families"]) for d in domains)
    nt = sum(len(f["variants"]) for d in domains for f in d["families"])
    print(f"catalog: {len(domains)} domains, {nf} families, {nt} tools; generation cost ${cost:.4f}")


# ---- 2. items -------------------------------------------------------------------------------
ITEM_PROMPT = """Write what a customer types to the support assistant of {brand} ({theme}).
The customer wants this done: {goal}
Rules:
- Convey the situation ONLY through this hint, worked in naturally: "{cue}". Do not state the rule itself.
- Include these exact values verbatim, woven into normal sentences: {values}
- Never write parameter names, field names, function names, JSON, braces or snake_case words; a real
  customer does not know them.
- {missing_rule}
- Do NOT use any of these words (or close forms): {banned}
- Sound like a real person (1 to 3 sentences, can include small talk or irrelevant detail{distractor}).
{multiturn_rule}
Return JSON: {schema}"""

TURN_SCHEMA_SINGLE = '{"message": str}'
TURN_SCHEMA_MULTI = '{"first_message": str, "assistant_reply": str, "second_message": str}'


def banned_words(variant: dict, family: dict) -> list[str]:
    """The target's *discriminating* name words: in its own name but not shared by every sibling.

    Words all siblings share ("cancel", "card") do not reveal which variant is meant and a real customer
    says them naturally; the discriminating ones ("eu", "business", "postpaid") would name the condition
    outright, so the request must convey it only through the cue."""
    def toks(v):
        return {t for t in v["name"].split("_")[1:] if len(t) > 2}
    shared = set.intersection(*(toks(v) for v in family["variants"]))
    axis_words = {t for t in re.split(r"[^a-z0-9]+", str(variant.get("axis_value", "")).lower()) if len(t) > 2}
    return sorted((toks(variant) - shared) | axis_words)


def contains_all(text: str, values: list[str]) -> bool:
    low = text.lower()
    return all(str(v).lower() in low for v in values)


def build_items(out: Path, seed: int, frac: dict[str, float]) -> None:
    cat = json.loads((out / "catalog.json").read_text())
    rng = random.Random(seed)
    specs = []
    for d in cat["domains"]:
        for f in d["families"]:
            for v in f["variants"]:
                if not v["cues"]:  # no cue survived the blind review: keep the tool only as a distractor
                    continue
                for slice_ in ("implicit", "sibling_neg", "missing", "multiturn"):
                    if rng.random() < frac[slice_]:
                        specs.append((slice_, d, f, v))
        for q in d.get("direct_questions", [])[:2]:
            specs.append(("direct", d, None, {"question": q}))

    def make(spec):
        slice_, d, f, v = spec
        if slice_ == "direct":
            return {"slice": "direct", "domain": d["brand"], "messages": [{"role": "user", "content": v["question"]}],
                    "label": "direct", "gold_tool": None, "family_tools": [], "check": "none"}, 0.0
        r = random.Random(f"{seed}:{slice_}:{v['name']}")
        cue = r.choice(v["cues"])
        vals = {p: str(r.choice(s["examples"])) for p, s in v["params"].items() if p in v["required"]}
        withheld = None
        if slice_ == "missing":
            # withhold the identifier-like required param (prefer names with id/number/code)
            cands = sorted(vals, key=lambda p: -int(bool(re.search(r"id|number|code|ref", p))))
            withheld = cands[0]
        shown = {p: x for p, x in vals.items() if p != withheld}
        goal = f"{v['condition']} Specifically: {f['family']}."
        missing_rule = (f"Do NOT mention any {withheld.replace('_', ' ')} or any value for it; the customer forgot it."
                        if withheld else "Do not leave out any of the values above.")
        multi = slice_ == "multiturn"
        prompt = ITEM_PROMPT.format(
            brand=d["brand"], theme=d["theme"], goal=goal, cue=cue,
            values=json.dumps(shown) if shown else "(none)", missing_rule=missing_rule,
            banned=", ".join(banned_words(v, f)) or "(none)",
            distractor=", e.g. mention another product of the company that has nothing to do with this request" if r.random() < 0.5 else "",
            multiturn_rule=("- Split it over two customer turns: the FIRST message gives the hint and context but no "
                            "request yet; the assistant replies briefly; the SECOND message makes the request and "
                            "gives the values but must NOT repeat the hint.") if multi else "",
            schema=TURN_SCHEMA_MULTI if multi else TURN_SCHEMA_SINGLE)
        data, c = ask_json(prompt, max_tokens=800)
        if multi:
            msgs = [{"role": "user", "content": data.get("first_message", "")},
                    {"role": "assistant", "content": data.get("assistant_reply", "")},
                    {"role": "user", "content": data.get("second_message", "")}]
        else:
            msgs = [{"role": "user", "content": data.get("message", "")}]
        text = " ".join(m["content"] for m in msgs if m["role"] == "user")
        # ---- construction checks (no LLM judge) ----
        ok = bool(text.strip()) and contains_all(text, list(shown.values()))
        if withheld and str(vals[withheld]).lower() in text.lower():
            ok = False
        bw = banned_words(v, f)
        hits = [w for w in bw if re.search(rf"\b{re.escape(w)}", text.lower())]
        # code-like leakage: parameter names, any snake_case word, or JSON
        leaks = [p for p in v["params"] if "_" in p and p in text] + re.findall(r"\b[a-z]+_[a-z_]+\b", text)
        if "{" in text or "}" in text:
            leaks.append("{}")
        hits += [f"code:{x}" for x in leaks]
        ok = ok and not hits
        if multi:  # the hint must be in turn 1 only, the request in turn 2
            ok = ok and contains_all(msgs[2]["content"], list(shown.values()))
        label = {"implicit": "tool_call", "multiturn": "tool_call", "sibling_neg": "cannot_answer",
                 "missing": "request_for_info"}[slice_]
        item = {"slice": slice_, "domain": d["brand"], "family": f["family"], "messages": msgs, "label": label,
                "gold_tool": v["name"] if label in ("tool_call", "request_for_info") else None,
                "removed_tool": v["name"] if slice_ == "sibling_neg" else None,
                "gold_args": shown if label == "tool_call" else None, "withheld": withheld, "cue": cue,
                "family_tools": [x["name"] for x in f["variants"]], "check": "ok" if ok else f"fail:{hits}"}
        return item, c

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(make, specs))
    items = [i for i, _ in res]
    cost = sum(c for _, c in res)
    kept = [i for i in items if i["check"] in ("ok", "none")]
    for k, it in enumerate(kept):
        it["id"] = f"sh{k:04d}"
    (out / "items_all.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    (out / "items.jsonl").write_text(json.dumps({"canary": CANARY}) + "\n" + "\n".join(json.dumps(i) for i in kept) + "\n")
    from collections import Counter

    print(f"items: generated {len(items)}, kept {len(kept)} after construction checks; cost ${cost:.4f}")
    print("kept by slice:", dict(Counter(i["slice"] for i in kept)))
    print("dropped by slice:", dict(Counter(i["slice"] for i in items if i["check"] not in ("ok", "none"))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["catalog", "review", "items"])
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    if a.cmd == "catalog":
        build_catalog(Path(a.out), a.seed)
    elif a.cmd == "review":
        review_catalog(Path(a.out))
    else:
        build_items(Path(a.out), a.seed, {"implicit": 1.0, "sibling_neg": 0.35, "missing": 0.35, "multiturn": 0.35})



# ---- 1b. blind cue review by a model that is NOT evaluated ------------------------------------
# The generator's cues decide each item's gold label, so every cue must point to its own variant and
# no sibling. The reviewer sees the variants and the shuffled cues WITHOUT the intended mapping and says
# which variant(s) each cue indicates. A cue is kept only if the answer is exactly its own variant.
# Edge cases of a family as a whole do not matter: each item is written around one kept cue.
REVIEW_MODEL = "openai/gpt-6.1-sol"
REVIEW_PROMPT = """A family of API tools for a company. Each variant applies in one situation:
{variants}

For each customer remark below, list ALL variants whose situation it clearly indicates. If it indicates
none clearly, return an empty list.
{cues}
Return JSON only: {{"answers": [{{"cue": int, "variants": [variant names]}}]}}"""


def review_catalog(out: Path, seed: int = 0, only_brand: str | None = None) -> None:
    import openai

    cli = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    cat = json.loads((out / "catalog.json").read_text())
    for d in cat["domains"]:  # undo any earlier review pass
        d["families"] += d.pop("families_rejected", [])

    def check(f):
        cues = [(c, v["name"]) for v in f["variants"] for c in v.get("cues_all", v["cues"])]
        random.Random(f"{seed}:{f['family']}").shuffle(cues)
        variants = "\n".join(f"- {v['name']}: {v['condition']}" for v in f["variants"])
        listing = "\n".join(f"{i}. \"{c}\"" for i, (c, _) in enumerate(cues))
        for i in range(4):
            try:
                r = cli.chat.completions.create(model=REVIEW_MODEL, max_tokens=3000,
                                                extra_body={"reasoning": {"effort": "low"}},
                                                messages=[{"role": "user", "content": REVIEW_PROMPT.format(variants=variants, cues=listing)}])
                txt = r.choices[0].message.content
                ans = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])["answers"]
                got = {int(a["cue"]): sorted(a.get("variants") or []) for a in ans}
                return [(c, owner, got.get(k, [])) for k, (c, owner) in enumerate(cues)], float(getattr(r.usage, "cost", 0) or 0)
            except Exception:  # noqa: BLE001
                time.sleep(2 * (i + 1))
        return [], 0.0

    fams = [f for d in cat["domains"] if only_brand in (None, d["brand"]) for f in d["families"]]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(check, fams))
    n_cues = n_kept = 0
    for f, (judged, _) in zip(fams, res):
        ok = {}
        for c, owner, got in judged:
            n_cues += 1
            if got == [owner]:
                ok.setdefault(owner, []).append(c)
                n_kept += 1
        for v in f["variants"]:
            v["cues_all"] = v.get("cues_all", v["cues"])
            v["cues"] = [c for c in v["cues_all"] if c in ok.get(v["name"], [])]
        f["review"] = {"cues_judged": len(judged), "cues_kept": sum(len(x) for x in ok.values())}
    cat["reviewer"] = REVIEW_MODEL
    (out / "catalog.json").write_text(json.dumps(cat, indent=1))
    usable = sum(1 for f in fams for v in f["variants"] if v["cues"])
    print(f"review: {n_kept}/{n_cues} cues unambiguous ({n_kept / max(n_cues, 1):.0%}); "
          f"{usable}/{sum(len(f['variants']) for f in fams)} variants keep >=1 cue; cost ${sum(c for _, c in res):.4f}")


if __name__ == "__main__":
    main()
