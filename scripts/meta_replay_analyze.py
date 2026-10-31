"""Scores results/meta/replay_rows.jsonl (see meta_replay.py) and writes results/meta/REPLAY.md."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_replay import CATALOG, EQUIV  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/meta")
rows = [json.loads(l) for l in open(out / "replay_rows.jsonl")]
ok = lambda r, k=1: r["gold"] in r["rank"][:k]  # noqa: E731
lenient = lambda r: r["pred"] == r["gold"] or r["pred"] in EQUIV.get(r["action"], set())  # noqa: E731
L = ["# Could a decision model have made this session's tool calls?", "",
     f"Replay of {len({r['msg'] for r in rows})} tool decisions from the session transcript; catalog of {len(CATALOG)} tools "
     f"the agent had; cost ${sum(r['cost_usd'] or 0 for r in rows):.3f}. Gold = the tool the agent actually called.", ""]


def table(sub, title):
    ctx = [r for r in sub if r["cond"] == "context"]
    msgs = {r["msg"]: r for r in ctx}.values()
    n = len(msgs)
    L.extend([f"### {title} (n = {n})", "", "| predictor | top-1 | top-3 | top-5 | top-1, equivalent tool ok |", "|---|---|---|---|---|"])
    L.append(f"| always Bash | {np.mean([r['gold'] == 'Bash' for r in msgs]):.3f} | – | – | "
             f"{np.mean(['Bash' == r['gold'] or 'Bash' in EQUIV.get(r['action'], set()) for r in msgs]):.3f} |")
    L.append(f"| repeat previous tool | {np.mean([r['gold'] == r['prev'] for r in msgs]):.3f} | – | – | "
             f"{np.mean([r['prev'] == r['gold'] or r['prev'] in EQUIV.get(r['action'], set()) for r in msgs]):.3f} |")
    for dm in ("msd1", "jev"):
        s = [r for r in ctx if r["dm"] == dm]
        L.append(f"| {dm}, context | {np.mean([ok(r) for r in s]):.3f} | {np.mean([ok(r, 3) for r in s]):.3f} | "
                 f"{np.mean([ok(r, 5) for r in s]):.3f} | {np.mean([lenient(r) for r in s]):.3f} |")
    L.append("")


table(rows, "All decisions")
table([r for r in rows if r["gold"] != "Bash"], "Decisions where the agent did not use Bash")

# intent condition, paired on the same decisions
L += ["### Adding the agent's stated intent (decisions where it wrote a sentence before the call)", "",
      "| DM | n | top-1 context only | top-1 with intent | top-3 with intent | equivalent tool ok, with intent |", "|---|---|---|---|---|---|"]
for dm in ("msd1", "jev"):
    it = {r["msg"]: r for r in rows if r["dm"] == dm and r["cond"] == "intent"}
    cx = {r["msg"]: r for r in rows if r["dm"] == dm and r["cond"] == "context" and r["msg"] in it}
    ids = sorted(it)
    L.append(f"| {dm} | {len(ids)} | {np.mean([ok(cx[i]) for i in ids]):.3f} | {np.mean([ok(it[i]) for i in ids]):.3f} | "
             f"{np.mean([ok(it[i], 3) for i in ids]):.3f} | {np.mean([lenient(it[i]) for i in ids]):.3f} |")
nb = [r for r in rows if r["cond"] == "intent" and r["gold"] != "Bash"]
for dm in ("msd1", "jev"):
    s = [r for r in nb if r["dm"] == dm]
    ids = {r["msg"] for r in s}
    c = [r for r in rows if r["dm"] == dm and r["cond"] == "context" and r["msg"] in ids]
    L.append(f"| {dm}, non-Bash only | {len(s)} | {np.mean([ok(r) for r in c]):.3f} | {np.mean([ok(r) for r in s]):.3f} | "
             f"{np.mean([ok(r, 3) for r in s]):.3f} | {np.mean([lenient(r) for r in s]):.3f} |")

# per gold tool recall (context, msd1 and jev)
L += ["", "### Per tool the agent used (context only): top-1 / top-5", "", "| tool | n | MSD1 | Jev |", "|---|---|---|---|"]
for t, n in Counter(r["gold"] for r in rows if r["cond"] == "context" and r["dm"] == "msd1").most_common():
    cells = []
    for dm in ("msd1", "jev"):
        s = [r for r in rows if r["cond"] == "context" and r["dm"] == dm and r["gold"] == t]
        cells.append(f"{np.mean([ok(r) for r in s]):.2f} / {np.mean([ok(r, 5) for r in s]):.2f}")
    L.append(f"| {t} | {n} | {cells[0]} | {cells[1]} |")
mist = Counter((r["gold"], r["pred"]) for r in rows if r["cond"] == "context" and r["dm"] == "msd1" and r["pred"] != r["gold"])
L += ["", "Most common MSD1 mistakes (gold → predicted): " + "; ".join(f"{g} → {p} ({k})" for (g, p), k in mist.most_common(8))]
(out / "REPLAY.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

# design B in practice: always keep the core tools, let the DM add k more from the rest of the catalog
CORE = ["Bash", "Read", "Write", "Edit"]
L += ["", "### Narrowing the catalog: core tools always shown + the DM's top-k of the rest", "",
      f"Core = {', '.join(CORE)}. Recall = the tool the agent used is among the tools shown. Catalog size {len(CATALOG)}.", "",
      "| DM | shown tools | recall, all decisions | recall, non-core decisions |", "|---|---|---|---|"]
for dm in ("msd1", "jev"):
    s = [r for r in rows if r["cond"] == "context" and r["dm"] == dm]
    for k in (0, 2, 3, 5):
        shown = lambda r: set(CORE) | set([t for t in r["rank"] if t not in CORE][:k])  # noqa: E731
        nc = [r for r in s if r["gold"] not in CORE]
        L.append(f"| {dm} | {len(CORE) + k} | {np.mean([r['gold'] in shown(r) for r in s]):.3f} | "
                 f"{np.mean([r['gold'] in shown(r) for r in nc]):.3f} (n={len(nc)}) |")
(out / "REPLAY.md").write_text("\n".join(L) + "\n")
print("\n".join(L[-12:]))
