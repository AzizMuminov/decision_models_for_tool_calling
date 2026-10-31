"""Classification metrics for the session replay (results/meta/replay_rows.jsonl) -> results/meta/REPLAY_METRICS.md.

Labels = tools. Macro averages are over the tools the agent actually used (rows of the confusion matrix),
so a predictor is not rewarded for tools that never occur. Weighted averages weight each tool by how often
the agent used it.
"""
import json
import sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/meta")
rows = [json.loads(l) for l in open(out / "replay_rows.jsonl")]


def metrics(pairs):
    gold = [g for g, _ in pairs]
    labels = sorted(set(gold))
    sup = Counter(gold)
    tp = Counter(g for g, p in pairs if g == p)
    pp = Counter(p for _, p in pairs)
    P = {l: tp[l] / pp[l] if pp[l] else 0.0 for l in labels}
    R = {l: tp[l] / sup[l] for l in labels}
    F = {l: 2 * P[l] * R[l] / (P[l] + R[l]) if P[l] + R[l] else 0.0 for l in labels}
    n = len(pairs)
    return {"n": n, "acc": sum(tp.values()) / n,
            "macro_p": sum(P.values()) / len(labels), "macro_r": sum(R.values()) / len(labels),
            "macro_f1": sum(F.values()) / len(labels),
            "weighted_f1": sum(F[l] * sup[l] for l in labels) / n,
            "per_tool": {l: (sup[l], P[l], R[l], F[l]) for l in labels}}


def table(title, systems):
    L = [f"### {title}", "", "| predictor | n | accuracy | macro precision | macro recall (= balanced acc.) | macro F1 | weighted F1 |",
         "|---|---|---|---|---|---|---|"]
    for name, pairs in systems:
        m = metrics(pairs)
        L.append(f"| {name} | {m['n']} | {m['acc']:.3f} | {m['macro_p']:.3f} | {m['macro_r']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} |")
    return L + [""]


ctx = {dm: {r["msg"]: r for r in rows if r["dm"] == dm and r["cond"] == "context"} for dm in ("msd1", "jev")}
ids = sorted(ctx["msd1"])
base = ctx["msd1"]
sys_all = [("always Bash", [(base[i]["gold"], "Bash") for i in ids]),
           ("repeat previous tool", [(base[i]["gold"], base[i]["prev"] or "Bash") for i in ids]),
           ("MSD1", [(ctx["msd1"][i]["gold"], ctx["msd1"][i]["pred"]) for i in ids]),
           ("Jev", [(ctx["jev"][i]["gold"], ctx["jev"][i]["pred"]) for i in ids])]
L = ["# Session replay: classification metrics", "",
     "Each decision is one example; the label is the tool the agent called; 38 possible tools, 13 actually used.", ""]
L += table("All decisions, context only", sys_all)
nb = [i for i in ids if base[i]["gold"] != "Bash"]
L += table("Decisions where the agent did not use Bash", [(n, [p for p, i in zip(pairs, ids) if i in set(nb)]) for n, pairs in sys_all])
it = {dm: {r["msg"]: r for r in rows if r["dm"] == dm and r["cond"] == "intent"} for dm in ("msd1", "jev")}
iids = sorted(it["msd1"])
L += table("Decisions with a stated intent: context only vs with intent", [
    ("always Bash", [(base[i]["gold"], "Bash") for i in iids]),
    ("MSD1, context only", [(ctx["msd1"][i]["gold"], ctx["msd1"][i]["pred"]) for i in iids]),
    ("MSD1, with intent", [(it["msd1"][i]["gold"], it["msd1"][i]["pred"]) for i in iids]),
    ("Jev, context only", [(ctx["jev"][i]["gold"], ctx["jev"][i]["pred"]) for i in iids]),
    ("Jev, with intent", [(it["jev"][i]["gold"], it["jev"][i]["pred"]) for i in iids])])
L += ["### Per tool (all decisions, context only): precision / recall / F1", "", "| tool | n | MSD1 | Jev |", "|---|---|---|---|"]
m1, m2 = metrics(sys_all[2][1]), metrics(sys_all[3][1])
for t, (n, *_) in sorted(m1["per_tool"].items(), key=lambda x: -x[1][0]):
    a, b = m1["per_tool"][t], m2["per_tool"][t]
    L.append(f"| {t} | {n} | {a[1]:.2f} / {a[2]:.2f} / {a[3]:.2f} | {b[1]:.2f} / {b[2]:.2f} / {b[3]:.2f} |")
(out / "REPLAY_METRICS.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
