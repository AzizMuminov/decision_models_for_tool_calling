import sys, collections
sys.argv.append("--heldout")
import selector_study as S, selector_coverage as C
agent = S.load_heldout(); S.TASK_OF.update({a["q"]: a["task"] for a in agent})
for mode in ("current: 5 best + usage", "2 services always"):
    cnt = collections.Counter(); svc = collections.Counter()
    for a in agent:
        sp = C.svc_probs(a["q"]); r = C.V(a["q"], mode)
        if sp is None: chosen = set()
        else: chosen = set(C.nucleus(*sp, kmin=2 if mode == "2 services always" else 1))
        for t in a["gold"] - set(r):
            k = "agent's tool in a service not searched" if S.SRV[t] not in chosen else "service searched, tool not in the 15"
            if sp is None: k = "answered none"
            cnt[k] += 1; svc[(S.SRV[t], k[:7])] += 1
    print(mode, sum(cnt.values()), "missed tools:", dict(cnt)); print("  top:", svc.most_common(6))
