"""Paper-style average: NDCG@10 per task, mean of tasks within each category, mean of the three categories."""
import sys, json, math, pickle, collections
from pathlib import Path
import pyarrow.parquet as pq
D = Path(sys.argv[1]); TOP = pickle.load(open(sys.argv[2], "rb"))
sys.path.insert(0, str(D.parent / "toolret_src")); from toolret.config import _TASK_2_CATEGORY as T2C
def ndcg10(r, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(r[:10])); ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(x / math.log2(i + 2) for i, x in enumerate(ideal)); return dcg / idcg if idcg else 0.0
for key in [(True, "all"), (False, "all")]:
    cat = collections.defaultdict(list)
    for qf in sorted((D / "q").glob("*.parquet")):
        Q = pq.read_table(qf).to_pylist()
        v = [ndcg10(TOP[key][q["id"]], {str(x["id"]): int(x["relevance"]) for x in json.loads(q["labels"])}) for q in Q]
        cat[T2C[qf.stem]].append(100 * sum(v) / len(v))
    cm = {c: sum(v) / len(v) for c, v in cat.items()}
    print(f"{'with' if key[0] else 'without'} instruction: web {cm['web']:.2f} code {cm['code']:.2f} customized {cm['customized']:.2f} | average {sum(cm.values())/3:.2f}")
