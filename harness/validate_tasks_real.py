"""Checks every task in harness/tasks_real before any agent sees it.

  1. hidden tests fail on the untouched repo (except absent tasks, where unchanged is correct)
  2. hidden tests pass once solution/ is applied
  3. every tool named in needs/gold/writes exists in the real catalog
  4. every gold call has the tool's required arguments and gets a non-error answer from the simulator
     (absent tasks: the gold call must come back empty or not-found)
  5. the gold calls satisfy the task's required writes
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_sim import answer, load_catalog  # noqa: E402

PY = HERE.parent / ".venv" / "bin" / "python"
EXTRA = "--extra" in sys.argv
TASK_DIR = HERE / (sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "tasks_real")
CAT = {t["name"]: t for t in load_catalog(extra=EXTRA)}


def run_hidden(task_dir: Path, with_solution: bool) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp) / "w"
        shutil.copytree(task_dir / "repo", w)
        if with_solution and (task_dir / "solution").exists():
            shutil.copytree(task_dir / "solution", w, dirs_exist_ok=True)
        (w / "tests").mkdir(exist_ok=True)
        shutil.copy(task_dir / "hidden" / "test_hidden.py", w / "tests" / "test_hidden.py")
        r = subprocess.run([str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_hidden.py"], cwd=w, capture_output=True, text=True)
        return r.returncode == 0


def write_ok(spec: dict, calls: list[tuple[str, dict]]) -> bool:
    from real_sim import _matches

    tools = spec["tool"] if isinstance(spec["tool"], list) else [spec["tool"]]
    return any(n in tools and _matches(spec, a) for n, a in calls)


def main() -> int:
    bad = 0
    for td in sorted(TASK_DIR.iterdir()):
        m = json.loads((td / "task.json").read_text())
        world = json.loads((td / "world.json").read_text())
        errs = []
        before, after = run_hidden(td, False), run_hidden(td, True)
        if m["category"] != "absent" and before:
            errs.append("hidden tests already pass on the untouched repo")
        if not after and (td / "solution").exists():
            errs.append("hidden tests fail with the solution")
        names = {n for g in m["needs"] for n in g} | {g[0] for g in m["gold"]} | {n for w in m["writes"] for n in (w["tool"] if isinstance(w["tool"], list) else [w["tool"]])}
        errs += [f"unknown tool {n}" for n in sorted(names) if n not in CAT]
        for name, args in m["gold"]:
            if name not in CAT:
                continue
            missing = [k for k in CAT[name]["inputSchema"].get("required") or [] if k not in args]
            if missing:
                errs.append(f"gold {name} misses required {missing}")
            resp, ok = answer(name, args, world)
            empty = isinstance(resp, dict) and resp.get("items") == [] or (isinstance(resp, str) and "No unresolved" in resp) \
                or (isinstance(resp, dict) and "error" in resp) or (isinstance(resp, dict) and "no decision" in json.dumps(resp))
            if m["category"] != "absent" and (not ok or (isinstance(resp, dict) and resp.get("items") == [])):
                errs.append(f"gold {name} gets no answer: {str(resp)[:100]}")
        for w in m["writes"]:
            if not write_ok(w, m["gold"]):
                errs.append(f"gold calls do not satisfy write {w}")
        print(f"{td.name:18s} {m['category']:8s} {'OK' if not errs else 'FAIL: ' + '; '.join(errs)}")
        bad += bool(errs)
    print("failures:", bad)
    return bad


if __name__ == "__main__":
    sys.exit(main())
