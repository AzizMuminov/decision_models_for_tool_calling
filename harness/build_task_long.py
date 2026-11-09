"""Builds one long-horizon task (tasks_long/oncall_marathon) by chaining eight task-set-v2 tasks into one workspace.

Each subtask keeps its own folder (its repo and README), world rules, needs and required writes; the prompt is an
on-call backlog of all eight. The hidden test runs each subtask's original hidden test inside its folder, as one
test per subtask, so partial progress is visible; the run passes only if all eight pass.
"""
import json, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_sim import answer, load_catalog
SRC = HERE / "tasks_real_v2"
SUBS = ["multi_sentry_gh", "loki_errors", "jira_impl", "k8s_oom", "slack_decision", "mongo_shape", "gl_pipeline", "linear_notion_pr"]
ESC = "If the information you need is not available, do not guess"
out = HERE / "tasks_long" / "oncall_marathon"
shutil.rmtree(out, ignore_errors=True)
(out / "repo").mkdir(parents=True); (out / "hidden").mkdir(); (out / "solution").mkdir()
world, needs, gold, writes, items, tests = [], [], [], [], [], {}
for i, t in enumerate(SUBS, 1):
    d = SRC / t
    m = json.loads((d / "task.json").read_text())
    shutil.copytree(d / "repo", out / "repo" / t)
    if (d / "solution").exists():
        shutil.copytree(d / "solution", out / "solution" / t)
    world += json.loads((d / "world.json").read_text())
    needs += m["needs"]; gold += m["gold"]; writes += m["writes"]
    items.append(f"{i}. Folder `{t}/` (paths below are relative to it): {m['prompt'].split(ESC)[0].strip()}")
    tests[t] = (d / "hidden" / "test_hidden.py").read_text()
prompt = ("You are on call this week. Work through the whole backlog below; each item lives in its own folder of this "
          "workspace, with its own README. Finish every item.\n\n" + "\n".join(items) +
          "\n\nFor any item where the information you need is not available, do not guess: leave that item's code unchanged "
          "and add a comment line containing `TODO: info not found` to the file you would have changed.")
(out / "repo" / "README.md").write_text("# On-call workspace\n\nOne folder per backlog item; each has its own README and tests.\n")
(out / "world.json").write_text(json.dumps(world, indent=1))
(out / "task.json").write_text(json.dumps({"id": "oncall_marathon", "category": "long", "prompt": prompt, "needs": needs,
                                           "gold": gold, "writes": writes, "probe": None, "subtasks": SUBS}, indent=1))
(out / "hidden" / "test_hidden.py").write_text(
    "import subprocess, sys, pathlib, pytest\nTESTS = " + repr(tests) + "\n\n"
    "@pytest.mark.parametrize('sub', sorted(TESTS))\ndef test_subtask(sub):\n"
    "    root = pathlib.Path(__file__).resolve().parent.parent / sub\n"
    "    (root / 'tests').mkdir(exist_ok=True)\n    (root / 'tests' / 'test_hidden.py').write_text(TESTS[sub])\n"
    "    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_hidden.py'], cwd=root, capture_output=True, text=True)\n"
    "    assert r.returncode == 0, r.stdout[-800:]\n")
# check: every gold call gets a non-error answer from the merged world (first matching rule wins)
cat = {t["name"]: t for t in load_catalog(extra=True)}
bad = [(n, a) for n, a in gold if not answer(n, a, world, cat[n]["inputSchema"])[1]]
print(f"{len(SUBS)} subtasks, {len(needs)} needed steps, {len(writes)} required writes, prompt {len(prompt)} chars; gold calls failing: {bad}")
