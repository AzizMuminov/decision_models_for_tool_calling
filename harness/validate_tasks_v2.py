"""Checks every v2 task: hidden tests fail on the original repo and pass with a reference solution."""
import json, re, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = str(HERE.parent / ".venv" / "bin" / "python")
REF = {
 "c_slug": ("textkit/slug.py", "import re, unicodedata\ndef slugify(text):\n    t = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode().lower()\n    return re.sub(r'[^a-z0-9]+','-',t).strip('-')\n"),
 "c_paginate": ("listing/paging.py", "def paginate(items, page, per_page):\n    return items[(page-1)*per_page:page*per_page]\n"),
 "c_duration": ("timeutil/duration.py", "import re\ndef parse_duration(text):\n    m = re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', text)\n    h, mi, s = (int(x or 0) for x in m.groups())\n    return h*3600+mi*60+s\n"),
 "c_dedupe": ("collections_util/dedupe.py", "def dedupe(items):\n    return list(dict.fromkeys(items))\n"),
 "c_leap": ("calendar_util/leap.py", "def is_leap(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n"),
 "c_chunk": ("batching/chunk.py", "def chunk(items, n):\n    return [items[i:i+n] for i in range(0, len(items), n)]\n"),
 "k_invoice": ("billing_client/invoice.py", "from dataclasses import dataclass\nfrom datetime import datetime, timezone\nfrom decimal import Decimal\n@dataclass\nclass Invoice:\n    id: str\n    amount: Decimal\n    currency: str\n    issued_at: datetime\ndef parse_invoice(p):\n    return Invoice(p['inv_ref'], Decimal(p['amt_minor'])/100, p['ccy'].upper(), datetime.fromtimestamp(p['issued_ts'], tz=timezone.utc))\n"),
 "m_discount": ("checkout/pricing.py", "from decimal import Decimal, ROUND_HALF_EVEN\ndef apply_discount(total, pct):\n    if pct < 0: raise ValueError\n    pct = min(pct, 30)\n    return (total - total*Decimal(pct)/100).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)\n"),
}
def hidden(d):
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_hidden.py"], cwd=d, capture_output=True, text=True, env={"TZ": "UTC", "PATH": "/usr/bin:/bin"})
    return r.returncode == 0
bad = 0
tmp = Path(sys.argv[1])
for t in sorted((HERE / "tasks_v2").iterdir()):
    meta = json.loads((t / "task.json").read_text())
    for kind in ("original", "reference"):
        d = tmp / f"{t.name}_{kind}"
        shutil.rmtree(d, ignore_errors=True); shutil.copytree(t / "repo", d)
        shutil.copy(t / "hidden" / "test_hidden.py", d / "tests" / "test_hidden.py")
        if kind == "reference":
            if t.name in REF:
                (d / REF[t.name][0]).write_text(REF[t.name][1])
            else:
                pr = meta["probe"]; f = d / pr["file"]; s = f.read_text()
                if meta["category"] == "absent":
                    s += "# TODO: value not found\n"
                else:
                    s = re.sub(rf"^{pr['name']} = .*$", f"{pr['name']} = {pr['expected']!r}", s, flags=re.M)
                f.write_text(s)
        ok = hidden(d)
        if ok != (kind == "reference"):
            bad += 1; print("PROBLEM", t.name, kind, "hidden passed" if ok else "hidden failed")
print("checked", len(list((HERE / 'tasks_v2').iterdir())), "tasks; problems:", bad)
