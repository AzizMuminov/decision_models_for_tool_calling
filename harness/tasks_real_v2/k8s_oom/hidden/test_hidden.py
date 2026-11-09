import re
from pathlib import Path
def test_limit():
    m = re.search(r"memory:\s*\"?(\d+)(Mi|Gi)", Path("deploy/report-builder.yaml").read_text())
    n = int(m.group(1)) * (1024 if m.group(2) == "Gi" else 1)
    assert n == 1024
