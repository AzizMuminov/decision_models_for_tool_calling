from pathlib import Path
def pins():
    return dict(l.split("==") for l in Path("requirements.txt").read_text().split() if "==" in l)
def v(s):
    return tuple(int(x) for x in s.split("."))
def test_patched():
    p = pins()
    assert v(p["urllib3"]) >= (2, 5, 0)
    assert v(p["jinja2"]) >= (3, 1, 6)
    assert p["requests"] == "2.32.3" and p["click"] == "8.1.7"
