import csv, io

def export_rows(rows: list[dict]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["id", "amount"], extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return out.getvalue()
