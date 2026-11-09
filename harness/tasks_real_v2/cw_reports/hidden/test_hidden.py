from reports.export import export_rows
def test_new_field_no_crash():
    # upstream added `currency`; keeping it as a column or ignoring it are both valid fixes
    lines = export_rows([{"id": 1, "amount": 5, "currency": "EUR"}]).splitlines()
    assert lines[0].split(",")[:2] == ["id", "amount"]
    assert lines[1].split(",")[:2] == ["1", "5"]
def test_old_rows_still_work():
    lines = export_rows([{"id": 2, "amount": 7}]).splitlines()
    assert lines[1].split(",")[:2] == ["2", "7"]
