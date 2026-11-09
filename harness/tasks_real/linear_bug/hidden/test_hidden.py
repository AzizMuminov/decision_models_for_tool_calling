from export.csv_out import to_csv
def test_finance_format():
    out = to_csv([["a", "b"], ["1", "2"]])
    assert out.startswith("\ufeff")
    assert out[1:] == "a;b\n1;2"
