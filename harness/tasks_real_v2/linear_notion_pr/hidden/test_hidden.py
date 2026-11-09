from api.headers import deprecation_headers
def test_headers():
    h = deprecation_headers("Wed, 31 Mar 2027 00:00:00 GMT")
    assert h["Deprecation"] == "true"
    assert h["Sunset"] == "Wed, 31 Mar 2027 00:00:00 GMT"
    assert 'rel="deprecation"' in h["Link"] and "developer.acme.example/migrate" in h["Link"]
