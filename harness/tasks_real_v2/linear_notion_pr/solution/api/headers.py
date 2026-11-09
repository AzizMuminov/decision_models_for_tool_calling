def deprecation_headers(sunset_date: str) -> dict:
    return {"Deprecation": "true", "Sunset": sunset_date, "Link": '<https://developer.acme.example/migrate>; rel="deprecation"'}
