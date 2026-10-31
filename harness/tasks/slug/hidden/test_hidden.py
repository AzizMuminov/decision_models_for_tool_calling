from textkit.slug import slugify


def test_visible_cases():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  Crème Brûlée ") == "creme-brulee"
    assert slugify("a -- b") == "a-b"


def test_more():
    assert slugify("Ça va?") == "ca-va"
    assert slugify("") == ""
