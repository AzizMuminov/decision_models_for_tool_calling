from textkit.slug import slugify


def test_punctuation():
    assert slugify("Hello, World!") == "hello-world"


def test_accents():
    assert slugify("  Crème Brûlée ") == "creme-brulee"


def test_repeated_separators():
    assert slugify("a -- b") == "a-b"
