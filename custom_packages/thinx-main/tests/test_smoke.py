def test_imports():
    import thinX
    from thinX.core import metrics

def test_version_attr_exists():
    import thinX
    assert hasattr(thinX, "__package__")