def test_imports():
    import thinx
    from thinx.core import metrics

def test_version_attr_exists():
    import thinx
    assert hasattr(thinx, "__package__")