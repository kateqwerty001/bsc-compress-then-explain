def test_imports():
    import bonXAI            
    from bonXAI.core import metrics 

def test_version_attr_exists():
    import bonXAI
    assert hasattr(bonXAI, "__package__")