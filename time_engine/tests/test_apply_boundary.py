def test_apply_boundary_modes():
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    state = {"a":1.0,"phi":2.0,"v":3.0}
    # causal_sampling should perturb values but keep keys
    b1 = {"type":"causal_sampling","params":{"sigma":0.0,"vars":["a","phi"],"seed":42}}
    s1 = eng.apply_boundary(dict(state), b1, t=0)
    assert "a" in s1 and "phi" in s1
    # reflective with bounds clamps
    b2 = {"type":"reflective","params":{"min":0.5,"max":2.5,"vars":["a","phi"]}}
    s2 = eng.apply_boundary(dict(state), b2, t=0)
    assert 0.5 <= s2["a"] <= 2.5
    # absorptive decays
    b3 = {"type":"absorptive","params":{"decay":0.5,"vars":["v"]}}
    s3 = eng.apply_boundary(dict(state), b3, t=0)
    assert s3["v"] == 1.5


