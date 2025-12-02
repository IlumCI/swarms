def test_evolve_state_dispatch():
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    state = {"field1": 1.0, "field2": 2.0, "volume": 100.0}
    # RK45 path
    s1 = eng.evolve_state(state, dt=0.1, method="RK45", phidesc={"params":{"lambda":0.01}})
    assert isinstance(s1, dict)
    # Leapfrog path
    s2 = eng.evolve_state(state, dt=0.1, method="Leapfrog", phidesc={"params":{"lambda":0.01}})
    assert isinstance(s2, dict)


