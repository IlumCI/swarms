def test_initialize_state_deterministic():
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    ph = {"seed_config": {"t_init":1e-36,"a_0":1.0,"phi_0":2.0,"noise_seed":123}}
    s1 = eng.initialize_state(ph, ph["seed_config"])
    s2 = eng.initialize_state(ph, ph["seed_config"])
    assert s1 == s2


