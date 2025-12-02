def test_multiscale_run_basic():
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    seed = eng.get_default_universe_config()["initial"]
    ph = {"tiles": {"n_tiles": 3}, "physics_modes": {"gravity_model":"FLRW"}, "seed_config": {"a_0":1.0, "phi_0":0.5, "noise_seed":42}, "boundary_model": {"type":"causal_sampling", "params": {"sigma":1e-6, "vars":["field1"]}}}
    out = eng.run_simulation(seed_state=seed, horizon=2, phidesc=ph, rng_seed=123)
    assert "tiles" in out or "summary" in out


