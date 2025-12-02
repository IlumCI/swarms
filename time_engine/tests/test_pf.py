def test_pf_basic_smoke():
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    cfg = eng.get_default_universe_config()
    seed = cfg["initial"]
    # synthetic observation at t=0 close to seed
    obs = {0: {"ETH_price": 100.5}}
    pf = eng.particle_filter(initial_state=seed, observations=obs, num_particles=32, horizon=1, rng_seed=1)
    assert "particles_history" in pf
    assert len(pf["particles_history"]) >= 1


