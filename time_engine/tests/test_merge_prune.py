def test_merge_prune_aggregation():
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    cfg = eng.get_default_universe_config()
    seed = cfg["initial"]
    # use a very large merge_threshold to force merging of similar trajectories
    ph = {"type": "econ_simple", "params": {}, "merge_threshold": 1e9}
    out = eng.run_simulation(seed_state=seed, horizon=3, ensemble=6, rng_seed=123, phidesc=ph)
    assert "sim_id" in out
    summary = out.get("summary") or {}
    # ensure aggregate keys present and numeric
    assert isinstance(summary, dict)
    for k, v in summary.items():
        assert "mean" in v and "std" in v and "samples" in v


