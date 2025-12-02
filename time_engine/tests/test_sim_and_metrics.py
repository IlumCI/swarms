def test_run_simulation_smoke():
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    cfg = eng.get_default_universe_config()
    seed = cfg["initial"]
    out = eng.run_simulation(seed_state=seed, horizon=2, ensemble=2, rng_seed=12345)
    assert "sim_id" in out
    sim_id = out["sim_id"]
    sims = eng.list_simulations(limit=10)
    assert any(s["id"] == sim_id for s in sims)


def test_metrics_and_health_smoke():
    # import JobManager locally to avoid starting servers
    from main import JobManager
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    mgr = JobManager(eng)
    # starting a PF job should increment metric counter
    start = mgr.start_pf_job({"seed": eng.get_default_universe_config()["initial"], "num_particles": 8, "horizon": 4})
    assert start.get("status") == "pf_started"
    assert mgr._metrics.get("pf_jobs", 0) >= 1
    # basic DB health check
    cur = eng._conn.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchone()[0] == 1


