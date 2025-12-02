def test_get_physics_adapter_and_step():
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    ph_pde = {"field_model": "pde_cpu", "params": {"alpha": 1e-3}}
    adapter = eng.get_physics_adapter(ph_pde)
    assert adapter is not None
    out = adapter.step([1.0, 2.0, 3.0], 0.1, ph_pde.get("params", {}))
    assert isinstance(out, list)
    ph_sur = {"field_model": "surrogate_dummy", "params": {"scale": 0.5}}
    adapter2 = eng.get_physics_adapter(ph_sur)
    assert adapter2 is not None
    out2 = adapter2.predict([1.0, 2.0], 0.1)
    assert out2[0] == 0.5


