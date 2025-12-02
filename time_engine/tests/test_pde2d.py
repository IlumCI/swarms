def test_pde2d_diffusion_basic():
    import numpy as np
    from engine import SimulacrumEngine
    eng = SimulacrumEngine(db_path=":memory:")
    adapter = eng.PDEAdapter2DCpu({"alpha":1e-3})
    grid = np.zeros((10,10))
    grid[5,5] = 1.0
    out = adapter.step(grid, 0.1, {"alpha":1e-3})
    # energy should spread; center lower than 1
    assert out[5,5] < 1.0


