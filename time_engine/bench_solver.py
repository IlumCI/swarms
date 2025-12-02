#!/usr/bin/env python3
import time
import numpy as np
from engine import SimulacrumEngine

def bench_cpu(grid_size=(128,128), iterations=10):
    eng = SimulacrumEngine(db_path=":memory:")
    adapter = eng.PDEAdapter2DCpu({"alpha":1e-3})
    grid = np.random.randn(*grid_size)
    t0 = time.perf_counter()
    for _ in range(iterations):
        out = adapter.step(grid, 0.01, {"alpha":1e-3})
    dt = (time.perf_counter() - t0) / iterations
    print(f"CPU 2D step avg: {dt*1000:.3f} ms")

def bench_gpu(grid_size=(128,128), iterations=10):
    try:
        from adapters.pde_gpu import PDEAdapterGpu
    except Exception as e:
        print("GPU adapter not available:", e)
        return
    import numpy as np
    eng = SimulacrumEngine(db_path=":memory:")
    adapter = PDEAdapterGpu({"alpha":1e-3})
    grid = np.random.randn(*grid_size)
    t0 = time.perf_counter()
    for _ in range(iterations):
        out = adapter.step(grid, 0.01, {"alpha":1e-3})
    dt = (time.perf_counter() - t0) / iterations
    print(f"GPU 2D step avg (host transfer included): {dt*1000:.3f} ms")

def bench_surrogate(grid_size=(4,4), iterations=10):
    try:
        from adapters.surrogate_torch import SurrogateAdapterTorch
    except Exception as e:
        print("Surrogate adapter not available:", e)
        return
    eng = SimulacrumEngine(db_path=":memory:")
    adapter = SurrogateAdapterTorch({"input_size": grid_size[0]*grid_size[1]}, device="cpu")
    grid = np.random.randn(*(grid_size))
    t0 = time.perf_counter()
    for _ in range(iterations):
        out = adapter.predict(grid, 0.01)
    dt = (time.perf_counter() - t0) / iterations
    print(f"Surrogate predict avg: {dt*1000:.3f} ms")

if __name__ == "__main__":
    bench_cpu()
    bench_gpu()
    bench_surrogate()


