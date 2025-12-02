try:
    import cupy as cp
except Exception:
    cp = None

class PDEAdapterGpu:
    """GPU PDE adapter using CuPy. Requires CuPy installed."""
    def __init__(self, params=None):
        if cp is None:
            raise RuntimeError("CuPy is not installed; PDEAdapterGpu unavailable.")
        self.params = params or {}
        self.alpha = float(self.params.get("alpha", 1e-3))

    def step(self, local_grid, dt, params):
        # convert to cupy array
        arr = cp.asarray(local_grid, dtype=cp.float64)
        alpha = float(params.get("alpha", self.alpha))
        # periodic laplacian
        lap = cp.roll(arr, -1, axis=0) + cp.roll(arr, 1, axis=0) + cp.roll(arr, -1, axis=1) + cp.roll(arr, 1, axis=1) - 4 * arr
        out = arr + dt * alpha * lap
        # return as host numpy array
        return cp.asnumpy(out)


