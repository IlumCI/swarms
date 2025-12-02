import numpy as np
from engine import SimulacrumEngine

def test_integrate_rk45_exp():
    eng = SimulacrumEngine(db_path=":memory:")
    # dx/dt = a * x, solution x(t) = x0 * exp(a t)
    a = 0.5
    def f(x, t):
        return a * x
    x0 = np.array([1.0])
    x_end = eng.integrate_rk45(x0, f, 0.0, 1.0, atol=1e-9, rtol=1e-7)
    expected = np.exp(a * 1.0)
    assert abs(float(x_end[0]) - expected) < 1e-6


