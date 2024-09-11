from pymkl.cblas import l3
from pymkl.util import types
import numpy as np
import itertools

import pytest

@pytest.mark.parametrize("dtype", types.scalar_types)
def test_gemm(dtype):
    m = 30
    n = 20
    k = 10
    rng = np.random.default_rng(0)
    for outorder, aorder, border in itertools.product(("C", "F"), repeat=3):
        c = types.random((m, n), dtype=dtype, order=outorder, rng=rng)
        a = types.random((m, k), dtype=dtype, order=aorder, rng=rng)
        b = types.random((k, n), dtype=dtype, order=border, rng=rng)
        alpha = rng.random()
        beta = rng.random()
        correct_res = alpha * a @ b + beta * c
        l3.gemm(a, b, c, alpha=alpha, beta=beta)
        assert np.allclose(c, correct_res)
