import itertools

import numpy as np
import scipy
import pytest

from pymkl.cblas import blasext
from pymkl.util import types


@pytest.mark.parametrize("dtype", types.scalar_types)
def test_gemm_batch_strided(dtype):
    nstride = 3
    m = 30
    n = 20
    k = 10
    rng = np.random.default_rng(0)
    for outorder, aorder, border in itertools.product(("C", "F"), repeat=3):
        if outorder == 'F':
            c = types.random((nstride, n, m), dtype=dtype, rng=rng)
            c = np.moveaxis(c, -1, -2)
        else:
            c = types.random((nstride, m, n), dtype=dtype, rng=rng)
        if aorder == 'F':
            a = types.random((nstride, k, m), dtype=dtype, rng=rng)
            a = np.moveaxis(a, -1, -2)
        else:
            a = types.random((nstride, m, k), dtype=dtype, rng=rng)
        if border == 'F':
            b = types.random((nstride, n, k), dtype=dtype, rng=rng)
            b = np.moveaxis(b, -1, -2)
        else:
            b = types.random((nstride, k, n), dtype=dtype, rng=rng)
        alpha = rng.random()
        beta = rng.random()
        correct_res = alpha * a @ b + beta * c
        blasext.gemm_batch_strided(a, b, c, alpha=alpha, beta=beta)
        assert np.allclose(c, correct_res)
