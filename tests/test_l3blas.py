import itertools

import numpy as np
import pytest

from pymkl.cblas import l3
from pymkl.util import types


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

@pytest.mark.parametrize("dtype", types.scalar_types)
def test_symm(dtype):
    m = 30
    n = 20
    rng = np.random.default_rng(0)
    for outorder in ("C", "F"):
        for side in ("L", "R"):
            for uplo in ("U", "L"):
                a = types.random((m, m), dtype=dtype, order=outorder, rng=rng)
                a += a.T
                alpha = rng.random()
                beta = rng.random()
                if side == "L":
                    b = types.random((m, n), dtype=dtype, order=outorder, rng=rng)
                    c = types.random((m, n), dtype=dtype, order=outorder, rng=rng)
                    correct_res = alpha * a @ b + beta * c
                else:
                    b = types.random((n, m), dtype=dtype, order=outorder, rng=rng)
                    c = types.random((n, m), dtype=dtype, order=outorder, rng=rng)
                    correct_res = alpha * b @ a + beta * c
                l3.symm(side, uplo, a, b, c, alpha=alpha, beta=beta)
                assert np.allclose(c, correct_res)

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_hemm(dtype):
    m = 30
    n = 20
    rng = np.random.default_rng(0)
    for outorder in ("C", "F"):
        for side in ("L", "R"):
            for uplo in ("U", "L"):
                a = types.random((m, m), dtype=dtype, order=outorder, rng=rng)
                a += a.T.conj()
                alpha = rng.random()
                beta = rng.random()
                if side == "L":
                    b = types.random((m, n), dtype=dtype, order=outorder, rng=rng)
                    c = types.random((m, n), dtype=dtype, order=outorder, rng=rng)
                    correct_res = alpha * a @ b + beta * c
                else:
                    b = types.random((n, m), dtype=dtype, order=outorder, rng=rng)
                    c = types.random((n, m), dtype=dtype, order=outorder, rng=rng)
                    correct_res = alpha * b @ a + beta * c
                l3.hemm(side, uplo, a, b, c, alpha=alpha, beta=beta)
                assert np.allclose(c, correct_res)