import itertools

import numpy as np
import scipy
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

@pytest.mark.parametrize("dtype", types.scalar_types)
def test_syrk(dtype):
    n = 30
    k = 20
    rng = np.random.default_rng(0)
    for outorder in ("C", "F"):
        for uplo in ("U", "L"):
            a = types.random((n, k), dtype=dtype, order=outorder, rng=rng)
            alpha = rng.random()
            beta = rng.random()
            c = types.random((n, n), dtype=dtype, order=outorder, rng=rng)
            correct_res = alpha * a @ a.T + beta * c
            l3.syrk(uplo, a, c, alpha=alpha, beta=beta)
            if uplo == 'U':
                correct_res = np.triu(correct_res)
                output = np.triu(c)
            else:
                correct_res = np.tril(correct_res)
                output = np.tril(c)
            assert np.allclose(output, correct_res)

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_herk(dtype):
    n = 30
    k = 20
    rng = np.random.default_rng(0)
    for outorder in ("C", "F"):
        for uplo in ("U", "L"):
            a = types.random((n, k), dtype=dtype, order=outorder, rng=rng)
            alpha = rng.random()
            beta = rng.random()
            c = types.random((n, n), dtype=dtype, rng=rng)
            c += c.T.conj()
            c = np.asarray(c, order=outorder)
            correct_res = alpha * a @ a.T.conj() + beta * c
            l3.herk(uplo, 'N', a, c, alpha=alpha, beta=beta)
            if uplo == 'U':
                correct_res = np.triu(correct_res)
                output = np.triu(c)
            else:
                correct_res = np.tril(correct_res)
                output = np.tril(c)
            assert np.allclose(output, correct_res)

@pytest.mark.parametrize("dtype", types.scalar_types)
def test_syr2k(dtype):
    n = 30
    k = 20
    rng = np.random.default_rng(0)
    for outorder in ("C", "F"):
        for a_order in ("C", "F"):
            for uplo in ("U", "L"):
                a = types.random((n, k), dtype=dtype, order=a_order, rng=rng)
                b = types.random((n, k), dtype=dtype, order=a_order, rng=rng)
                alpha = rng.random()
                beta = rng.random()
                c = types.random((n, n), dtype=dtype, order=outorder, rng=rng)
                correct_res = alpha * a @ b.T + alpha * b @ a.T + beta * c
                l3.syr2k(uplo, a, b, c, alpha=alpha, beta=beta)
                if uplo == 'U':
                    correct_res = np.triu(correct_res)
                    output = np.triu(c)
                else:
                    correct_res = np.tril(correct_res)
                    output = np.tril(c)
                assert np.allclose(output, correct_res)

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_her2k(dtype):
    n = 30
    k = 20
    rng = np.random.default_rng(0)
    for outorder in ("C", "F"):
        for uplo in ("U", "L"):
            a = types.random((n, k), dtype=dtype, order=outorder, rng=rng)
            b = types.random((n, k), dtype=dtype, order=outorder, rng=rng)
            alpha = rng.random()
            beta = rng.random()
            c = types.random((n, n), dtype=dtype, order=outorder, rng=rng)
            c += c.T.conj()
            c = np.asarray(c, order=outorder)
            correct_res = alpha * a @ b.T.conj() + alpha * b @ a.T.conj() + beta * c
            l3.her2k(uplo, 'N', a, b, c, alpha=alpha, beta=beta)
            if uplo == 'U':
                correct_res = np.triu(correct_res)
                output = np.triu(c)
            else:
                correct_res = np.tril(correct_res)
                output = np.tril(c)
            assert np.allclose(output, correct_res)

@pytest.mark.parametrize("dtype", types.scalar_types)
def test_trmm(dtype):
    m = 30
    n = 20
    rng = np.random.default_rng(0)
    for outorder in ("C", "F"):
        for aorder in ("C", "F"):
            for transconja in (True, False):
                for uplo in ("U", "L"):
                    A = types.random((m, m), dtype=dtype, order=aorder, rng=rng)
                    A = np.triu(A) if uplo == "U" else np.tril(A)
                    A = np.asarray(A, order=aorder)
                    B = types.random((m, n), dtype=dtype, order=outorder, rng=rng)
                    Acopy = A.copy()
                    if transconja:
                        Acopy = Acopy.T.conj()
                    alpha = types.random((1,), dtype=dtype, rng=rng)[0]
                    correct_res = alpha * Acopy @ B
                    try:
                        l3.trmm(A, B, alpha=alpha, uplo=uplo, transconja=transconja)
                        assert np.allclose(B, correct_res)
                    except ValueError:
                        break

@pytest.mark.parametrize("dtype", types.scalar_types)
def test_trsm(dtype):
    m = 30
    n = 20
    rng = np.random.default_rng(0)
    for outorder in ("C", "F"):
        for aorder in ("C", "F"):
            for transconja in (True, False):
                for uplo in ("U", "L"):
                    A = types.random((m, m), dtype=dtype, rng=rng) + 1e-1 * np.eye(m)
                    A = np.triu(A) if uplo == "U" else np.tril(A)
                    A = np.asarray(A, order=aorder)
                    B = types.random((m, n), dtype=dtype, order=outorder, rng=rng)
                    Acopy = A.copy()
                    if transconja:
                        Acopy = Acopy.T.conj()
                    alpha = types.random((1,), dtype=dtype, rng=rng)[0]
                    correct_res = alpha * scipy.linalg.solve(Acopy, B)
                    try:
                        l3.trsm(A, B, alpha=alpha, uplo=uplo, transconja=transconja)
                        assert np.allclose(B, correct_res)
                    except ValueError:
                        break
