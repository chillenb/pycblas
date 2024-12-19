import numpy as np

from pycblas.cblas import l1
from pycblas.util import types


def test_asum():
    for dtype in types.scalar_types:
        A = types.random((10,), dtype)
        assert np.isclose(l1.asum(A), np.sum(np.abs(A.real)) + np.sum(np.abs(A.imag)))
