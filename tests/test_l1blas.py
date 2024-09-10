from pymkl.cblas import l1
from pymkl.util import types
import numpy as np

def test_asum():
    for dtype in types.scalar_types:
        A = types.random((10,), dtype)
        assert np.isclose(l1.asum(A),
                          np.sum(np.abs(A.real)) +
                          np.sum(np.abs(A.imag)))