from __future__ import annotations

import importlib.metadata

import pycblas as m


def test_version():
    assert importlib.metadata.version("pycblas") == m.__version__
