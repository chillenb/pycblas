import ctypes
import os
import sys
from pathlib import Path

import numpy as np  # noqa: F401


def _try_find_mkl(dirpath):
    mkl_libs = ['libmkl_rt.so', 'libmkl_rt.dylib', 'mkl_rt.dll']
    for lib in mkl_libs:
        mkl_lib = dirpath / lib
        if mkl_lib.exists():
            return ctypes.CDLL(str(mkl_lib))
    return None

def _load_mkl():
    try:
        import threadpoolctl
        _threadpoolctrl = threadpoolctl.ThreadpoolController()
    except ImportError:
        _threadpoolctrl = None
    if _threadpoolctrl is not None:
        _controllers = _threadpoolctrl.select(internal_api='mkl').lib_controllers
        if _controllers and _controllers[0].dynlib is not None:
            return _controllers[0].dynlib
    if 'MKLROOT' in os.environ:
        is_64bit = sys.maxsize > 2**32
        if is_64bit:
            mkldir = Path(os.environ['MKLROOT']) / 'lib' / 'intel64'
        else:
            mkldir = Path(os.environ['MKLROOT']) / 'lib' / 'ia32'
        candidate = _try_find_mkl(mkldir)
        if candidate is not None:
            return candidate
    if 'LD_LIBRARY_PATH' in os.environ:
        for libpath in os.environ['LD_LIBRARY_PATH'].split(':'):
            candidate = _try_find_mkl(Path(libpath))
            if candidate is not None:
                return candidate
    if 'CONDA_PREFIX' in os.environ:
        conda_dir = Path(os.environ['CONDA_PREFIX'])
        candidate = _try_find_mkl(conda_dir / 'lib')
        if candidate is not None:
            return candidate
    return None

_mkl_lib = _load_mkl()
