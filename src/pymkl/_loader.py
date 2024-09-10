import numpy as np
from threadpoolctl import MKLController

_mkl_ctrl = MKLController()
_mkl_lib = _mkl_ctrl.dynlib
