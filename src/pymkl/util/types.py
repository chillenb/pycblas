import numpy as np
import ctypes

_MKL_INT = ctypes.c_int
c_float_p = ctypes.POINTER(ctypes.c_float)
c_double_p = ctypes.POINTER(ctypes.c_double)

scalar_types = [np.float32, np.float64, np.complex64, np.complex128]

def random(shape, dtype=np.float64):
    """Generate a random array

    Parameters
    ----------
    shape : tuple of int
        Shape of the output array
    dtype : numpy.dtype or np.dtype.type
        Datatype of the output array

    Returns
    -------
    numpy.ndarray
        Random array
    """
    if dtype not in scalar_types:
        msg = "Unsupported type: {}".format(dtype)
        raise ValueError(msg)
    if dtype == np.complex64 or dtype == np.complex128:
        real = np.random.random(shape).astype(dtype)
        imag = np.random.random(shape).astype(dtype)
        return real + 1j * imag
    return np.random.random(shape).astype(dtype)


def typechar(nptype):
    """Return the type character for a given numpy type

    Parameters
    ----------
    nptype : numpy.dtype or np.dtype.type
        Input numpy datatype

    Returns
    -------
    str
        Type character
    """
    if nptype == np.float32:
        return "s"
    elif nptype == np.float64:
        return "d"
    elif nptype == np.complex64:
        return "c"
    elif nptype == np.complex128:
        return "z"
    else:
        msg = "Unsupported type: {}".format(nptype)
        raise ValueError(msg)


def check_vec_types(*args):
    """Check that all arguments are 1D arrays of the same type and size

    Returns
    -------
    numpy.dtype.type
        Datatype of the input arrays
    int
        Size of the input arrays
    """
    if not args:
        msg = "At least one argument must be provided"
        raise ValueError(msg)
    scalar_type = args[0].dtype.type
    size = args[0].size
    for arg in args:
        if arg.size != size:
            msg = "All arguments must have the same size"
            raise ValueError(msg)
        if arg.dtype.type != scalar_type:
            msg = "All arguments must have the same type"
            raise ValueError(msg)
        if arg.ndim != 1:
            msg = "All arguments must be 1D arrays"
            raise ValueError(msg)
    return scalar_type, size
