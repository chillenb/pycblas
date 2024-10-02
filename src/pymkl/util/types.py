import ctypes

import numpy as np

MKL_INT = ctypes.c_int
c_float_p = ctypes.POINTER(ctypes.c_float)
c_double_p = ctypes.POINTER(ctypes.c_double)

scalar_types = [np.float32, np.float64, np.complex64, np.complex128]
type2index = {np.float32: 0, np.float64: 1, np.complex64: 2, np.complex128: 3}
type2prefix = {np.float32: "s", np.float64: "d", np.complex64: "c", np.complex128: "z"}


def scalar_arg_to_ctype(nptype, scalar):
    if nptype == np.float32:
        return ctypes.c_float(scalar)
    if nptype == np.float64:
        return ctypes.c_double(scalar)
    if nptype == np.complex64:
        return (ctypes.c_float * 2)(scalar.real, scalar.imag)
    if nptype == np.complex128:
        return (ctypes.c_double * 2)(scalar.real, scalar.imag)
    msg = f"Unsupported type: {nptype}"
    raise ValueError(msg)

def scalar_arg_to_real_ctype(nptype, scalar):
    if nptype == np.float32:
        return ctypes.c_float(scalar)
    if nptype == np.float64:
        return ctypes.c_double(scalar)
    if nptype == np.complex64:
        return ctypes.c_float(scalar.real)
    if nptype == np.complex128:
        return ctypes.c_double(scalar.real)
    msg = f"Unsupported type: {nptype}"
    raise ValueError(msg)

def random(shape, dtype=np.float64, order="C", rng=None):
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
    if rng is None:
        rng = np.random.default_rng()
    if dtype not in scalar_types:
        msg = f"Unsupported type: {dtype}"
        raise ValueError(msg)
    if dtype in (np.complex64, np.complex128):
        real = rng.random(shape).astype(dtype)
        imag = rng.random(shape).astype(dtype)
        arr = real + 1j * imag
    else:
        arr = rng.random(shape).astype(dtype)
    return np.asarray(arr, order=order)


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
    if nptype == np.float64:
        return "d"
    if nptype == np.complex64:
        return "c"
    if nptype == np.complex128:
        return "z"
    msg = f"Unsupported type: {nptype}"
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


def check_nd_types(*args):
    """Check that all arguments are arrays of the same type

    Returns
    -------
    numpy.dtype.type
        Datatype of the input arrays
    """
    if not args:
        msg = "At least one argument must be provided"
        raise ValueError(msg)
    scalar_type = args[0].dtype.type
    for arg in args:
        if arg.dtype.type != scalar_type:
            msg = "All arguments must have the same type"
            raise ValueError(msg)
    return scalar_type


def check_matmul_shapes(a, b, c):
    """Check that the shapes of the input arrays are compatible for matrix multiplication.

    Parameters
    ----------
    a : array_like
        First input array, shape (m, k)
    b : array_like
        Second input array, shape (k, n)
    c : array_like
        Output array, shape (m, n)

    Returns
    -------
    tuple
        (m, n, k)
    """
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if b.shape[0] == k and c.shape == (m, n):
        return m, n, k
    msg = f"Bad matmul shapes: {a.shape}, {b.shape} -> {c.shape}"
    raise ValueError(msg)

def check_vmatmul_shapes(a, b, c):
    """Check that the shapes of the input arrays are compatible for matrix multiplication.

    Parameters
    ----------
    a : array_like
        First input array, shape (m, k)
    b : array_like
        Second input array, shape (k, n)
    c : array_like
        Output array, shape (m, n)

    Returns
    -------
    tuple
        (m, n, k)
    """
    if a.shape[0] != c.shape[0] or a.shape[0] != b.shape[0]:
        msg = f"Bad vmatmul shapes: {a.shape}, {b.shape} -> {c.shape}"
        raise ValueError(msg)
    nbatch = a.shape[0]
    m = a.shape[-2]
    k = a.shape[-1]
    n = b.shape[-1]
    if b.shape[-2] == k and c.shape[-2] == m and c.shape[-1] == n:
        return nbatch, m, n, k
    msg = f"Bad matmul shapes: {a.shape}, {b.shape} -> {c.shape}"
    raise ValueError(msg)