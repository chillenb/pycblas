CblasRowMajor = 101
CblasColMajor = 102

CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113

CblasUpper = 121
CblasLower = 122

CblasNonUnit = 131
CblasUnit = 132

CblasLeft = 141
CblasRight = 142


def get_elem_strides(arr):
    """Get the strides of an array in elements, not bytes

    Parameters
    ----------
    arr : numpy.ndarray
        Input array

    Returns
    -------
    tuple
        Tuple of strides
    """
    elem_size = arr.dtype.itemsize
    elem_strides = tuple(s // elem_size for s in arr.strides)
    for byte_stride, el_stride in zip(arr.strides, elem_strides):
        if byte_stride != el_stride * elem_size:
            msg = "Array strides are not aligned with element size"
            raise ValueError(msg)
    return elem_strides


def leading_dimension_and_order(arr):
    """Get the leading dimension and memory order of an array

    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    """
    if arr.ndim != 2:
        msg = "Array must be 2D"
        raise ValueError(msg)
    elem_strides = get_elem_strides(arr)
    if elem_strides[0] == 1:
        order = CblasColMajor
        ld = elem_strides[1]
    elif elem_strides[1] == 1:
        order = CblasRowMajor
        ld = elem_strides[0]
    else:
        msg = f"Array with strides {elem_strides} is in neither row-major nor column-major order"
        raise ValueError(msg)
    return ld, order


def get_array_args(output_layout, arr):
    """Get the array arguments for a BLAS function

    Parameters
    ----------
    output_layout : int
        Layout of the output array (CblasRowMajor or CblasColMajor)
    arr : numpy.ndarray
        Input array

    Returns
    -------
    outarr : numpy.ndarray
        Array laid out in the output layout
    trans : int
        CblasNoTrans or CblasTrans
    ld : int
        Leading dimension of the array
    """
    assert arr.ndim == 2
    ld, order = leading_dimension_and_order(arr)
    if order == output_layout:
        outarr = arr
        trans = CblasNoTrans
    else:
        outarr = arr.T
        trans = CblasTrans
        ld, order = leading_dimension_and_order(arr.T)
        assert order == output_layout
    return outarr, trans, ld
