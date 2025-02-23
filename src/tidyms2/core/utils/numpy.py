"""Serializable numpy array types."""

from __future__ import annotations

import base64
import json
from typing import Iterable, Literal, TypeVar

import numpy
from numpy import floating, frombuffer, integer
from numpy.typing import NDArray
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated


def array_to_json_str(arr: NDArray) -> str:
    """Serialize a numpy array as a JSON string.

    :param arr: The numpy array to serialize
    :return: JSON string with the following three fields. `dtype` store the array dtype, `shape` contains the array
        shape and `base64_bytes` stores the array data in base64 format.

    """
    d = {
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "base64_bytes": base64.b64encode(arr.tobytes()).decode("utf8"),
    }
    return json.dumps(d)


def json_str_to_array(s: str):
    """Decode a string generated with array_to_json_str into a numpy array.

    :param s: A string serialized numpy array
    :return: a new numpy instance

    """
    d = json.loads(s)
    dtype = d["dtype"]
    shape = d["shape"]
    data = base64.b64decode(bytes(d["base64_bytes"], "utf8"))
    return frombuffer(data, dtype=dtype).reshape(shape).copy()


def validate_serializable_array(arr: NDArray) -> NDArray:
    """Create an array if a serialized string is provided."""
    if isinstance(arr, str):
        arr = json_str_to_array(arr)
    return arr


def cartesian_product_from_ranges(*args: list[int]) -> IntArray:
    """Create an array where each row is an element in the cartesian product.

    :param args: the list of integers to build the cartesian product

    """
    product = numpy.zeros(shape=(1, 0), dtype=int)

    for x in args:
        n_row, n_col = product.shape
        new_shape = (n_row * len(x), n_col + 1)
        new_product = numpy.zeros(shape=new_shape, dtype=int)
        ind = numpy.repeat(numpy.arange(n_row), len(x))
        new_column = numpy.tile(x, n_row)
        new_product[:, :n_col] = product[ind]
        new_product[:, -1] = new_column
        product = new_product
    return product


def cartesian_product_from_iterable(*args: Iterable):
    """Compute the cartesian product of args as a 2D array."""
    if not args:
        raise ValueError("At least one argument is required to compute the cartesian product.")
    res = None
    for x in args:
        if res is None:
            # initialize cartesian product array
            res = numpy.array(x)
            res = res.reshape((res.size, 1))
        else:
            x = numpy.array(x)
            row, col = res.shape
            new_res_shape = (row * x.size, col + 1)
            new_res = numpy.zeros(shape=new_res_shape, dtype=res.dtype)
            ind = numpy.repeat(numpy.arange(row), x.size)
            new_col = numpy.tile(x, row)
            new_res[:, :col] = res[ind]
            new_res[:, -1] = new_col
            res = new_res
    return res


def _find_closest_sorted(x: numpy.ndarray, xq: numpy.ndarray | float | int) -> IntArray1D:
    """Find the index in x closest to each xq element. Assumes that x is sorted.

    :param x: Sorted 1D array
    :param xq: search vector
    :return: an int array with indices of elements in `x` closest to each `xq` element.
    :raises ValueError: when x or xq are empty.

    """
    if isinstance(xq, (float, int)):
        xq = numpy.array(xq)

    if x.size == 0:
        msg = "`x` must be a non-empty array."
        raise ValueError(msg)

    if xq.size == 0:
        return numpy.array([], dtype=int)

    ind = numpy.searchsorted(x, xq)

    if ind.size == 1:
        if ind == 0:
            return ind
        elif ind == x.size:
            return ind - 1
        else:
            return ind - ((xq - x[ind - 1]) < (x[ind] - xq))

    else:
        # cases where the index is between 1 and x.size - 1
        mask = (ind > 0) & (ind < x.size)
        ind[mask] -= (xq[mask] - x[ind[mask] - 1]) < (x[ind[mask]] - xq[mask])
        # when the index is x.size, then the closest index is x.size -1
        ind[ind == x.size] = x.size - 1
        return ind


def find_closest(x: numpy.ndarray, xq: numpy.ndarray | float | int, is_sorted: bool = True) -> IntArray1D:
    """Find the index in `x` closest to each `xq` element.

    :param x: reference array
    :param xq: search vector
    :param is_sorted: if set to ``True``, assumes that `x` is sorted.
    :return: an int array with indices of elements in `x` closest to each `xq` element.
    :raises ValueError: when x or xq are empty.

    """
    if is_sorted:
        return _find_closest_sorted(x, xq)
    else:
        sorted_index = numpy.argsort(x)
        closest_index = _find_closest_sorted(x[sorted_index], xq)
        return sorted_index[closest_index]


def gauss(grid: numpy.ndarray, mu: float, sigma: float, amp: float) -> numpy.ndarray:
    """Create a gaussian curve from a grid.

    :param grid: an array with grid points used to create the gaussian curve
    :param mu : the gaussian mean
    :param sigma: the gaussian standard deviation
    :param amp: the gaussian amplitude

    """
    gaussian = amp * numpy.power(numpy.e, -0.5 * ((grid - mu) / sigma) ** 2)
    return gaussian


def gaussian_mixture(x: numpy.ndarray, *gaussian_params: tuple[float, float, float]) -> numpy.ndarray:
    """Create a gaussian mixture using a grid.

    :param grid: an array with grid points used to create the gaussian curve
    :param gaussian_params: a tuple of mean, standard deviation and amplitude for each gaussian component
    """
    components = numpy.zeros((len(gaussian_params), x.size))
    for row, component_params in enumerate(gaussian_params):
        components[row] = gauss(x, *component_params)
    return components.sum(axis=0)


def check_matrix_shape(X: NDArray) -> None:
    """Check that the array has the expected data matrix shape."""
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array.")

    n_rows, n_cols = X.shape

    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"X must be a 2D non-empty array. Got shape {X.shape}")


FloatDtype = TypeVar("FloatDtype", bound=floating)
IntDtype = TypeVar("IntDtype", bound=integer)


FloatArray = Annotated[
    NDArray[FloatDtype],
    BeforeValidator(validate_serializable_array),
    PlainSerializer(array_to_json_str, return_type=str),
]

IntArray = Annotated[
    NDArray[IntDtype],
    BeforeValidator(validate_serializable_array),
    PlainSerializer(array_to_json_str, return_type=str),
]

FloatArray1D = Annotated[
    NDArray[FloatDtype],
    Literal["N"],
    BeforeValidator(validate_serializable_array),
    PlainSerializer(array_to_json_str, return_type=str),
]

IntArray1D = Annotated[
    NDArray[IntDtype],
    Literal["N"],
    BeforeValidator(validate_serializable_array),
    PlainSerializer(array_to_json_str, return_type=str),
]
