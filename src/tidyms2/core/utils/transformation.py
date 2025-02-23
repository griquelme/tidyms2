"""Transformation utilities for numpy arrays."""

from typing import assert_never

import numpy

from ..enums import NormalizationMethod, ScalingMethod
from .numpy import FloatArray, check_matrix_shape


def scale(X: FloatArray, method: ScalingMethod | str = ScalingMethod.AUTOSCALING) -> FloatArray:
    """Scale the columns of a 2D array.

    :param X: The 2D array to scale
    :method: Available scaling methods.``"autoscaling"`` applies mean centering and scaling to
        unitary population variance. ``"rescaling"`` scales data to the :math:`[0, 1]` range. ``"pareto"``
        performs mean centering and scaling using the square root of the population standard deviation
    :return: A scaled 2D array

    """
    check_matrix_shape(X)

    if not isinstance(method, ScalingMethod):
        method = ScalingMethod(method)

    match method:
        case ScalingMethod.AUTOSCALING:
            return (X - numpy.nanmean(X, axis=0, keepdims=True)) / numpy.nanstd(X, axis=0, keepdims=True)
        case ScalingMethod.RESCALING:
            Xmin = numpy.nanmin(X, axis=0, keepdims=True)
            return (X - Xmin) / (numpy.nanmax(X, axis=0, keepdims=True) - Xmin)
        case ScalingMethod.PARETO:
            return (X - numpy.nanmean(X, axis=0, keepdims=True)) / numpy.sqrt(numpy.nanstd(X, axis=0, keepdims=True))
        case _ as never:
            assert_never(never)


def normalize(
    X: FloatArray, method: NormalizationMethod | str = NormalizationMethod.SUM, index: int | None = None
) -> FloatArray:
    """Normalize rows of a 2D array.

    :param X: the 2D array to normalize
    :param method: the normalization method. ``"sum"`` normalizes rows using the sum of values in each row,
        ``"max"` normalizes rows using the maximum value of each row. ``"euclidean"`` normalizes rows
        using the euclidean norm of the row. ``"feature"`` normalizes rows using the value of a column.
    :param index: the column index used for ``"feature"`` normalization. This parameter for all other feature
        normalization methods.
    :return: the normalized array. Rows values may be set to NaN or Inf in the following cases: for `"sum"`,
        `"max"` or `"euclidean"` normalization, if all values in the row are `0.0` or NaN, all values will be
        set to `NaN`. For `"feature"` NaN value will be obtained if the reference column value is NaN or if
        column value is NaN. NaN values will also be obtained if the reference feature and column value are both
        0.0. If only the reference feature value is 0.0, then Inf will be obtained.

    """
    check_matrix_shape(X)

    if not isinstance(method, NormalizationMethod):
        method = NormalizationMethod(method)

    if method is NormalizationMethod.FEATURE and index is None:
        raise ValueError("`index` must be specified for feature normalization.")

    match method:
        case NormalizationMethod.SUM:
            factor = numpy.nansum(X, axis=1, keepdims=True)
        case NormalizationMethod.MAX:
            factor = numpy.nanmax(X, axis=1, keepdims=True)
        case NormalizationMethod.EUCLIDEAN:
            factor = numpy.sqrt(numpy.nansum(X**2, axis=1, keepdims=True))
        case NormalizationMethod.FEATURE:
            factor = X[:, index]
        case _ as never:
            assert_never(never)
    return X / factor
