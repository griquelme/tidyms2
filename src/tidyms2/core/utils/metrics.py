"""Core metrics used by the data matrix."""

from typing import assert_never, cast

import numpy
from scipy.stats import median_abs_deviation, pearsonr, rankdata
from sklearn.decomposition import PCA

from ..enums import CorrelationMethod, NormalizationMethod, ScalingMethod
from .numpy import FloatArray, FloatArray1D, check_matrix_shape
from .transformation import normalize, scale


def cv(X: FloatArray, robust: bool = False) -> FloatArray1D:
    """Compute the :term:`CV` on the columns of a 2D array.

    The CV is computed as the quotient between the sample standard deviation and the sample mean.

    NaN values are ignored in the calculation.

    :param X: a 2D numpy array
    :param robust: If set to ``True`` uses the median and median absolute deviation instead of
        sample mean and sample standard deviation.
    :return: a 1D array with the CV of each column. If robust is set to ``False`` and less than two values in
        a column are not NaN, a NaN value will be returned. If all values in the column are zero or NaN, a NaN
        value will be obtained.

    """
    check_matrix_shape(X)
    x_std = _sample_std(X, robust=robust)
    x_mean = numpy.nanmedian(X, axis=0) if robust else numpy.nanmean(X, axis=0)
    return x_std / x_mean


def detection_rate(X: FloatArray, threshold: float | FloatArray = 0.0) -> FloatArray1D:
    """Compute the detection rate for all columns.

    The detection rate is defined as the fraction of elements in a column greater or equal than a threshold.
    NaN values are considered to be below the detection threshold.

    :param X: a 2D numpy array
    :param threshold: the detection threshold. If a scalar is provided, the same threshold is
        used for all columns. An array may be provided to use a different threshold for each
        column. If the threshold contains NaNs, the detection rate for the column will be set to zero.
    :return: a 1D with the detection rate of each column.

    """
    check_matrix_shape(X)

    n_rows, n_cols = X.shape

    if not isinstance(threshold, numpy.ndarray):
        threshold = numpy.ones(shape=(1, n_cols)) * threshold

    if threshold.size != n_cols:
        msg = "The threshold must be a scalar or a 1D float with size equal to the number of columns in X."
        raise ValueError(msg)

    threshold = threshold.reshape((1, threshold.size))

    non_detected_count = numpy.sum(numpy.isnan(X) | (X < threshold), axis=0)
    # set columns where threshold is nan to all undetected
    non_detected_count[numpy.isnan(threshold.flatten())] = n_rows

    return (n_rows - non_detected_count) / n_rows


def dratio(Xs: FloatArray, Xqc: FloatArray, robust: bool = False) -> FloatArray1D:
    r"""Compute the D-ratio metric for columns.

    The D-ratio is defined as the quotient between the standard deviation of QC data, or data that
    is expected to exhibit instrumental variation only and the standard deviation of sample data
    or data that presents `biological` variation.

    .. math::

        \textrm{D-Ratio} = \frac{S_{\textrm{QC}}}{S_{\textrm{sample}}}

    where :math:`S_{\textrm{sample}` is the sample standard deviation and :math:`S_{\textrm{QC}` is
    the QC standard deviation.

    NaN values in the sample or QC data will be ignored in the computation of the standard deviation.

    :param Xs: 2D array with sample data
    :param Xqc: 2D array with QC data
    :param robust: if set to ``True`` estimate the D-ratio using the median absolute deviation instead.
    :return: an 1D array with the D-ratio of each column. NaN values will be obtained if the sample standard
        deviation is zero. If the number of rows in `Xqc` or `Xs` is one, NaN values will also be obtained.

    """
    check_matrix_shape(Xs)
    check_matrix_shape(Xqc)
    return _sample_std(Xqc, robust=robust) / _sample_std(Xs, robust=robust)


def correlation(
    X: FloatArray,
    y: FloatArray1D,
    method: CorrelationMethod | str = CorrelationMethod.PEARSON,
) -> FloatArray1D:
    """Compute the correlation between the columns of a 2D array and a 1D array.

    :param X: the 2D array
    :param y: the 1D array
    :param method: ``"pearson"`` computes the Pearson's r coefficient. ``"spearman"``
        computes the spearman rank correlation coefficient.
    :return: the correlation coefficient between `y` and each `X` column.

    """
    if not isinstance(method, CorrelationMethod):
        method = CorrelationMethod(method)

    check_matrix_shape(X)

    if y.size != X.shape[0]:
        raise ValueError("The number of rows in `X` must match the length of `y`.")

    if y.size < 2:
        raise ValueError("At least a size of two is required to compute the correlation.")

    match method:
        case CorrelationMethod.PEARSON:
            return _corr_pearson(X, y)
        case CorrelationMethod.SPEARMAN:
            return _corr_spearman(X, y)
        case _ as never:
            assert_never(never)


def lod(X: FloatArray) -> FloatArray1D:
    """Estimates the limit of detection (LOD) of the data.

    :param X: a data matrix
    """
    check_matrix_shape(X)
    lod_ = numpy.nanmean(X, axis=0) + 3 * numpy.nanstd(X, axis=0, ddof=1)
    lod_[numpy.isnan(lod_)] = 0.0
    return lod_


def loq(X: FloatArray) -> FloatArray1D:
    """Estimates the limit of quantitation (LOQ) of the data.

    :param X: a data matrix
    """
    check_matrix_shape(X)
    lod_ = numpy.nanmean(X, axis=0) + 10 * numpy.nanstd(X, axis=0, ddof=1)
    lod_[numpy.isnan(lod_)] = 0.0
    return lod_


def pca(
    X: FloatArray,
    *,
    n_components=2,
    normalization=None,
    scaling=None,
    return_loadings=False,
    return_variance=False,
) -> tuple[FloatArray, FloatArray | None, FloatArray1D | None]:
    """Compute PCA on data.

    PCA requires that no NaN values

    :param X: the data array
    :param n_components: the number of Principal Components to compute.
    :param scaling: the scaling method applied to `X` columns. Refer to :py:class:`~tidyms2.core.enums.ScalingMethod`
        for the list of available scaling methods. If set to ``None``, no scaling is applied.
    :param normalization: One of the available normalization methods. Refer to
        :py:class:`~tidyms2.core.enums.NormalizationMethod` for the list of available normalization methods. If set
        to ``None``, do not perform row normalization.
    :param return_loadings: wether to return the PCA feature loadings.
    :pram return_variance: wether to return the PC variance.
    :return: A 2D array with PCA scores. If `return_loadings` is set to ``True`` also include a 2D array with
        the PCA loadings. If `return_pc_variance` is set to ``True`` also include a 1D array with PC variances.

    """
    check_matrix_shape(X)

    if isinstance(normalization, str):
        normalization = NormalizationMethod(normalization)

    if isinstance(scaling, str):
        scaling = ScalingMethod(scaling)

    if normalization is not None:
        X = normalize(X, normalization)

    if scaling is not None:
        X = scale(X, scaling)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    n_components = pca.n_components_

    if return_loadings:
        loadings = pca.components_.T * numpy.sqrt(pca.explained_variance_)
    else:
        loadings = None

    if return_variance:
        variance = pca.explained_variance_
    else:
        variance = None

    return scores, loadings, variance


def _sample_std(X: FloatArray, robust: bool = False) -> FloatArray1D:
    """Compute an estimation of the sample standard deviation."""
    if robust:
        return median_abs_deviation(X, axis=0, scale="normal", nan_policy="omit")  # type: ignore
    return numpy.nanstd(X, axis=0, ddof=1)


def _corr_pearson(X: FloatArray, y: FloatArray1D) -> FloatArray1D:
    Y = numpy.tile(numpy.reshape(y, (y.size, 1)), X.shape[1])
    return cast(FloatArray1D, pearsonr(Y, X, axis=0)[0])


def _corr_spearman(X: FloatArray, y: FloatArray1D) -> FloatArray1D:
    X = rankdata(X, axis=0)
    y = rankdata(y)
    return _corr_pearson(X, y)
