from typing import cast

import numpy as np
import pytest
from scipy.stats import ConstantInputWarning, median_abs_deviation

from tidyms2.core.enums import CorrelationMethod
from tidyms2.core.utils import metrics
from tidyms2.core.utils.numpy import FloatArray


class TestCV:
    def test_non_2d_array_raise_error(self):
        X = np.arange(5)
        with pytest.raises(ValueError):
            metrics.cv(X)

    def test_2d_empty_arrays_raise_error(self):
        X = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            metrics.cv(X)

    def test_2d_array_single_column(self):
        X = np.array([[1], [2], [3]])
        actual = metrics.cv(X, robust=False)
        expected = np.array([np.std(X, ddof=1)]) / np.array([np.mean(X)])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_single_column_single_value_returns_nan(self):
        X = np.array([[1]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.cv(X, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_zeros_column_return_nan(self):
        X = np.zeros(shape=(3, 1))
        with pytest.warns(RuntimeWarning):
            actual = metrics.cv(X, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_constant_value_return_zero(self):
        X = np.ones(shape=(3, 1))
        actual = metrics.cv(X, robust=False)

        assert actual.shape == (1,)
        assert np.allclose(actual, 0.0)

    def test_single_column_two_non_nans_ok(self):
        X = np.array([[1.0], [np.nan], [1.0]])
        actual = metrics.cv(X, robust=False)

        assert actual.shape == (1,)
        assert np.allclose(actual, 0.0)

    def test_single_column_one_non_nan_return_nan(self):
        X = np.array([[1.0], [np.nan], [np.nan]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.cv(X, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_all_nans_return_nan(self):
        X = np.array([[np.nan], [np.nan], [np.nan]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.cv(X, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_multiple_column_ok(self):
        _, n_cols = shape = (4, 5)
        X = np.ones(shape=shape)
        actual = metrics.cv(X, robust=False)

        assert actual.shape == (n_cols,)
        assert np.allclose(actual, 0.0)


class TestCVRobust:
    def test_non_2d_array_raise_error(self):
        X = np.arange(5)
        with pytest.raises(ValueError):
            metrics.cv(X, robust=True)

    def test_2d_empty_arrays_raise_error(self):
        X = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            metrics.cv(X, robust=True)

    def test_2d_array_single_column(self):
        X = np.array([[1], [2], [3]])
        actual = metrics.cv(X, robust=True)
        s = median_abs_deviation(X, scale="normal")  # type: ignore
        expected = np.array([s]) / np.array([np.median(X)])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_2d_array_single_column_single_value_returns_zero(self):
        X = np.array([[1.0]])
        actual = metrics.cv(X, robust=True)
        expected = np.array([0.0])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_single_zeros_column_return_nan(self):
        X = np.zeros(shape=(3, 1))
        with pytest.warns(RuntimeWarning):
            actual = metrics.cv(X, robust=True)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_constant_value_return_zero(self):
        X = np.ones(shape=(3, 1))
        actual = metrics.cv(X, robust=True)

        assert actual.shape == (1,)
        assert np.allclose(actual, 0.0)

    def test_single_column_with_nans_ignore_nans(self):
        X = np.array([[1.0], [np.nan], [1.0]])
        actual = metrics.cv(X, robust=True)

        assert actual.shape == (1,)
        assert np.allclose(actual, 0.0)

    def test_single_column_all_nans_return_nan(self):
        X = np.array([[np.nan], [np.nan], [np.nan]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.cv(X, robust=True)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_multiple_column_ok(self):
        _, n_cols = shape = (4, 5)
        X = np.ones(shape=shape)
        actual = metrics.cv(X, robust=True)

        assert actual.shape == (n_cols,)
        assert np.allclose(actual, 0.0)


class TestDetectionRate:
    def test_non_2D_array_raises_error(self):
        X = np.arange(5)
        with pytest.raises(ValueError):
            metrics.detection_rate(X)

    def test_2d_empty_arrays_raise_error(self):
        X = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            metrics.detection_rate(X)

    def test_single_column(self):
        X = np.ones(shape=(3, 1), dtype=float)
        actual = metrics.detection_rate(X)
        expected = np.ones(shape=(1,))

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_single_column_nans_are_counted_as_undetected(self):
        X = np.array([[1], [np.nan], [2], [3]], dtype=float)
        actual = metrics.detection_rate(X)
        expected = np.array([0.75])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_single_column_values_below_threshold_are_counted_as_undetected(self):
        X = np.array([[1], [np.nan], [2], [3]], dtype=float)
        actual = metrics.detection_rate(X, threshold=1.5)
        expected = np.array([0.5])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_single_column_values_below_equal_to_threshold_are_counted_as_detected(self):
        X = np.array([[1], [np.nan], [2], [3]], dtype=float)
        actual = metrics.detection_rate(X, threshold=1.0)
        expected = np.array([0.75])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_single_column_nan_threshold_returns_zero(self):
        X = np.array([[1], [np.nan], [2], [3]], dtype=float)
        actual = metrics.detection_rate(X, threshold=np.nan)
        expected = np.array([0.0])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_multiple_columns_scalar_threshold(self):
        X = np.array([[1.0, 2.0], [np.nan, 2.0], [2.0, 2.0], [3.0, 2.0]], dtype=float)
        actual = metrics.detection_rate(X, threshold=1.0)
        expected = np.array([0.75, 1.0])

        assert actual.shape == (2,)
        assert np.allclose(actual, expected)

    def test_multiple_columns_array_threshold(self):
        X = np.array([[1.0, 2.0], [np.nan, 2.0], [2.0, 2.0], [3.0, 2.0]], dtype=float)
        threshold = np.array([1.0, np.nan])
        actual = metrics.detection_rate(X, threshold=threshold)
        expected = np.array([0.75, 0.0])

        assert actual.shape == (2,)
        assert np.allclose(actual, expected)


class TestDRatio:
    def test_non_2d_x_sample_raise_error(self):
        Xs = np.arange(5)
        Xqc = np.ones(shape=(2, 2), dtype=float)
        with pytest.raises(ValueError):
            metrics.dratio(Xs, Xqc)

    def test_non_2d_x_qc_raise_error(self):
        Xs = np.ones(shape=(2, 2), dtype=float)
        Xqc = np.arange(5)
        with pytest.raises(ValueError):
            metrics.dratio(Xs, Xqc)

    def test_2d_empty_x_sample_raise_error(self):
        Xs = np.random.normal(size=(2, 0))
        Xqc = np.ones(shape=(2, 2), dtype=float)
        with pytest.raises(ValueError):
            metrics.dratio(Xs, Xqc)

    def test_2d_empty_x_qc_raise_error(self):
        Xs = np.ones(shape=(2, 2), dtype=float)
        Xqc = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            metrics.dratio(Xs, Xqc)

    def test_single_column_single_row_x_sample_return_nan(self):
        Xs = np.array([[1]])
        Xqc = np.array([[1], [2], [3]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.dratio(Xs, Xqc, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_single_row_x_qc_return_nan(self):
        Xs = np.array([[1], [2], [3]])
        Xqc = np.array([[1]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.dratio(Xs, Xqc, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_one_non_nan_in_x_sample_return_nan(self):
        Xs = np.array([[1], [np.nan], [np.nan]])
        Xqc = np.array([[1], [np.nan], [3]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.dratio(Xs, Xqc, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_one_non_nan_in_x_qc_return_nan(self):
        Xs = np.array([[1], [np.nan], [3]])
        Xqc = np.array([[1], [np.nan], [np.nan]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.dratio(Xs, Xqc, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_two_non_nans_in_x_sample_ok(self):
        Xs = np.array([[1], [np.nan], [3]])
        Xqc = np.array([[1], [np.nan], [3]])
        actual = metrics.dratio(Xs, Xqc, robust=False)
        expected = np.array([1.0])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_constant_values_in_x_qc_returns_zero(self):
        Xs = np.array([[1], [2.0], [3]])
        Xqc = np.array([[1.0], [1.0], [1.0]])
        actual = metrics.dratio(Xs, Xqc, robust=False)
        expected = np.array([0.0])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_constant_values_in_x_sample_returns_nan(self):
        Xs = np.array([[1.0], [1.0], [1.0]])
        Xqc = np.array([[1], [2.0], [3]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.dratio(Xs, Xqc, robust=False)

        assert actual.shape == (1,)
        assert np.all(np.isinf(actual))

    def test_multiple_columns_ok(self):
        Xs = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        Xqc = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        actual = metrics.dratio(Xs, Xqc, robust=False)
        expected = np.array([1.0, 0.0])

        assert actual.shape == (2,)
        assert np.allclose(actual, expected)


class TestDRatioRobust:
    def test_non_2d_x_sample_raise_error(self):
        Xs = np.arange(5)
        Xqc = np.ones(shape=(2, 2), dtype=float)
        with pytest.raises(ValueError):
            metrics.dratio(Xs, Xqc, robust=True)

    def test_non_2d_x_qc_raise_error(self):
        Xs = np.ones(shape=(2, 2), dtype=float)
        Xqc = np.arange(5)
        with pytest.raises(ValueError):
            metrics.dratio(Xs, Xqc, robust=True)

    def test_2d_empty_x_sample_raise_error(self):
        Xs = np.random.normal(size=(2, 0))
        Xqc = np.ones(shape=(2, 2), dtype=float)
        with pytest.raises(ValueError):
            metrics.dratio(Xs, Xqc, robust=True)

    def test_2d_empty_x_qc_raise_error(self):
        Xs = np.ones(shape=(2, 2), dtype=float)
        Xqc = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            metrics.dratio(Xs, Xqc, robust=True)

    def test_single_column_single_row_x_sample_return_inf(self):
        Xs = np.array([[1]])
        Xqc = np.array([[1], [2], [3]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.dratio(Xs, Xqc, robust=True)

        assert actual.shape == (1,)
        assert np.all(np.isinf(actual))

    def test_single_column_single_row_x_qc_return_nan(self):
        Xs = np.array([[1], [2], [3]])
        Xqc = np.array([[1]])
        actual = metrics.dratio(Xs, Xqc, robust=True)

        expected = np.zeros(shape=(1,), dtype=float)

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_single_column_one_non_nan_in_x_sample_ok(self):
        Xs = np.array([[1], [np.nan], [np.nan]])
        Xqc = np.array([[1], [np.nan], [3]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.dratio(Xs, Xqc, robust=True)

        assert actual.shape == (1,)
        assert np.all(np.isinf(actual))

    def test_single_column_one_non_nan_in_x_qc_return_zero(self):
        Xs = np.array([[1], [np.nan], [3]])
        Xqc = np.array([[1], [np.nan], [np.nan]])
        actual = metrics.dratio(Xs, Xqc, robust=True)

        expected = np.zeros(shape=(1,), dtype=float)
        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_single_column_two_non_nans_in_x_sample_ok(self):
        Xs = np.array([[1], [np.nan], [3]])
        Xqc = np.array([[1], [np.nan], [3]])
        actual = metrics.dratio(Xs, Xqc, robust=True)
        expected = np.array([1.0])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_constant_values_in_x_qc_returns_zero(self):
        Xs = np.array([[1], [2.0], [3]])
        Xqc = np.array([[1.0], [1.0], [1.0]])
        actual = metrics.dratio(Xs, Xqc, robust=True)
        expected = np.array([0.0])

        assert actual.shape == (1,)
        assert np.allclose(actual, expected)

    def test_constant_values_in_x_sample_returns_nan(self):
        Xs = np.array([[1.0], [1.0], [1.0]])
        Xqc = np.array([[1], [2.0], [3]])
        with pytest.warns(RuntimeWarning):
            actual = metrics.dratio(Xs, Xqc, robust=True)

        assert actual.shape == (1,)
        assert np.all(np.isinf(actual))

    def test_multiple_columns_ok(self):
        Xs = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        Xqc = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        actual = metrics.dratio(Xs, Xqc, robust=True)
        expected = np.array([1.0, 0.0])

        assert actual.shape == (2,)
        assert np.allclose(actual, expected)


class TestPCA:
    def test_return_only_scores(self):
        n_rows = 20
        n_cols = 10
        n_components = 2
        X = cast(FloatArray, np.random.normal(size=(n_rows, n_cols)))
        scores, loadings, variance = metrics.pca(X)
        assert scores.shape == (n_rows, n_components)
        assert loadings is None
        assert variance is None

    def test_return_loadings(self):
        n_rows = 20
        n_cols = 10
        n_components = 2
        X = np.random.normal(size=(n_rows, n_cols))
        scores, loadings, variance = metrics.pca(X, n_components=n_components, return_loadings=True)
        assert scores.shape == (n_rows, n_components)
        assert loadings is not None
        assert loadings.shape == (n_cols, n_components)
        assert variance is None

    def test_return_variance(self):
        n_rows = 20
        n_cols = 10
        n_components = 2

        X = np.random.normal(size=(n_rows, n_cols))
        scores, loadings, variance = metrics.pca(X, n_components=n_components, return_variance=True)

        assert scores.shape == (n_rows, n_components)
        assert loadings is None
        assert variance is not None
        assert variance.shape == (n_components,)

    def test_return_variance_and_return_loadings(self):
        n_rows = 20
        n_cols = 10
        n_components = 2
        X = np.random.normal(size=(n_rows, n_cols))

        scores, loadings, variance = metrics.pca(
            X,
            n_components=n_components,
            return_loadings=True,
            return_variance=True,
        )
        assert scores.shape == (n_rows, n_components)
        assert loadings is not None
        assert loadings.shape == (n_cols, n_components)
        assert variance is not None
        assert variance.shape == (n_components,)

    def test_with_scaling(self):
        n_rows = 20
        n_cols = 10
        n_components = 2
        X = np.random.normal(size=(n_rows, n_cols))

        scores, loadings, variance = metrics.pca(
            X,
            n_components=n_components,
            scaling="autoscaling",
            return_loadings=True,
            return_variance=True,
        )
        assert scores.shape == (n_rows, n_components)
        assert loadings is not None
        assert loadings.shape == (n_cols, n_components)
        assert variance is not None
        assert variance.shape == (n_components,)
        assert ~np.any(np.isnan(scores))

    def test_with_normalization(self):
        n_rows = 20
        n_cols = 10
        n_components = 2
        X = np.random.normal(size=(n_rows, n_cols))

        scores, loadings, variance = metrics.pca(
            X,
            n_components=n_components,
            normalization="sum",
            return_loadings=True,
            return_variance=True,
        )
        assert scores.shape == (n_rows, n_components)
        assert loadings is not None
        assert loadings.shape == (n_cols, n_components)
        assert variance is not None
        assert variance.shape == (n_components,)
        assert ~np.any(np.isnan(scores))

    def test_with_normalization_and_scaling(self):
        n_rows = 20
        n_cols = 10
        n_components = 2
        X = np.random.normal(size=(n_rows, n_cols))

        scores, loadings, variance = metrics.pca(
            X,
            n_components=n_components,
            normalization="sum",
            scaling="autoscaling",
            return_loadings=True,
            return_variance=True,
        )
        assert scores.shape == (n_rows, n_components)
        assert loadings is not None
        assert loadings.shape == (n_cols, n_components)
        assert variance is not None
        assert variance.shape == (n_components,)
        assert ~np.any(np.isnan(scores))

    def test_n_components_greater_than_rank_raises_error(self):
        n_rows = 20
        n_cols = 10
        X = np.random.normal(size=(n_rows, n_cols))

        with pytest.raises(ValueError):
            metrics.pca(X, n_components=n_rows)

    def test_matrix_with_nan_raises_error(self):
        n_rows = 20
        n_cols = 10
        n_components = 2
        X = np.random.normal(size=(n_rows, n_cols))
        X[1:1] = np.nan
        X[2:4] = np.nan

        with pytest.raises(ValueError):
            metrics.pca(X, n_components=n_components)


class TestCorrelationPearson:
    def test_single_row_raises_error(self):
        y = np.array([1.0])
        x = np.array([[1.0]])

        with pytest.raises(ValueError):
            metrics.correlation(x, y, method=CorrelationMethod.PEARSON)

    def test_non_matching_shapes_raise_error(self):
        y = np.array([1.0])
        x = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError):
            metrics.correlation(x, y, method=CorrelationMethod.PEARSON)

    def test_single_column(self):
        y = np.array([1.0, 2.0])
        x = np.array([[1.0], [2.0]])

        actual = metrics.correlation(x, y, method=CorrelationMethod.PEARSON)
        expected = np.array([1.0])
        assert np.allclose(actual, expected)

    def test_single_column_with_constant_values_return_nan(self):
        y = np.array([1.0, 1.0])
        x = np.array([[1.0], [1.0]])

        with pytest.warns(ConstantInputWarning):
            actual = metrics.correlation(x, y, method=CorrelationMethod.PEARSON)
        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_with_nan_returns_nan(self):
        y = np.array([1.0, 2.0])
        x = np.array([[1.0], [np.nan]])

        actual = metrics.correlation(x, y, method=CorrelationMethod.PEARSON)
        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_multiple_columns(self):
        y = np.array([1.0, 2.0])
        x = np.array([[1.0, -1.0, 1.0], [-1.0, 2.0, np.nan]])

        actual = metrics.correlation(x, y, method=CorrelationMethod.PEARSON)
        expected = np.array([-1.0, 1.0, np.nan])
        assert actual.shape == (3,)
        assert np.allclose(actual, expected, equal_nan=True)


class TestCorrelationSpearman:
    def test_single_row_raises_error(self):
        y = np.array([1.0])
        x = np.array([[1.0]])

        with pytest.raises(ValueError):
            metrics.correlation(x, y, method=CorrelationMethod.SPEARMAN)

    def test_non_matching_shapes_raise_error(self):
        y = np.array([1.0])
        x = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError):
            metrics.correlation(x, y, method=CorrelationMethod.SPEARMAN)

    def test_single_column(self):
        y = np.array([1.0, 2.0])
        x = np.array([[1.0], [2.0]])

        actual = metrics.correlation(x, y, method=CorrelationMethod.SPEARMAN)
        expected = np.array([1.0])
        assert np.allclose(actual, expected)

    def test_single_column_with_constant_values_return_nan(self):
        y = np.array([1.0, 1.0])
        x = np.array([[1.0], [1.0]])

        with pytest.warns(ConstantInputWarning):
            actual = metrics.correlation(x, y, method=CorrelationMethod.SPEARMAN)
        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_single_column_with_nan_returns_nan(self):
        y = np.array([1.0, 2.0])
        x = np.array([[1.0], [np.nan]])

        actual = metrics.correlation(x, y, method=CorrelationMethod.SPEARMAN)
        assert actual.shape == (1,)
        assert np.all(np.isnan(actual))

    def test_multiple_columns(self):
        y = np.array([1.0, 2.0])
        x = np.array([[1.0, -1.0, 1.0], [-1.0, 2.0, np.nan]])

        actual = metrics.correlation(x, y, method=CorrelationMethod.SPEARMAN)
        expected = np.array([-1.0, 1.0, np.nan])
        assert actual.shape == (3,)
        assert np.allclose(actual, expected, equal_nan=True)
