import numpy as np
import pytest

from tidyms2.core.enums import NormalizationMethod, ScalingMethod
from tidyms2.core.utils import transformation


class TestNormalizeSum:
    def test_non_2d_array_raises_error(self):
        X = np.arange(5)
        with pytest.raises(ValueError):
            transformation.normalize(X)

    def test_2d_empty_array_raise_error(self):
        X = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            transformation.normalize(X)

    def test_single_column_single_use_str_method(self):
        X = np.array([[10.0]])

        actual = transformation.normalize(X, method="sum")
        expected = np.array([[1.0]])

        assert np.allclose(actual, expected)

    def test_single_column_single_row_returns_one(self):
        X = np.array([[10.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.SUM)
        expected = np.array([[1.0]])

        assert np.allclose(actual, expected)

    def test_single_single_row_returns_one(self):
        X = np.array([[1.0, 2.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.SUM)
        assert X.shape == actual.shape
        row_sum = np.nansum(actual, axis=1, keepdims=True)
        assert row_sum.shape == (X.shape[0], 1)
        assert np.allclose(row_sum, 1.0)

    def test_single_single_row_with_nans_returns_one(self):
        X = np.array([[1.0, 2.0, np.nan]])

        actual = transformation.normalize(X, method=NormalizationMethod.SUM)
        assert X.shape == actual.shape
        row_sum = np.nansum(actual, axis=1, keepdims=True)
        assert row_sum.shape == (X.shape[0], 1)
        assert np.allclose(row_sum, 1.0)

    def test_single_single_row_with_all_nans_return_all_nan(self):
        X = np.array([[np.nan, np.nan, np.nan]])

        actual = transformation.normalize(X, method=NormalizationMethod.SUM)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_single_single_row_with_all_zeros_return_all_nan(self):
        X = np.array([[0.0, 0.0, 0.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.SUM)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_single_single_row_with_all_zeros_or_nan_return_all_nan(self):
        X = np.array([[0.0, np.nan, 0.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.SUM)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_multiple_rows(self):
        X = np.array([[0.0, np.nan, 0.0], [1.0, 1.0, 1.0], [1.0, np.nan, np.nan]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.SUM)
        assert X.shape == actual.shape

        actual_row_sum = np.nansum(actual, axis=1, keepdims=True)
        expected_row_sum = np.array([[0.0], [1.0], [1.0]])
        assert np.allclose(actual_row_sum, expected_row_sum)


class TestNormalizeMax:
    def test_single_column_single_row_returns_one(self):
        X = np.array([[10.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.MAX)
        expected = np.array([[1.0]])

        assert np.allclose(actual, expected)

    def test_single_single_row(self):
        X = np.array([[1.0, 2.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.MAX)
        expected = np.array([[0.5, 1.0]])
        assert np.allclose(actual, expected)

    def test_single_single_row_with_nans(self):
        X = np.array([[1.0, 2.0, np.nan]])

        actual = transformation.normalize(X, method=NormalizationMethod.MAX)
        expected = np.array([0.5, 1.0, np.nan])
        assert np.allclose(actual, expected, equal_nan=True)

    def test_single_single_row_with_all_nans_return_all_nan(self):
        X = np.array([[np.nan, np.nan, np.nan]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.MAX)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_single_single_row_with_all_zeros_return_all_nan(self):
        X = np.array([[0.0, 0.0, 0.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.MAX)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_single_single_row_with_all_zeros_or_nan_return_all_nan(self):
        X = np.array([[0.0, np.nan, 0.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.MAX)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_multiple_rows(self):
        X = np.array([[0.0, np.nan, 0.0], [1.0, 1.0, 2.0], [1.0, np.nan, np.nan]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.MAX)

        expected = np.array([[np.nan, np.nan, np.nan], [0.5, 0.5, 1.0], [1.0, np.nan, np.nan]])
        assert np.allclose(actual, expected, equal_nan=True)


class TestNormalizeEuclidean:
    def test_single_column_single_row_returns_one(self):
        X = np.array([[10.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.EUCLIDEAN)
        expected = np.array([[1.0]])

        assert np.allclose(actual, expected)

    def test_single_single_row(self):
        X = np.array([[3.0, 4.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.EUCLIDEAN)
        expected = np.array([[0.6, 0.8]])
        assert np.allclose(actual, expected)

    def test_single_single_row_with_nans(self):
        X = np.array([[3.0, 4.0, np.nan]])

        actual = transformation.normalize(X, method=NormalizationMethod.EUCLIDEAN)
        expected = np.array([0.6, 0.8, np.nan])
        assert np.allclose(actual, expected, equal_nan=True)

    def test_single_single_row_with_all_nans_return_all_nan(self):
        X = np.array([[np.nan, np.nan, np.nan]])

        actual = transformation.normalize(X, method=NormalizationMethod.EUCLIDEAN)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_single_single_row_with_all_zeros_return_all_nan(self):
        X = np.array([[0.0, 0.0, 0.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.EUCLIDEAN)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_single_single_row_with_all_zeros_or_nan_return_all_nan(self):
        X = np.array([[0.0, np.nan, 0.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.EUCLIDEAN)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_multiple_rows(self):
        X = np.array([[0.0, np.nan, 0.0], [2.0, 3.0, 6.0], [1.0, np.nan, np.nan]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.EUCLIDEAN)

        expected = np.array([[np.nan, np.nan, np.nan], [2.0 / 7, 3.0 / 7.0, 6.0 / 7.0], [1.0, np.nan, np.nan]])
        assert np.allclose(actual, expected, equal_nan=True)


class TestNormalizeFeature:
    def test_undefined_index_raises_error(self):
        X = np.array([[10.0]])

        with pytest.raises(ValueError):
            transformation.normalize(X, method=NormalizationMethod.FEATURE)

    def test_single_column_single_row_returns_one(self):
        X = np.array([[10.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.FEATURE, index=0)
        expected = np.array([[1.0]])

        assert np.allclose(actual, expected)

    def test_single_single_row(self):
        X = np.array([[1.0, 2.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.FEATURE, index=0)
        expected = np.array([[1.0, 2.0]])
        assert np.allclose(actual, expected)

    def test_single_single_row_with_nans(self):
        X = np.array([[1.0, 2.0, np.nan]])

        actual = transformation.normalize(X, method=NormalizationMethod.FEATURE, index=0)
        expected = np.array([1.0, 2.0, np.nan])
        assert np.allclose(actual, expected, equal_nan=True)

    def test_single_single_with_feature_nan_return_all_nan(self):
        X = np.array([[np.nan, 1.0, 2.0]])

        actual = transformation.normalize(X, method=NormalizationMethod.FEATURE, index=0)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual))

    def test_single_single_row_with_feature_zeros_return_all_nan(self):
        X = np.array([[0.0, 1.0, 1.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.normalize(X, method=NormalizationMethod.FEATURE, index=0)
        assert X.shape == actual.shape
        assert np.all(np.isnan(actual[:, 0]))
        assert np.all(np.isinf(actual[:, 1:]))

    def test_multiple_rows(self):
        X = np.array([[1.0, np.nan, 0.0], [1.0, 1.0, 2.0], [1.0, np.nan, np.nan]])

        actual = transformation.normalize(X, method=NormalizationMethod.FEATURE, index=0)

        expected = np.array([[1.0, np.nan, 0.0], [1.0, 1.0, 2.0], [1.0, np.nan, np.nan]])
        assert np.allclose(actual, expected, equal_nan=True)


class TestScaleAutoscaling:
    def test_non_2d_array_raises_error(self):
        X = np.arange(5)
        with pytest.raises(ValueError):
            transformation.scale(X, method=ScalingMethod.AUTOSCALING)

    def test_2d_empty_array_raise_error(self):
        X = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            transformation.scale(X, method=ScalingMethod.AUTOSCALING)

    def test_use_str_scaling_method(self):
        X = np.array([[2.0], [0.0]])

        actual = transformation.scale(X, method="autoscaling")
        expected = np.array([[1.0], [-1.0]])

        assert np.allclose(actual, expected)

    def test_single_column_single_row_returns_nan(self):
        X = np.array([[10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.AUTOSCALING)

        assert np.all(np.isnan(actual))

    def test_constant_single_column_returns_nan(self):
        X = np.array([[10.0], [10.0], [10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.AUTOSCALING)

        assert np.all(np.isnan(actual))

    def test_single_column_nans_are_ignored_nan(self):
        X = np.array([[2.0], [np.nan], [0.0]])

        actual = transformation.scale(X, method=ScalingMethod.AUTOSCALING)
        expected = np.array([[1.0], [np.nan], [-1.0]])

        assert np.allclose(actual, expected, equal_nan=True)

    def test_single_column_no_nans(self):
        X = np.array([[2.0], [0.0]])

        actual = transformation.scale(X, method=ScalingMethod.AUTOSCALING)
        expected = np.array([[1.0], [-1.0]])

        assert np.allclose(actual, expected)

    def test_multiple_columns_ok(self):
        X = np.array([[2.0, 10.0], [0.0, 10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.AUTOSCALING)
        expected = np.array([[1.0, np.nan], [-1.0, np.nan]])

        assert np.allclose(actual, expected, equal_nan=True)


class TestScaleRescaling:
    def test_non_2d_array_raises_error(self):
        X = np.arange(5)
        with pytest.raises(ValueError):
            transformation.scale(X, method=ScalingMethod.RESCALING)

    def test_2d_empty_array_raise_error(self):
        X = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            transformation.scale(X, method=ScalingMethod.RESCALING)

    def test_single_column_single_row_returns_nan(self):
        X = np.array([[10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.RESCALING)

        assert np.all(np.isnan(actual))

    def test_constant_single_column_returns_nan(self):
        X = np.array([[10.0], [10.0], [10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.AUTOSCALING)

        assert np.all(np.isnan(actual))

    def test_single_column_nans_are_ignored_nan(self):
        X = np.array([[2.0], [np.nan], [0.0]])

        actual = transformation.scale(X, method=ScalingMethod.RESCALING)
        expected = np.array([[1.0], [np.nan], [0.0]])

        assert np.allclose(actual, expected, equal_nan=True)

    def test_single_column_no_nans(self):
        X = np.array([[2.0], [0.0]])

        actual = transformation.scale(X, method=ScalingMethod.RESCALING)
        expected = np.array([[1.0], [0.0]])

        assert np.allclose(actual, expected)

    def test_multiple_columns_ok(self):
        X = np.array([[2.0, 10.0], [0.0, 10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.RESCALING)
        expected = np.array([[1.0, np.nan], [0.0, np.nan]])

        assert np.allclose(actual, expected, equal_nan=True)


class TestScalePareto:
    def test_non_2d_array_raises_error(self):
        X = np.arange(5)
        with pytest.raises(ValueError):
            transformation.scale(X, method=ScalingMethod.PARETO)

    def test_2d_empty_array_raise_error(self):
        X = np.random.normal(size=(2, 0))
        with pytest.raises(ValueError):
            transformation.scale(X, method=ScalingMethod.PARETO)

    def test_single_column_single_row_returns_nan(self):
        X = np.array([[10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.PARETO)

        assert np.all(np.isnan(actual))

    def test_constant_single_column_returns_nan(self):
        X = np.array([[10.0], [10.0], [10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.PARETO)

        assert np.all(np.isnan(actual))

    def test_single_column_nans_are_ignored_nan(self):
        X = np.array([[2.0], [np.nan], [0.0]])

        actual = transformation.scale(X, method=ScalingMethod.PARETO)
        expected = np.array([[1.0], [np.nan], [-1.0]])

        assert np.allclose(actual, expected, equal_nan=True)

    def test_single_column_no_nans(self):
        X = np.array([[2.0], [0.0]])

        actual = transformation.scale(X, method=ScalingMethod.PARETO)
        expected = np.array([[1.0], [-1.0]])

        assert np.allclose(actual, expected)

    def test_multiple_columns_ok(self):
        X = np.array([[2.0, 10.0], [0.0, 10.0]])

        with pytest.warns(RuntimeWarning):
            actual = transformation.scale(X, method=ScalingMethod.PARETO)
        expected = np.array([[1.0, np.nan], [-1.0, np.nan]])

        assert np.allclose(actual, expected, equal_nan=True)
