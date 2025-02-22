import numpy as np
import pydantic
import pytest

from tidyms2.core.utils.numpy import FloatArray, IntArray, find_closest


class NumpyFloatModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    arr: FloatArray


class NumpyIntModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    arr: IntArray


class TestSerializeFloatArray:
    def test_empty_array(self):
        expected = np.array([], dtype=float)
        model = NumpyFloatModel(arr=expected)
        actual = NumpyFloatModel(**model.model_dump()).arr
        assert np.array_equal(actual, expected)

    def test_1D_array(self):
        expected = np.random.normal(size=100)
        model = NumpyFloatModel(arr=expected)
        actual = NumpyFloatModel(**model.model_dump()).arr
        assert np.array_equal(actual, expected)

    def test_2D_array(self):
        expected = np.random.normal(size=(100, 20))
        model = NumpyFloatModel(arr=expected)
        actual = NumpyFloatModel(**model.model_dump()).arr
        assert np.array_equal(actual, expected)


class TestSerializeIntArray:
    def test_empty_array(self):
        expected = np.array([], dtype=int)
        model = NumpyIntModel(arr=expected)
        actual = NumpyIntModel(**model.model_dump()).arr
        assert np.array_equal(actual, expected)

    def test_1D_array(self):
        expected = np.random.randint(low=0, high=100, size=100)
        model = NumpyIntModel(arr=expected)
        actual = NumpyIntModel(**model.model_dump()).arr
        assert np.array_equal(actual, expected)

    def test_2D_array(self):
        expected = np.random.randint(low=0, high=100, size=(100, 25))
        model = NumpyIntModel(arr=expected)
        actual = NumpyIntModel(**model.model_dump()).arr
        assert np.array_equal(actual, expected)


class TestFindClosest:
    def test_find_closest_left_border(self):
        x = np.arange(10)
        y = -1
        ind = find_closest(x, y)
        assert ind == 0

    def test_find_closest_right_border(self):
        x = np.arange(10)
        y = 10
        ind = find_closest(x, y)
        assert ind == (x.size - 1)

    def test_find_closest_middle(self):
        x = np.arange(10)
        y = 4.6
        ind = find_closest(x, y)
        assert ind == 5

    def test_find_closest_empty_x(self):
        x = np.array([])
        y = 10
        with pytest.raises(ValueError):
            find_closest(x, y)

    def test_find_closest_empty_y(self):
        x = np.arange(10)
        y = np.array([])
        res = find_closest(x, y)
        assert res.size == 0

    def test_find_closest_multiple_values(self):
        x = np.arange(100)
        y = np.array([-10, 4.6, 67.1, 101])
        ind = np.array([0, 5, 67, 99], dtype=int)
        result = find_closest(x, y)
        assert np.array_equal(result, ind)

    def test_find_closest_unsorted_single_value(self):
        n = 100
        x = np.random.normal(size=n)
        # select three random points
        random_index = np.random.choice(n)
        xq = x[random_index]
        closest_index = find_closest(x, xq, is_sorted=False)
        assert np.equal(random_index, closest_index).all()

    def test_find_closest_unsorted_multiple_values(self):
        n = 100
        x = np.random.normal(size=n)
        # select three random points
        random_index = np.random.choice(n, 3)
        xq = x[random_index]
        closest_index = find_closest(x, xq, is_sorted=False)
        assert np.equal(random_index, closest_index).all()
