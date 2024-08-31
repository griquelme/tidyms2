import numpy as np
import pydantic

from tidyms2.utils.numpy import FloatArray, IntArray


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
