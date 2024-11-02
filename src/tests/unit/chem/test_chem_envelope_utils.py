from itertools import product

import numpy as np
import pytest

from tidyms2.chem import PTABLE
from tidyms2.chem.envelope_utils import HomoAtomicEnvelopeCache, combine_envelopes


class TestHomoAtomicEnvelopeCache:
    @pytest.fixture(scope="module")
    def cache(self):
        return HomoAtomicEnvelopeCache()

    @pytest.mark.parametrize("symbol,n,length", product(["2H", "31P"], [0, 1, 5], [1, 2, 5]))
    def test_get_envelope_from_isotope(self, cache: HomoAtomicEnvelopeCache, symbol: str, n: int, length: int):
        isotope = PTABLE.get_isotope(symbol)
        actual = cache.get_envelope(isotope, n, length)
        M_expected = np.zeros((1, length))
        M_expected[0, 0] = n * isotope.m
        p_expected = np.zeros((1, length))
        p_expected[0, 0] = 1.0
        assert np.array_equal(actual.M, M_expected)
        assert np.array_equal(actual.p, p_expected)

    @pytest.mark.parametrize("symbol,n,length", product(["23Na", "31P"], [0, 1, 5], [1, 2, 5]))
    def test_get_envelope_from_isotope_and_element_are_equal_for_elements_with_one_isotope(
        self, cache: HomoAtomicEnvelopeCache, symbol: str, n: int, length: int
    ):
        isotope = PTABLE.get_isotope(symbol)
        element = PTABLE.get_element(isotope.z)
        env_isotope = cache.get_envelope(isotope, n, length)
        env_element = cache.get_envelope(element, n, length)
        assert np.array_equal(env_isotope.M, env_element.M)
        assert np.array_equal(env_isotope.p, env_element.p)

    def test_create_envelope_array_from_zero_to_zero(self, cache: HomoAtomicEnvelopeCache):
        symbol = "P"
        element = PTABLE.get_element(symbol)
        actual = cache.make_envelope_array(element, 0, 0, 10)
        expected = cache.get_empty()
        assert np.array_equal(actual.M, expected.M)
        assert np.array_equal(actual.p, expected.p)

    def test_create_envelope_array(self, cache: HomoAtomicEnvelopeCache):
        symbol = "P"
        element = PTABLE.get_element(symbol)
        n_max = 5
        length = 10
        actual = cache.make_envelope_array(element, 0, n_max, length)
        for k in range(n_max):
            expected_row = cache.get_envelope(element, k, length)
            assert np.array_equal(actual.M[k], expected_row.M[0])
            assert np.array_equal(actual.p[k], expected_row.p[0])


@pytest.mark.parametrize("symbol", ["C", "S", "P"])
def test_combine_envelopes(symbol):
    element = PTABLE.get_element(symbol)
    cache = HomoAtomicEnvelopeCache()
    n1 = 5
    env1 = cache.get_envelope(element, n1, length=10)
    n2 = 3
    env2 = cache.get_envelope(element, n2, length=10)
    n = n1 + n2
    actual = combine_envelopes(env1, env2)
    expected = cache.get_envelope(element, n, length=10)
    assert np.allclose(actual.M, expected.M, atol=10e-5)
    assert np.allclose(actual.p, expected.p, atol=10e-5)
