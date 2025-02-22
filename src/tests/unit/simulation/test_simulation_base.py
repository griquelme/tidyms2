from math import isclose

import pytest
from pydantic import ValidationError

from tidyms2.simulation.base import AbundanceSpec, InstrumentResponseSpec, MeasurementNoiseSpec


class TestAbundanceSpec:
    def test_sample_abundance_default_settings_return_constant_value(self):
        spec = AbundanceSpec()
        assert isclose(spec.mean, spec.sample_abundance())

    def test_sample_abundance_with_std_return_different_values(self):
        spec = AbundanceSpec(std=5.0)

        # first collect multiple values and check that they are different
        values = list()
        n = 10
        for _ in range(n):
            values.append(spec.sample_abundance())
        assert len(set(values)) > 1

        tol = 5 * spec.std  # p ~ 0.99999 of all values being within this tolerance
        assert isclose(sum(values) / n, spec.mean, abs_tol=tol)

    def test_sample_abundance_with_prevalence_return_mean_or_zero(self):
        spec = AbundanceSpec(prevalence=0.5)

        values = [spec.sample_abundance() for _ in range(30)]
        assert all(isclose(spec.mean, x) or isclose(0.0, x) for x in values)
        # check at least one zero (p ~ 1e-10 of all values being non zero)
        assert sum(isclose(x, 0.0) for x in values) >= 1

    def test_sample_abundance_with_prevalence_and_std(self):
        spec = AbundanceSpec(prevalence=0.5)

        n = 30
        values = [spec.sample_abundance() for _ in range(n)]
        non_zeros = [x for x in values if not isclose(x, 0.0)]

        # check at least one non_zero (p ~ 1e-10 of all values being non zero)
        assert len(non_zeros) < n

        # then check that non zero values follow gaussian dist
        tol = 5 * spec.std  # p ~ 0.9997 of all values being within this tolerance
        non_zero_values_mean = sum(non_zeros) / len(non_zeros)
        assert isclose(non_zero_values_mean, spec.mean, abs_tol=tol)


class TestMeasurementNoiseSpec:
    def test_sample_noise_no_snr_returns_zero(self):
        spec = MeasurementNoiseSpec()
        assert isclose(0.0, spec.sample_noise(100.0, 1.0))

    def test_snr_lower_than_min_snr_raises_error(self):
        with pytest.raises(ValidationError):
            MeasurementNoiseSpec(base_snr=5.0, min_snr=10.0)

    def test_compute_snr_no_base_snr_return_none(self):
        spec = MeasurementNoiseSpec(base_snr=None)
        assert spec.compute_snr(1.0) is None

    def test_compute_snr_greater_than_min_snr(self):
        spec = MeasurementNoiseSpec(base_snr=100.0, min_snr=10.0)
        pk = 0.20
        snr = spec.compute_snr(pk)
        assert snr is not None
        assert snr > spec.min_snr

    def test_compute_snr_lower_than_min_snr_returns_min_snr(self):
        spec = MeasurementNoiseSpec(base_snr=100.0, min_snr=10.0)
        pk = 0.01
        snr = spec.compute_snr(pk)
        assert snr is not None
        assert snr == spec.min_snr

    def test_sample_noise_default_configuration_returns_zero(self):
        spec = MeasurementNoiseSpec()
        for _ in range(10):
            assert isclose(0.0, spec.sample_noise(1000.0, 1.0))

    def test_sample_noise_is_within_expected_values(self):
        snr = 100.0
        spec = MeasurementNoiseSpec(base_snr=snr)
        signal = 1000.0
        pk = 1.0

        tol = 5 * signal / snr  # p ~ 0.999999 of noise being within this value

        noise = spec.sample_noise(signal, pk)
        assert isclose(noise, 0.0, abs_tol=tol)


class TestInstrumentResponseSpec:
    def test_get_interbatch_with_default_configuration_return_one(self):
        spec = InstrumentResponseSpec()
        for k in range(10):
            actual = spec.get_interbatch_factor(k)
            expected = 1.0
            assert isclose(actual, expected)

    def test_get_interbatch_factor_caches_values(self):
        spec = InstrumentResponseSpec(interbatch_variation=0.8)
        b11 = spec.get_interbatch_factor(1)
        b12 = spec.get_interbatch_factor(1)
        b21 = spec.get_interbatch_factor(2)
        b22 = spec.get_interbatch_factor(2)
        assert b11 == b12
        assert b21 == b22
        assert b11 != b21

    def test_get_sensitivity_loss_cond1(self):
        # max_sensitivity loss set to zero always returns the response factor
        spec = InstrumentResponseSpec()

        # small run order
        actual = spec.get_sensitivity_loss_factor(0)
        expected = 1.0
        assert isclose(actual, expected)

        # large run order
        actual = spec.get_sensitivity_loss_factor(100)
        expected = 1.0
        assert isclose(actual, expected)

    def test_get_sensitivity_loss_cond2(self):
        # max_sensitivity loss set to one, decay set to zero
        spec = InstrumentResponseSpec(max_sensitivity_loss=1.0)

        # zero run order return 1.0
        actual = spec.get_sensitivity_loss_factor(0)
        expected = 1.0
        assert isclose(actual, expected)

        # large run order return 1.0
        actual = spec.get_sensitivity_loss_factor(100)
        expected = 1.0
        assert isclose(actual, expected)

    def test_get_sensitivity_loss_cond3(self):
        # max_sensitivity loss set to zero, non zero decay
        spec = InstrumentResponseSpec(max_sensitivity_loss=0.0, sensitivity_decay=0.5)

        # small run order
        actual = spec.get_sensitivity_loss_factor(0)
        expected = 1.0
        assert isclose(actual, expected)

        # large run order
        actual = spec.get_sensitivity_loss_factor(100)
        expected = 1.0
        assert isclose(actual, expected)

    def test_get_sensitivity_loss_cond4(self):
        # max_sensitivity loss set to one, decay set to zero
        spec = InstrumentResponseSpec(max_sensitivity_loss=1.0, sensitivity_decay=0.5)

        # zero run order return 1.0
        actual = spec.get_sensitivity_loss_factor(0)
        expected = 1.0
        assert isclose(actual, expected)

        # large run should return a smaller value
        actual = spec.get_sensitivity_loss_factor(100)
        assert actual < 0.001
