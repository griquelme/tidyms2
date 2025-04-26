from pathlib import Path
from random import seed

import pytest
from numpy import allclose, nan
from pydantic import ValidationError

from tidyms2 import operators
from tidyms2.core.enums import AggregationMethod
from tidyms2.core.exceptions import EmptyDataMatrix, ProcessStatusError
from tidyms2.core.models import Sample
from tidyms2.simulation.lcms import SimulatedLCMSAdductSpec, simulate_data_matrix


class TestBlankCorrector:
    @pytest.fixture
    def samples(self):
        sample_list = list()
        for k in range(10):
            id_ = f"sample-{k}"
            path = Path.cwd() / f"{id_}.mzML"
            s = Sample(id=id_, path=path)
            s.meta.order = k
            s.meta.group = "group-a" if k % 2 else "group-b"
            s.meta.type = "blank" if k % 2 else "sample"
            sample_list.append(s)
        return sample_list

    @pytest.fixture
    def adducts(self):
        adduct1 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C54H104O6]+",
                "abundance": {
                    "group-a": {"mean": 100.0},
                    "group-b": {"mean": 1000.0},
                },
            },
        )

        adduct2 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C27H40O2]+",
                "abundance": {
                    "group-a": {"mean": 100.0},
                    "group-b": {"mean": 1000.0},
                },
            }
        )
        return [adduct1, adduct2]

    def test_default(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        op = operators.BlankCorrector()
        op.apply(data)

        sample_ids = data.query.filter(type="sample").fetch_sample_ids()[0][1]
        blank_sample_ids = data.query.filter(type="blank").fetch_sample_ids()[0][1]
        assert allclose(data.get_data(sample_ids=sample_ids), 900.0)
        assert allclose(data.get_data(sample_ids=blank_sample_ids), 100.0)

    def test_with_blank_groups(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        op = operators.BlankCorrector(blank_groups=["group-a"])
        op.apply(data)

        sample_ids = data.query.filter(type="sample").fetch_sample_ids()[0][1]
        blank_sample_ids = data.query.filter(type="blank").fetch_sample_ids()[0][1]
        assert allclose(data.get_data(sample_ids=sample_ids), 900.0)
        assert allclose(data.get_data(sample_ids=blank_sample_ids), 100.0)

    def test_with_excluded_samples(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        excluded = data.samples[2].id
        row_before = data.get_rows(excluded)[0]
        op = operators.BlankCorrector(exclude_samples=[excluded])
        op.apply(data)

        row_after = data.get_rows(excluded)[0]
        assert allclose(row_before.data, row_after.data)

    def test_with_excluded_features(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        col_before = data.get_columns(1)[0]
        op = operators.BlankCorrector(exclude_features=[1])
        op.apply(data)
        col_after = data.get_columns(1)[0]
        assert allclose(col_before.data, col_after.data)

    def test_with_excluded_features_and_samples(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        excluded_sample = data.samples[2].id
        col_before = data.get_columns(1)[0]
        row_before = data.get_rows(excluded_sample)[0]
        op = operators.BlankCorrector(exclude_features=[1], exclude_samples=[excluded_sample])
        op.apply(data)
        col_after = data.get_columns(1)[0]
        row_after = data.get_rows(excluded_sample)[0]
        assert allclose(col_before.data, col_after.data)
        assert allclose(row_before.data, row_after.data)

    def test_default_correct_blanks(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        op = operators.BlankCorrector(apply_to_blanks=True)
        op.apply(data)

        sample_ids = data.query.filter(type="sample").fetch_sample_ids()[0][1]
        blank_sample_ids = data.query.filter(type="blank").fetch_sample_ids()[0][1]
        assert allclose(data.get_data(sample_ids=sample_ids), 900.0)
        assert allclose(data.get_data(sample_ids=blank_sample_ids), 0.0)

    def test_default_no_blank_raise_error(self, adducts, samples):
        for sample in samples:
            sample.meta.type = "sample"
        data = simulate_data_matrix(adducts, samples)
        op = operators.BlankCorrector()
        with pytest.raises(ValueError):
            op.apply(data)


class TestSampleFilter:
    @pytest.fixture
    def samples(self):
        sample_list = list()
        for k in range(10):
            id_ = f"sample-{k}"
            path = Path.cwd() / f"{id_}.mzML"
            s = Sample(id=id_, path=path)
            s.meta.order = k
            s.meta.group = "group-a" if k % 2 else "group-b"
            s.meta.type = "qc" if k % 2 else "sample"
            sample_list.append(s)
        return sample_list

    @pytest.fixture
    def adducts(self):
        adduct1 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C54H104O6]+",
                "abundance": {
                    "group-a": {"mean": 1000.0},
                    "group-b": {"mean": 1000.0, "std": 1000.0},
                },
            },
        )

        adduct2 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C27H40O2]+",
                "abundance": {
                    "group-a": {"mean": 1000.0, "std": 1000.0},
                    "group-b": {"mean": 1000.0},
                },
            }
        )
        return [adduct1, adduct2]

    def test_remove_sample_group(self, samples, adducts):
        data = simulate_data_matrix(adducts, samples)
        op = operators.SampleFilter(group="group-a")  # type: ignore
        op.apply(data)
        assert data.get_n_samples()
        assert all(sample.meta.group == "group-b" for sample in data.samples)

    def test_remove_sample_type(self, samples, adducts):
        data = simulate_data_matrix(adducts, samples)
        op = operators.SampleFilter(type="qc")  # type: ignore
        op.apply(data)
        assert data.get_n_samples()
        assert all(sample.meta.type == "sample" for sample in data.samples)


class TestFeatureFilter:
    @pytest.fixture
    def samples(self):
        sample_list = list()
        for k in range(10):
            id_ = f"sample-{k}"
            path = Path.cwd() / f"{id_}.mzML"
            s = Sample(id=id_, path=path)
            s.meta.order = k
            s.meta.group = "group-a" if k % 2 else "group-b"
            s.meta.type = "qc" if k % 2 else "sample"
            sample_list.append(s)
        return sample_list

    @pytest.fixture
    def adducts(self):
        adduct1 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C54H104O6]+",
                "rt": {"mean": 50.0},
                "abundance": {
                    "group-a": {"mean": 1000.0},
                    "group-b": {"mean": 1000.0, "std": 1000.0},
                },
            },
        )

        adduct2 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C27H40O2]+",
                "rt": {"mean": 100.0},
                "abundance": {
                    "group-a": {"mean": 1000.0, "std": 1000.0},
                    "group-b": {"mean": 1000.0},
                },
            }
        )
        return [adduct1, adduct2]

    def test_filter_is_not_a_pair_raises_error(self):
        with pytest.raises(ValidationError):
            operators.FeatureFilter(rt=125)  # type: ignore

    def test_remove_features_using_rt(self, samples, adducts):
        data = simulate_data_matrix(adducts, samples)
        op = operators.FeatureFilter(rt=(75, 125))  # type: ignore
        op.apply(data)
        assert data.get_n_features() == 1
        assert data.has_feature(1)


class TestCVFilter:
    @pytest.fixture
    def samples(self):
        sample_list = list()
        for k in range(10):
            id_ = f"sample-{k}"
            path = Path.cwd() / f"{id_}.mzML"
            s = Sample(id=id_, path=path)
            s.meta.order = k
            s.meta.group = "group-a" if k % 2 else "group-b"
            s.meta.type = "qc" if k % 2 else "sample"
            sample_list.append(s)
        return sample_list

    @pytest.fixture
    def adducts(self):
        adduct1 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C54H104O6]+",
                "abundance": {
                    "group-a": {"mean": 1000.0},
                    "group-b": {"mean": 1000.0, "std": 200.0},
                },
            },
        )

        adduct2 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C27H40O2]+",
                "abundance": {
                    "group-a": {"mean": 1000.0, "std": 200.0},
                    "group-b": {"mean": 1000.0},
                },
            }
        )
        return [adduct1, adduct2]

    def test_lb_greater_than_ub_raises_error(self):
        with pytest.raises(ValidationError):
            operators.CVFilter(lb=0.5, ub=0.25)

    def test_default(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        # only adduct 2 should be removed as it has CV ~ 1.0 in the QC type
        rm_group = data.features[1].group
        assert data.get_n_features() == 2
        assert data.has_feature(rm_group)
        op = operators.CVFilter(ub=0.01)
        op.apply(data)
        assert data.get_n_features() == 1
        assert not data.has_feature(rm_group)

    def test_default_no_qc_type_raise_error(self, adducts, samples):
        for sample in samples:
            sample.meta.type = "sample"
        data = simulate_data_matrix(adducts, samples)
        op = operators.CVFilter()

        with pytest.raises(ValueError):
            op.apply(data)

    def test_using_filter(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        # only adduct 1 should be removed as it has CV ~ 1.0 in the group-b
        rm_group = data.features[0].group
        assert data.get_n_features() == 2
        assert data.has_feature(rm_group)
        op = operators.CVFilter(ub=0.01, filter={"group": "group-b"})
        op.apply(data)
        assert data.get_n_features() == 1
        assert not data.has_feature(rm_group)

    def test_group_by(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        # using max aggregation should remove both features as both have a
        # a group with CV ~ 1.0
        op = operators.CVFilter(ub=0.01, group_by=["group"], aggregation=AggregationMethod.MIN)
        op.apply(data)
        assert data.get_n_features() == len(adducts)

    def test_aggregation(self, adducts, samples):
        seed(1000)
        data = simulate_data_matrix(adducts, samples)
        # using max aggregation should remove both features as both have a
        # a group with CV ~ 1.0
        op = operators.CVFilter(ub=0.01, group_by=["group"], aggregation=AggregationMethod.MAX)
        with pytest.raises(EmptyDataMatrix):
            op.apply(data)


class TestDRatioFilter:
    @pytest.fixture
    def samples(self):
        sample_list = list()
        for k in range(20):
            id_ = f"sample-{k}"
            path = Path.cwd() / f"{id_}.mzML"
            s = Sample(id=id_, path=path)
            s.meta.order = k
            s.meta.group = "group-a" if k % 2 else "group-b"
            s.meta.type = "qc" if k % 2 else "sample"
            sample_list.append(s)
        return sample_list

    @pytest.fixture
    def adducts(self):
        adduct1 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C54H104O6]+",
                "abundance": {
                    "group-a": {"mean": 2000.0, "std": 100.0},  # QC
                    "group-b": {"mean": 2000.0, "std": 10.0},  # sample
                },
            },
        )

        adduct2 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C27H40O2]+",
                "abundance": {
                    "group-a": {"mean": 2000.0, "std": 10.0},  # QC
                    "group-b": {"mean": 2000.0, "std": 100.0},  # sample
                },
            }
        )
        return [adduct1, adduct2]

    def test_using_filter_param_raises_error(self):
        with pytest.raises(ValidationError):
            operators.DRatioFilter(filter={"type": "sample"})

    def test_using_group_by_param_raises_error(self):
        with pytest.raises(ValidationError):
            operators.DRatioFilter(group_by=["type"])

    def test_default(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        # only adduct 1 should be removed as its D-ratio is > 1.0
        rm_group = data.features[0].group
        assert data.get_n_features() == 2
        assert data.has_feature(rm_group)
        op = operators.DRatioFilter()
        op.apply(data)
        assert data.get_n_features() == 1
        assert not data.has_feature(rm_group)

    def test_default_no_qc_type_raise_error(self, adducts, samples):
        for sample in samples:
            sample.meta.type = "sample"
        data = simulate_data_matrix(adducts, samples)
        op = operators.DRatioFilter()

        with pytest.raises(ValueError):
            op.apply(data)

    def test_default_no_sample_type_raise_error(self, adducts, samples):
        for sample in samples:
            sample.meta.type = "qc"
        data = simulate_data_matrix(adducts, samples)
        op = operators.DRatioFilter()

        with pytest.raises(ValueError):
            op.apply(data)


class TestDetectionRateFilter:
    @pytest.fixture
    def samples(self):
        sample_list = list()
        for k in range(10):
            id_ = f"sample-{k}"
            path = Path.cwd() / f"{id_}.mzML"
            s = Sample(id=id_, path=path)
            s.meta.order = k
            s.meta.group = "group-a" if k % 2 else "group-b"
            s.meta.type = "blank" if k % 2 else "sample"
            sample_list.append(s)
        return sample_list

    @pytest.fixture
    def adducts(self):
        adduct1 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C54H104O6]+",
                "abundance": {
                    "group-a": {"mean": 100.0},
                    "group-b": {"mean": 1000.0, "prevalence": 0.5},
                },
            },
        )

        adduct2 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C27H40O2]+",
                "abundance": {
                    "group-a": {"mean": 100.0},
                    "group-b": {"prevalence": 1.0},
                },
            }
        )
        return [adduct1, adduct2]

    def test_default(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        # only adduct 1 should be removed as it has DR ~ 0.5 in the sample sample type
        rm_group = data.features[0].group
        assert data.get_n_features() == 2
        assert data.has_feature(rm_group)
        op = operators.DetectionRateFilter()
        op.apply(data)
        assert data.get_n_features() == 1
        assert not data.has_feature(rm_group)

    def test_default_no_blank(self, adducts, samples):
        for sample in samples:
            sample.meta.type = "sample"
        data = simulate_data_matrix(adducts, samples)
        op = operators.DetectionRateFilter()
        op.apply(data)
        assert data.get_n_features() == 1


class TestCorrelationFilter:
    @pytest.fixture
    def samples(self):
        sample_list = list()
        for k in range(10):
            id_ = f"sample-{k}"
            path = Path.cwd() / f"{id_}.mzML"
            s = Sample(id=id_, path=path)
            s.meta.order = k
            s.meta.group = "group-a" if k % 2 else "group-b"
            s.meta.type = "blank" if k % 2 else "sample"
            sample_list.append(s)
        return sample_list

    @pytest.fixture
    def adducts(self):
        adduct1 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C54H104O6]+",
                "abundance": {
                    "group-a": {"mean": 100.0, "std": 1.0},
                    "group-b": {"mean": 100.0, "std": 1.0},
                },
            },
        )

        adduct2 = SimulatedLCMSAdductSpec.model_validate(
            {
                "formula": "[C27H40O2]+",
                "abundance": {
                    "group-a": {"mean": 100.0, "std": 1.0},
                    "group-b": {"mean": 100.0, "std": 1.0},
                },
            }
        )
        return [adduct1, adduct2]

    def test_data_with_missing_values_raise_error(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        data_with_nan = data.get_data()
        data_with_nan[1] = nan
        data.set_data(data_with_nan)
        op = operators.CorrelationFilter(field="order", lb=1, ub=2)
        with pytest.raises(ProcessStatusError):
            op.apply(data)

    def test_default(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        # should remove both features as max corr is 1.0
        op = operators.CorrelationFilter(field="order", lb=1, ub=2)
        with pytest.raises(EmptyDataMatrix):
            op.apply(data)

    def test_default_features_with_correlation(self, adducts, samples):
        data = simulate_data_matrix(adducts, samples)
        order = data.list_sample_field("order")
        data_with_corr = data.get_data()
        data_with_corr[:, 0] = order
        data_with_corr[:, 1] = -1 * data_with_corr[:, 0]
        data.set_data(data_with_corr)

        op = operators.CorrelationFilter(field="order", lb=0.5, ub=1.0)
        op.apply(data)

        assert data.has_feature(0)
        assert not data.has_feature(1)
