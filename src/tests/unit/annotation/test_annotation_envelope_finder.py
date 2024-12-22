import pytest

from tidyms2.annotation import envelope_finder as ef
from tidyms2.annotation.annotation_data import AnnotationData
from tidyms2.annotation.config import AnnotatorParameters
from tidyms2.chem import DEFAULT_CONTEXT, PTABLE

from ..helpers import create_features_from_formula, create_sample

# Variable name convention
# m denote nominal mass values
# M denote exact mass values
# p denote abundance


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    return create_sample(tmp_path, 1)


@pytest.fixture(scope="module")
def formula_groups():
    formulas = {
        "cho": ["C27H34O9", "C62H120O6", "C59H114O6", "C62H120O6", "C56H42O10"],
        "chnops": ["C41H80NO8P", "C54H104O6", "C27H40O2", "C24H26O12", "C55H106O6"],
    }
    return formulas


class TestEnvelopeFinderComponents:
    element_groups = {"CHO": ["C", "H", "O"], "CHNOPS": ["C", "H", "N", "O", "P", "S"]}
    formula_groups = {
        "CHO": ["C27H34O9", "C62H120O6", "C59H114O6", "C62H120O6", "C56H42O10"],
        "CHNOPS": ["C41H80NO8P", "C54H104O6", "C27H40O2", "C24H26O12", "C55H106O6"],
    }

    @pytest.mark.parametrize("group", ["CHO", "CHNOPS"])
    def test__make_exact_mass_difference_bounds(self, group):
        # test bounds for different element combinations
        elements = [PTABLE.get_element(x) for x in self.element_groups[group]]
        bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
        # m and M are the bounds for each nominal mass increment
        for e in elements:
            m = [i.a - e.mmi.a for i in e.isotopes]
            M = [i.m - e.mmi.m for i in e.isotopes]
            for i, mi in zip(m[1:], M[1:]):
                m_min, m_max = bounds[i]
                assert m_min <= mi
                assert m_max >= mi

    @pytest.mark.parametrize("group", ["CHO", "CHNOPS"])
    def test__get_next_mz_search_interval_mz(self, group, sample):
        elements = [PTABLE.get_element(x) for x in self.element_groups[group]]
        dM_bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
        # test bounds for different formulas
        for f_str in self.formula_groups[group]:
            feature_list = create_features_from_formula(f_str, sample)
            length = len(feature_list)
            for k in range(1, length - 1):
                k_ft = feature_list[k]
                min_mz, max_mz = ef._get_next_mz_search_interval(feature_list[:k], dM_bounds, 1, 0.005)
                assert (min_mz < k_ft.mz) and (k_ft.mz < max_mz)

    @pytest.mark.parametrize("charge", list(range(1, 6)))
    def test_get_k_bounds_multiple_charges(self, charge, sample):
        formulas = self.formula_groups["CHNOPS"]
        elements = [PTABLE.get_element(x) for x in self.element_groups["CHNOPS"]]
        bounds = ef._make_exact_mass_difference_bounds(elements, 0.0)
        for f_str in formulas:
            features = create_features_from_formula(f"[{f_str}]{charge}+", sample)
            length = len(features)
            for k in range(1, length - 1):
                m_min, m_max = ef._get_next_mz_search_interval(features[:k], bounds, charge, 0.005)
                assert (m_min < features[k].mz) and (features[k].mz < m_max)

    @pytest.mark.parametrize("group,charge", [["CHO", 1], ["CHO", 2], ["CHNOPS", 1], ["CHNOPS", 2]])
    def test__find_envelopes(self, group, charge, sample):
        # test that the function works using as a list m/z values generated from
        # formulas.
        formulas = self.formula_groups[group]
        config = AnnotatorParameters(bounds={x: (0, 10) for x in group}, max_M=1000.0)
        finder = ef.EnvelopeFinder(config, DEFAULT_CONTEXT)
        for f_str in formulas:
            features = create_features_from_formula(f"[{f_str}]{charge}+", sample, n_isotopologues=config.max_length)
            data = AnnotationData(features)
            mmi = data.features[0]
            results = finder.find(data, mmi, charge)
            expected = features
            assert results[0] == expected

    @pytest.mark.parametrize("group", ["CHO", "CHNOPS"])
    def test__find_envelopes_no_charge(self, group, sample):
        # test that the function works using as a list m/z values generated from
        # formulas.
        config = AnnotatorParameters(bounds={x: (0, 10) for x in group}, max_M=1000.0)
        finder = ef.EnvelopeFinder(config, DEFAULT_CONTEXT)
        charge = 0
        for f_str in self.formula_groups[group]:
            features = create_features_from_formula(f_str, sample, config.max_length)
            data = AnnotationData(features)
            mmi = features[0]
            results = finder.find(data, mmi, charge)
            expected = features
            assert results[0] == expected

    @pytest.mark.parametrize("group", ["CHNOPS", "CHO"])
    def test_EnvelopeFinder(self, group, sample):
        formulas = self.formula_groups[group]
        config = AnnotatorParameters(bounds={x: (0, 10) for x in group}, max_M=1000.0)
        envelope_finder = ef.EnvelopeFinder(config, DEFAULT_CONTEXT)
        charge = 1
        for f_str in formulas:
            features = create_features_from_formula(f_str, sample, config.max_length)
            mmi = features[0]
            data = AnnotationData(features)
            results = envelope_finder.find(data, mmi, charge)
            expected = features
            assert len(results) == 1
            assert results[0] == expected
