from typing import Sequence

import pytest

from tidyms2.annotation import mmi_finder
from tidyms2.annotation.annotation_data import AnnotationData
from tidyms2.annotation.config import AnnotatorParameters
from tidyms2.chem import DEFAULT_CONTEXT, PTABLE
from tidyms2.core.models import Sample

from ..helpers import ConcreteFeature, ConcreteRoi, create_sample

BIN_SIZE = 100


class TestSelectTwoIsotopeElements:
    CHNOPS = [PTABLE.get_element(x) for x in "CHNOPS"]

    def test_dm_1_p0_greater_than_pi(self):
        expected = [PTABLE.get_element("C")]
        dm = 1
        res = mmi_finder._select_two_isotope_element(self.CHNOPS, dm)
        assert len(res) == len(expected)
        assert set(res) == set(expected)

    def test_dm_1_no_elements(self):
        elements = [PTABLE.get_element(x) for x in "OPS"]
        dm = 1
        res = mmi_finder._select_two_isotope_element(elements, dm)
        assert len(res) == 0

    def test_dm_1_p0_lower_than_pi(self):
        elements = [PTABLE.get_element(x) for x in ["B", "Li", "O", "P", "S"]]
        expected = [PTABLE.get_element(x) for x in ["B", "Li"]]
        dm = 1
        actual = mmi_finder._select_two_isotope_element(elements, dm)
        assert len(actual) == len(expected)
        assert set(actual) == set(expected)

    def test_dm_1_p0_lower_and_higher_than_pi(self):
        elements = [PTABLE.get_element(x) for x in ["C", "H", "B", "Li", "O", "P", "S"]]
        expected = [PTABLE.get_element(x) for x in ["C", "B", "Li"]]
        dm = 1
        actual = mmi_finder._select_two_isotope_element(elements, dm)
        assert len(actual) == len(expected)
        assert set(actual) == set(expected)

    def test_dm_2_p0_greater_than_pi(self):
        elements = [PTABLE.get_element(x) for x in ["Cl", "H", "N", "O", "P", "S"]]
        expected = [PTABLE.get_element(x) for x in ["Cl"]]
        dm = 2
        actual = mmi_finder._select_two_isotope_element(elements, dm)
        assert len(actual) == len(expected)
        assert set(actual) == set(expected)

    def test_dm_2_no_elements(self):
        elements = [PTABLE.get_element(x) for x in ["O", "P", "S"]]
        dm = 2
        res = mmi_finder._select_two_isotope_element(elements, dm)
        assert len(res) == 0

    def test_dm_2_p0_lower_than_pi(self):
        elements = [PTABLE.get_element(x) for x in ["In", "H", "O", "P", "S"]]
        expected = [PTABLE.get_element(x) for x in ["In"]]
        dm = 2
        res = mmi_finder._select_two_isotope_element(elements, dm)
        assert len(res) == len(expected)
        assert set(res) == set(expected)

    def test_dm_2_p0_lower_and_higher_than_pi(self):
        elements = [PTABLE.get_element(x) for x in ["Cl", "In", "Br", "O", "P", "S"]]
        expected = [PTABLE.get_element(x) for x in ["Br", "In"]]
        dm = 2
        res = mmi_finder._select_two_isotope_element(elements, dm)
        assert len(res) == len(expected)
        assert set(res) == set(expected)


class TestSelectMultipleIsotopeElements:
    def test_ok(self):
        elements = [PTABLE.get_element(x) for x in ["Cl", "H", "N", "O", "P", "S"]]
        expected = [PTABLE.get_element(x) for x in ["O", "S"]]
        res = mmi_finder._select_multiple_isotope_elements(elements)
        assert len(res) == len(expected)
        assert set(res) == set(expected)

    def test_no_elements(self):
        elements = [PTABLE.get_element(x) for x in ["Cl", "H", "N", "P"]]
        expected = []
        res = mmi_finder._select_multiple_isotope_elements(elements)
        assert len(res) == len(expected)
        assert set(res) == set(expected)


@pytest.mark.parametrize(
    "elements,expected",
    [
        [["C", "H", "N", "O", "P", "S"], ["C", "O", "S"]],
        [["C", "H", "N", "O", "P", "S", "Cl", "Li", "Na"], ["C", "O", "S", "Li", "Cl"]],
    ],
)
def test__select_elements(elements, expected):
    element_list = [PTABLE.get_element(x) for x in elements]
    expected_elements = [PTABLE.get_element(x) for x in expected]
    actual_elements = mmi_finder._select_elements(element_list)
    assert sorted(actual_elements, key=lambda x: x.z) == sorted(expected_elements, key=lambda x: x.z)


@pytest.fixture(scope="module")
def config():
    bounds = {"C": (0, 108), "H": (0, 100), "S": (0, 8), "Cl": (0, 2)}
    return AnnotatorParameters(max_M=2000.0, bounds=bounds)


@pytest.fixture(scope="module")
def rules(config: AnnotatorParameters) -> dict:
    return mmi_finder._create_rules_dict(config, BIN_SIZE, DEFAULT_CONTEXT)


def create_feature_list(mz: list[float], sp: list[float], sample: Sample) -> Sequence[ConcreteFeature]:
    features = list()
    for k in range(len(mz)):
        ft = ConcreteFeature(data_mz=mz[k], data_area=sp[k], roi=ConcreteRoi(sample=sample))
        features.append(ft)
    return features


@pytest.fixture(scope="module")
def sample(tmp_path_factory) -> Sample:
    tmp_path = tmp_path_factory.mktemp("data")
    return create_sample(tmp_path, 1)


def test__find_candidates(config, rules, sample):
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the isotopic envelope.

    cl = PTABLE.get_element("Cl")
    dm_cl = cl.isotopes[1].a - cl.mmi.a
    mono_mz = 400.0
    charge = 1
    mono_index = 3
    mz = [100.0, 300.0, mono_mz - dm_cl, mono_mz, 456.0]
    sp = [100.0, 200.0, 500.0, 501.0, 34.0]
    peak_list = create_feature_list(mz, sp, sample)
    monoisotope = peak_list[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // BIN_SIZE)
    i_rules = rules.get(m_bin)[0]

    data = AnnotationData(peak_list)

    test_candidates = mmi_finder._find_candidate(data, monoisotope, charge, i_rules, config)
    mmi = peak_list[2]
    expected_candidates = [(mmi, 1)]
    assert test_candidates == expected_candidates


def test__find_candidates_multiple_candidates(config, rules, sample):
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the
    # isotopic envelope.
    cl = PTABLE.get_element("Cl")
    dm_cl = cl.isotopes[1].a - cl.mmi.a
    mono_mz = 400.0
    charge = 1
    mono_index = 4
    M01 = mono_mz - dm_cl
    M02 = M01 + 0.00001
    mz = [100.0, 300.0, M01, M02, mono_mz, 456.0]
    sp = [100.0, 200.0, 500.0, 500.5, 501.0, 34.0]
    peak_list = create_feature_list(mz, sp, sample)
    monoisotopologue = peak_list[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // BIN_SIZE)
    i_rules = rules.get(m_bin)[0]

    data = AnnotationData(peak_list)

    test_candidates = mmi_finder._find_candidate(data, monoisotopologue, charge, i_rules, config)
    expected_candidates = [(peak_list[2], 1), (peak_list[3], 1)]
    assert test_candidates == expected_candidates


def test__find_candidates_no_candidates(config, rules, sample):
    # create an m/z and sp list where the monoisotopic m/z is the M1 in the isotopic envelope.
    mono_mz = 400.0
    charge = 1
    mono_index = 2
    mz = [100.0, 300.0, mono_mz, 456.0]
    sp = [100.0, 200.0, 501.0, 34.0]
    peak_list = create_feature_list(mz, sp, sample)
    monoisotopologue = peak_list[mono_index]

    # find the rule to search the mmi candidate
    m_bin = int(mono_mz // BIN_SIZE)
    i_rules = rules.get(m_bin)[0]

    data = AnnotationData(peak_list)

    test_candidates = mmi_finder._find_candidate(data, monoisotopologue, charge, i_rules, config)
    assert len(test_candidates) == 0


def test_MMIFinder(config, sample):
    finder = mmi_finder.MMIFinder(config, BIN_SIZE, DEFAULT_CONTEXT)

    cl = PTABLE.get_element("Cl")
    dm_cl = cl.isotopes[1].a - cl.mmi.a
    mono_mz = 400.0
    mz = [100.0, 300.0, mono_mz - dm_cl, mono_mz, 456.0]
    sp = [100.0, 200.0, 500.0, 501.0, 34.0]
    peak_list = create_feature_list(mz, sp, sample)
    data = AnnotationData(peak_list)
    monoisotopologue = data.get_monoisotopologue()
    test_mmi_index = finder.find(data)
    expected_mmi_index = [
        (monoisotopologue, 1),
        (monoisotopologue, 2),
        (monoisotopologue, 3),
        (peak_list[2], 1),
    ]
    # check with set because features may be in a different order
    assert set(test_mmi_index) == set(expected_mmi_index)
