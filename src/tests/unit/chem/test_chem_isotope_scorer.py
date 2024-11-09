import numpy as np
import pytest

from tidyms2.chem import EnvelopeScorer, EnvelopeValidator, Formula
from tidyms2.chem.config import EnvelopeScorerConfiguration, EnvelopeValidatorConfiguration

formula_str_list = ["C11H12N2O2", "C6H12O6", "C27H46O", "CO2", "HCOOH"]


class TestEnvelopeValidator:
    max_length = 5

    @pytest.fixture(scope="class")
    def validator(self):
        config = EnvelopeValidatorConfiguration.from_chnops(500, max_length=self.max_length)
        return EnvelopeValidator(config)

    @pytest.mark.parametrize("f_str", formula_str_list)
    def test_validate_valid_envelope_ok(self, f_str: str, validator: EnvelopeValidator):
        f = Formula(f_str)
        env = f.get_isotopic_envelope(self.max_length)
        validated_length = validator.validate(env.mz, env.p)
        assert validated_length == self.max_length

    def test_validate_invalid_envelope_return_zero_validated_length(self, validator):
        f = Formula("C2H8B")
        env = f.get_isotopic_envelope(self.max_length)
        validated_length = validator.validate(env.mz, env.p)
        expected_length = 0
        assert validated_length == expected_length


class TestEnvelopeScorer:
    max_length = 5
    max_M = 1000.0

    @pytest.fixture(scope="class")
    def scorer(self):
        config = EnvelopeScorerConfiguration.from_chnops(500, max_length=5)
        return EnvelopeScorer(config)

    @pytest.mark.parametrize("f_str", formula_str_list)
    def test_EnvelopeScorer(self, f_str, scorer):
        # test that the best scoring candidate has the same molecular formula
        f = Formula(f_str)
        f_envelope = f.get_isotopic_envelope(self.max_length)
        tolerance = 0.005
        scorer.score(f_envelope.mz, f_envelope.p, tolerance)
        coeff, isotopes, _ = scorer.get_top_results(5)
        expected_coeff = [f.composition[x] for x in isotopes]
        assert np.array_equal(expected_coeff, coeff[0])

    @pytest.mark.parametrize("f_str", formula_str_list)
    def test_EnvelopeScorer_length_gt_scorer_max_length_raises_error(self, scorer, f_str):
        # test that the best scoring candidate has the same molecular formula
        f = Formula(f_str)
        f_envelope = f.get_isotopic_envelope(self.max_length + 1)
        tolerance = 0.005

        with pytest.raises(ValueError):
            scorer.score(f_envelope.mz, f_envelope.p, tolerance)

    @pytest.mark.parametrize("f_str", formula_str_list)
    def test_EnvelopeScorer_custom_scorer(self, f_str, scorer):
        def cosine_scorer(mz1, ab1, mz2, ab2, **scorer_params):
            n1 = np.linalg.norm(ab1)
            n2 = np.linalg.norm(ab2)
            norm = np.linalg.norm(ab1 - ab2)
            cosine = norm / (n1 * n2)
            return 1 - cosine

        scorer_config = EnvelopeScorerConfiguration.from_chnops(500)
        scorer = EnvelopeScorer(scorer_config, scorer=cosine_scorer)

        f = Formula(f_str)
        max_length = 5
        f_envelope = f.get_isotopic_envelope(max_length)
        tolerance = 0.005
        scorer.score(f_envelope.mz, f_envelope.p, tolerance)
        coeff, isotopes, _ = scorer.get_top_results(5)
        expected_coeff = [f.composition[x] for x in isotopes]
        assert np.array_equal(expected_coeff, coeff[0])

    @pytest.mark.parametrize("f_str", ["C2H3N", "N2H4", "C3N3H3"])
    def test_scorer_positive_defects_element_only(self, f_str):
        bounds = {"C": (0, 10), "H": (0, 10), "N": (0, 10)}
        config = EnvelopeScorerConfiguration(bounds=bounds, max_length=self.max_length, max_M=self.max_M)
        scorer = EnvelopeScorer(config)

        f = Formula(f_str)
        f_envelope = f.get_isotopic_envelope(self.max_length)
        tolerance = 0.005
        scorer.score(f_envelope.mz, f_envelope.p, tolerance)
        coeff, isotopes, _ = scorer.get_top_results(5)
        expected_coeff = [f.composition[x] for x in isotopes]
        assert np.array_equal(expected_coeff, coeff[0])

    @pytest.mark.parametrize("f_str", ["CS2", "C2OS2", "C3SO"])
    def test_scorer_negative_elements_only(self, f_str):
        bounds = {"C": (0, 10), "O": (0, 10), "S": (0, 10)}
        config = EnvelopeScorerConfiguration(bounds=bounds, max_length=self.max_length, max_M=self.max_M)
        scorer = EnvelopeScorer(config)

        f = Formula(f_str)
        env = f.get_isotopic_envelope(self.max_length)
        tolerance = 0.001
        scorer.score(env.mz, env.p, tolerance)
        coeff, isotopes, _ = scorer.get_top_results(5)
        expected_coeff = [f.composition[x] for x in isotopes]
        assert np.array_equal(expected_coeff, coeff[0])

    @pytest.mark.parametrize("f_str", ["H2O", "H3PO4", "H2SO4"])
    def test_no_carbon_scorer(self, f_str):
        bounds = {"H": (0, 10), "O": (0, 5), "S": (0, 5), "P": (0, 5)}
        config = EnvelopeScorerConfiguration(bounds=bounds, max_length=self.max_length, max_M=self.max_M)
        scorer = EnvelopeScorer(config)

        f = Formula(f_str)
        env = f.get_isotopic_envelope(self.max_length)
        tolerance = 0.005
        scorer.score(env.mz, env.p, tolerance)
        coeff, isotopes, _ = scorer.get_top_results(5)
        expected_coeff = [f.composition[x] for x in isotopes]
        assert np.array_equal(expected_coeff, coeff[0])
