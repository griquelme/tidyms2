"""Implementation of the Minimum Mass Finder."""

import bisect

import numpy as np

from ..chem import EM, ChemicalContext
from ..chem.atoms import Element
from ..chem.envelope import make_formula_coefficients_envelopes
from ..chem.formula_generator import FormulaCoefficientBounds
from ..core.models import AnnotableFeatureType
from .annotation_data import AnnotationData
from .config import AnnotatorParameters


class MMIFinder:
    """Find the Minimum Mass Isotopologue (MMI) in a list of features.

    Parameters
    ----------
    bounds : dict
        Mapping from an element symbol str to the minimum and maximum
        allowed values in formulas.
    max_mass : float
        Maximum mass to build rules.
    length : int
        length of the theoretical envelopes used to compute the search
        rules.
    bin_size : int
        Mass interval used to build the rules.
    mz_tol : float
        m/z tolerance to search candidates.
    p_tol : float
        abundance tolerance used to search candidates.
    min_similarity : float, default=0.9
        Minimum similarity to create candidates.
    context: chemical context

    """

    def __init__(self, config: AnnotatorParameters, bin_size: int, context: ChemicalContext):
        self.context = context
        self.rules = _create_rules_dict(config, bin_size, context)
        self.bin_size = bin_size
        self.max_charge = abs(config.max_charge)
        self.polarity = 1 if config.max_charge >= 0 else -1
        self.config = config

    def find(self, data: AnnotationData[AnnotableFeatureType]) -> list[tuple[AnnotableFeatureType, int]]:
        """Search MMI candidates using features m/z and area.

        :param features: feature data to search.
        :return: list of MMI candidates and their possible charge.

        """
        mono = data.get_monoisotopologue()
        candidates = list()

        if mono is None:
            return candidates

        for charge in range(1, self.max_charge + 1):
            M_mono = mono.mz * charge - self.polarity * charge * EM
            if M_mono < self.config.max_M:
                candidates.append((mono, charge))
            M_bin = int(M_mono // self.bin_size)
            mmi_rules = self.rules.get(M_bin)
            if mmi_rules is not None:
                for i_rules in mmi_rules:
                    i_candidates = _find_candidate(data, mono, charge, i_rules, self.config)
                    candidates.extend(i_candidates)
        return candidates


def _find_candidate(
    data: AnnotationData[AnnotableFeatureType],
    mono: AnnotableFeatureType,
    charge: int,
    i_rules: dict,
    config: AnnotatorParameters,
) -> list[tuple[int, int]]:
    # search valid m/z values
    min_dM, max_dM = i_rules["dM"]
    min_mz = mono.mz - max_dM / charge - config.max_M_tol
    max_mz = mono.mz - min_dM / charge + config.max_M_tol
    min_qp = i_rules["qp"][0] - config.p_tol
    max_qp = i_rules["qp"][1] + config.p_tol

    if (mono.mz * charge) < config.max_M:
        start = bisect.bisect(data.features, min_mz, key=lambda x: x.mz)
        end = bisect.bisect(data.features, max_mz, key=lambda x: x.mz)
    else:
        start, end = 0, 0  # dummy values

    # if valid m/z where found, check if the abundance quotient qp is valid
    candidates = list()
    if start < end:
        for k in range(start, end):
            candidate = data.features[k]
            is_valid = _check_candidate(data, mono, candidate, config.min_similarity, min_qp, max_qp)
            if is_valid:
                candidates.append((candidate, charge))
    return candidates


def _check_candidate(
    data: AnnotationData[AnnotableFeatureType],
    mono: AnnotableFeatureType,
    candidate: AnnotableFeatureType,
    min_similarity: float,
    min_qp: float,
    max_qp: float,
) -> bool:
    if candidate not in data.non_annotated:
        return False

    similarity = data.similarity_cache.get_similarity(mono, candidate)

    if similarity < min_similarity:
        return False

    p = mono.compute_isotopic_envelope(candidate, mono).p
    qp = p[1] / p[0]
    is_valid_qp = (qp >= min_qp) & (qp <= max_qp)

    return is_valid_qp


def _create_rules_dict(
    config: AnnotatorParameters, bin_size: int, context: ChemicalContext
) -> dict[int, list[dict[str, tuple[float, float]]]]:
    Ma, pa = _create_envelope_arrays(config, context)
    # find the monoisotopic index, its Mass difference with the MMI (dM) and
    # its abundance quotient with the MMI (qp)
    bins = (Ma[:, 0] // bin_size).astype(int)

    # find unique values for bins and monoisotopic index that will be used
    # as key for the rule dictionary
    unique_bins = np.unique(bins)
    # unique_mono_index = np.unique(mono_index)
    # unique_mono_index = unique_mono_index[unique_mono_index > 0]

    rules = dict()
    for b in unique_bins:
        b_rules = list()
        bin_mask = bins == b
        for mi in range(1, config.max_length):
            qp = pa[bin_mask, mi] / pa[bin_mask, 0]
            dM = Ma[bin_mask, mi] - Ma[bin_mask, 0]
            qp_mask = qp >= (1.0 - config.p_tol)
            if qp_mask.any():
                mi_rules = dict()
                dM_b_mi = dM[qp_mask]
                qp_b_mi = qp[qp_mask]
                mi_rules["dM"] = dM_b_mi.min(), dM_b_mi.max()
                mi_rules["qp"] = qp_b_mi.min(), qp_b_mi.max()
                b_rules.append(mi_rules)
        if b_rules:
            rules[b] = b_rules
    return rules


def _create_envelope_arrays(config: AnnotatorParameters, context: ChemicalContext) -> tuple[np.ndarray, np.ndarray]:
    selected_elements = {x.symbol for x in _select_elements([context.table.get_element(x) for x in config.bounds])}
    bounds = {k: v for k, v in config.bounds.items() if k in selected_elements}
    coeff_bounds = FormulaCoefficientBounds.from_isotope_str(bounds)
    formula_coefficients = coeff_bounds.make_coefficients(config.max_M)
    envelope = make_formula_coefficients_envelopes(
        bounds, formula_coefficients, context.envelope_cache, context.table, config.max_length
    )
    M = envelope.M
    p = envelope.p
    return M, p


def _select_two_isotope_element(elements: list[Element], dm: int) -> list[Element]:
    """Select elements from the list with two isotopes.

    All elements where the abundance of the MMI is lower than the other isotope are kept.
    In cases where the MMI is greater than the other isotope, only the isotope with
    the minimum abundance from this subset is kept.

    """
    selected: list[Element] = list()
    max_p1_where_p0_gt_p1 = 0.0  # maximum value of two element isotopes p0 where p0 > p1
    best_p0_greater_than_p1 = None
    for element in elements:
        if len(element.isotopes) != 2:
            continue

        other = element.isotopes[1]

        element_dm = other.a - element.mmi.a
        if element_dm != dm:
            continue

        if other.p >= element.mmi.p:
            selected.append(element)
        elif other.p > max_p1_where_p0_gt_p1:
            max_p1_where_p0_gt_p1 = other.p
            best_p0_greater_than_p1 = element

    if best_p0_greater_than_p1 is not None:
        selected.append(best_p0_greater_than_p1)
    return selected


def _select_multiple_isotope_elements(e_list: list[Element]) -> list[Element]:
    selected = list()
    for e in e_list:
        n_isotopes = len(e.isotopes)
        if n_isotopes > 2:
            selected.append(e)
    return selected


def _select_elements(e_list: list[Element]) -> list[Element]:
    two_isotope_dm1 = _select_two_isotope_element(e_list, 1)
    two_isotope_dm2 = _select_two_isotope_element(e_list, 2)
    selected = _select_multiple_isotope_elements(e_list)
    selected.extend(two_isotope_dm1)
    selected.extend(two_isotope_dm2)
    return selected
