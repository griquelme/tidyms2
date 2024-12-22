"""Functions to find isotopic envelopes candidates in a list of m/z values."""

import bisect
from collections.abc import Sequence

from ..chem import ChemicalContext
from ..chem.atoms import Element
from ..core.models import AnnotableFeatureType, Feature
from .annotation_data import AnnotationData
from .config import AnnotatorParameters

# name conventions
# M is used for Molecular mass
# m for nominal mass
# p for abundances


class EnvelopeFinder:
    r"""Find isotopic envelopes candidates in a list of sorted m/z values.

    :param elements : the list of elements used to compute mass difference windows.
    :param mz_tolerance : m/z tolerance used to match candidates.
    :param max_length: maximum envelope length to search.
    :param min_p: the minimum abundance of the isotopes of each element to be used
        for m/z estimation. Must be a number between 0 and 1.
    min_similarity : minimum similarity to create candidates.

    Notes
    -----
    Using a list of elements, theoretical bounds are computed for each M1, M2,
    M3, etc... isotopologue. Then using these values and the `mz_tolerance` and
    the `max_charge`, the bounds are adjusted according to the following
    equations:

    .. math::

        mz_{k, min}= \frac{m_{k, min}{q} - mz_{tolerance}

        mz_{k, max}= \frac{m_{k, max}{q} + mz_{tolerance}

    where :math:`m_{k, min}` is the minimum theoretical value for the k-th
    isotopologue and q is the charge.

    The envelopes candidates found are determined based on m/z compatibility
    only. To reduce the number of candidates, the list of m/z values should be
    reduced by other means, such as correlation of the values.

    """

    def __init__(self, config: AnnotatorParameters, context: ChemicalContext):
        self.config = config
        elements = [context.table.get_element(x) for x in config.bounds]
        self.bounds = _make_exact_mass_difference_bounds(elements, config.min_p)

    def find(
        self, data: AnnotationData[AnnotableFeatureType], mmi: AnnotableFeatureType, charge: int
    ) -> list[Sequence[AnnotableFeatureType]]:
        """Find isotopic envelope candidates starting from the minimum mass isotopologue (MMI).

        :param data: annotation data with features used in the search
        :param mmi: a feature in `data used as the :term:`mmi`.
        :param charge: the absolute value of the charge state of the isotopic envelope

        """
        completed_candidates = list()

        candidates = [[mmi]]
        while candidates:
            # remove and extend a candidate
            candidate = candidates.pop()

            # find features with compatible m/z and similarities
            min_mz, max_mz = _get_next_mz_search_interval(candidate, self.bounds, charge, self.config.max_M_tol)
            start = bisect.bisect(data.features, min_mz, key=lambda x: x.mz)
            end = bisect.bisect(data.features, max_mz, key=lambda x: x.mz)
            new_features = list()
            for k in range(start, end):
                k_ft = data.features[k]
                is_similar = data.similarity_cache.get_similarity(mmi, k_ft) >= self.config.min_similarity
                is_non_annotated = k_ft in data.non_annotated
                if is_similar and is_non_annotated:
                    new_features.append(k_ft)

            # extend candidates with compatible features
            length = len(candidate)
            if new_features and (length < self.config.max_length):
                tmp = [candidate + [x] for x in new_features]
                candidates.extend(tmp)
            else:
                completed_candidates.append(candidate)
        completed_candidates = [x for x in completed_candidates if len(x) > 1]
        return _remove_sub_candidates(completed_candidates)


def _remove_sub_candidates(candidates: list[Sequence[AnnotableFeatureType]]) -> list[Sequence[AnnotableFeatureType]]:
    """Remove candidates that are subsets of other candidates."""
    validated = list()
    while candidates:
        last = candidates.pop()
        last_set = set(last)
        is_subset = False
        for candidate in candidates:
            is_subset = last_set <= set(candidate)
        if not is_subset:
            validated.append(last)
    return validated


def _get_next_mz_search_interval(
    envelope: Sequence[Feature],
    elements_mass_difference: dict[int, tuple[float, float]],
    charge: int,
    mz_tolerance: float,
) -> tuple[float, float]:
    """Compute the valid m/z range for a k-th isotopologue using previous isotopologues information.

    :return: a tuple with the minimum and maximum m/z values expected for the next isotopologue in
        the envelope.

    """
    # If the charge is 0 (neutral mass) the results are the same as using
    # charge = 1. There is no difference between positive and negative
    # charges
    charge = max(1, abs(charge))
    length = len(envelope)
    min_mz = envelope[-1].mz + 2  # dummy values
    max_mz = envelope[-1].mz - 2
    for dm, (min_dM, max_dM) in elements_mass_difference.items():
        i = length - dm
        if i >= 0:
            min_mz = min(min_mz, envelope[i].mz + min_dM / charge)
            max_mz = max(max_mz, envelope[i].mz + max_dM / charge)
    min_mz -= mz_tolerance
    max_mz += mz_tolerance
    return min_mz, max_mz


def _make_exact_mass_difference_bounds(elements: list[Element], min_p: float) -> dict[int, tuple[float, float]]:
    """Compute possible mass differences obtaining from changing one isotope."""
    dm_to_dM_list: dict[int, list[float]] = dict()
    for e in elements:
        for isotope in e.isotopes:
            if isotope == e.mmi or isotope.p < min_p:
                continue
            dm = isotope.a - e.mmi.a
            dM = isotope.m - e.mmi.m
            dM_list = dm_to_dM_list.setdefault(dm, list())
            dM_list.append(dM)

    return {k: (min(v), max(v)) for k, v in dm_to_dM_list.items()}
