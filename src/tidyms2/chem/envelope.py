"""Utilities to validate and score isotopic envelopes."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.special import erfc

from ..core.utils.numpy import FloatArray1D
from .config import EnvelopeScorerConfiguration, EnvelopeValidatorConfiguration
from .context import DEFAULT_CONTEXT, ChemicalContext
from .envelope_utils import (
    Envelope,
    EnvelopeQuery,
    EnvelopeQueryTolerance,
    HomoAtomicEnvelopeCache,
    combine_envelopes,
)
from .formula_generator import FormulaCoefficients, FormulaGenerator
from .table import PeriodicTable


class _EnvelopeGenerator:
    """Base class to generate envelopes from a list of molecular formulas."""

    def __init__(self, config: EnvelopeScorerConfiguration):
        if config.context is None:
            self._context = DEFAULT_CONTEXT
        else:
            self._context = ChemicalContext(config.context)

        self.config = config

        self._formula_generator = FormulaGenerator(config)
        self._pos_env = make_formula_coefficients_envelopes(
            config.bounds,
            self._formula_generator.pos,
            self._context.envelope_cache,
            self._context.table,
            config.max_length,
        )

        self._neg_env = make_formula_coefficients_envelopes(
            config.bounds,
            self._formula_generator.neg,
            self._context.envelope_cache,
            self._context.table,
            config.max_length,
        )
        self._c_env = _create_12c_envelope(
            self._formula_generator, self._context.table, self._context.envelope_cache, config.max_length
        )

        self._query: EnvelopeQuery | None = None
        self.results: Envelope | None = None

    def generate_envelopes(self, M: Sequence[float], p: Sequence[float], tolerance: float):
        """Compute isotopic envelopes for formula candidates generated using the MMI mass from the envelope.

        :param M: the exact mass of the envelope
        :param p: the envelope normalized abundances.
        :param tolerance: mass tolerance to generate formulas.

        """
        if len(M) > self.config.max_length:
            msg = "`max_length` ({}) is lower than the query length ({})"
            raise ValueError(msg)

        query = EnvelopeQuery(M=M[: self.config.max_length], p=p[: self.config.max_length])
        self._formula_generator.generate_formulas(query.get_mmi_mass(), tolerance)
        query_envelopes = _find_result_envelopes(self._formula_generator, self._pos_env, self._neg_env, self._c_env)
        self._query = query
        self.results = query_envelopes

    def filter(self, min_M_tol: float, max_M_tol: float, p_tol: float, k: int):
        r"""Filter values from the k-th envelope that are outside the specified bounds.

        :param min_M_tol: mass values lower than this value are filtered.
        :param max_M_tol: mass values greater than this value are filtered.
        :param p_tol: abundance tolerance
        :param k: the envelope index to filter values.

        Notes
        -----
        Envelopes are filtered based on the following inequality. For each i-th
        peak the m/z tolerance is defined as follows:

        .. math::

            t_{i} = t + (T - t)(1 - y_{i})

        where :math:`t_{i}` is the mass tolerance for the i-th peak, t is the
        `min_mz_tolerance`, T is the `max_mz_tolerance` and :math:`y_[i]` is
        the abundance of the i-th value. Using this tolerance, an interval is
        built for the query mass, and candidates outside this interval are
        removed. This approach accounts for greater m/z errors for lower
        intensity peaks in the envelope.

        """
        if self.results is None or self._query is None:
            msg = "Envelopes must be generated first using the generate method."
            raise ValueError(msg)
        tol = EnvelopeQueryTolerance(min_M_tol=min_M_tol, max_M_tol=max_M_tol, p_tol=p_tol)
        bounds = self._query.get_bounds(tol, k)
        self.results.exclude(bounds)


class EnvelopeValidator(_EnvelopeGenerator):
    r"""Envelope validator.

    Notes
    -----
    Envelope validation is performed as follows:

    1.  For a query envelope mass and abundance `Mq`and `pq`, all formulas
        compatibles with the MMI are computed (see FormulaGenerator).
    2.  For each i-th pair of `Mq` and `pq`, a mass tolerance and abundance
        tolerance is defined as follows:

        .. math::
            dM_{i} = dM^{\textrm{max}} * pq_{i} + dM^{\textrm{min}} (1 - pq_{i})

        Where :math:`dM^{\textrm{max}}` is `min_M_tol`, :math:`dM^{\textrm{min}}`
        is `max_M_tol` and :math:`pq_{i}` is the i-th query abundance.
        Using the mass tolerance and abundance tolerance, candidates with
        mass or abundance values outside this interval are removed.
    3.  The candidates that remains define a mass and abundance window for
        the i + 1 elements of `Mq` and `pq`. If the values fall inside the
        window, the i + 1 elements are validated and the procedure is repeated
        until all isotopologues are validated or until an invalid isotopologue
        is found.

    """

    def __init__(self, config: EnvelopeValidatorConfiguration):
        config = config.model_copy(deep=True)
        config.remove_elements_with_a_single_isotope()
        super(EnvelopeValidator, self).__init__(config)
        self.config = config

    def validate(self, M: Sequence[float], p: Sequence[float]) -> int:
        """Validate an envelope."""
        valid_length = 0
        tol = p[0] * self.config.min_M_tol + (1 - p[0]) * self.config.max_M_tol
        self.generate_envelopes(M, p, tol)
        if self.results is None:
            return 0
        query = self._query
        assert query is not None
        length = len(query.M)

        while (length >= 2) and (valid_length <= 1):
            for k in range(length):
                self.filter(self.config.min_M_tol, self.config.max_M_tol, self.config.p_tol, k)
                if self.results.get_n_rows():
                    valid_length = k + 1
                else:
                    break
            if (valid_length <= 1) and (length > 2):
                valid_length = 0
                query.crop()
                length -= 1
                self.results.reset_exclude()
                self.results.crop(length)
            else:
                break

        return valid_length


class EnvelopeScorer(_EnvelopeGenerator):
    r"""Rank formula candidates by comparing measured and theoretical isotopic envelopes.

    Refer to the :ref:`user guide <scoring-formulas-guide>` for details usage instructions.

    :param config: the envelope generator configuration
    :param scorer: function that scores formula candidate envelopes. If ``None``, the
        function :func:`score_envelope` is used. A custom scoring function can be passed
        with the following signature:

        .. code-block:: python

            def score(M, p, Mq, pq, **kwargs):
                pass

        where `M` and `p` are arrays of the formula candidates exact mass and abundances and
        `Mq` and `pq` are the query mass and query abundance.

    kwargs :
        Optional parameter to pass into the scoring function.

    """

    def __init__(self, config: EnvelopeScorerConfiguration, scorer: Callable | None = None, **kwargs):
        super(EnvelopeScorer, self).__init__(config)

        if callable(scorer):
            self.scorer = scorer
            self.scorer_params = kwargs
        else:
            self.scorer = score_envelope
            self.scorer_params = kwargs
        self.scores: FloatArray1D | None = None

    def score(self, M: Sequence[float], p: Sequence[float], tol: float):
        """Score the isotopic envelope.

        The results can be recovered using the `get_top_results` method.

        Formulas are generated assuming that the first element in the envelope
        is the minimum mass isotopologue.

        :param M: exact mass of the envelope.
        :param p: abundance of the envelope.
        :param tol: mass tolerance used in formula generation.

        """
        self.generate_envelopes(M, p, tol)
        assert self.results is not None
        assert self._query is not None
        n_results = self.results.get_n_rows()
        scores = np.zeros(n_results)

        query_M = np.array(self._query.M)
        query_p = np.array(self._query.p)

        for i, (Mi, pi) in enumerate(self.results.iterate_rows()):
            scores[i] = self.scorer(Mi, pi, query_M, query_p, **self.scorer_params)
        self.scores = scores

    def get_top_results(self, n: int | None = 10):
        """Fetch the top ranked formula candidates and their score.

        :param n: number of first n results to return. If ``None``, return all formula candidates.

        """
        coefficients, elements, _ = self._formula_generator.results_to_array()

        # sort coefficients using the score and keep the first n values
        assert self.scores is not None
        top_n_index = np.argsort(self.scores)
        if n is not None:
            top_n_index = top_n_index[: (-n - 1) : -1]

        scores = self.scores[top_n_index]
        coefficients = coefficients[top_n_index]
        return coefficients, elements, scores


def score_envelope(
    M: np.ndarray,
    p: np.ndarray,
    Mq: np.ndarray,
    pq: np.ndarray,
    min_sigma_M: float = 0.01,
    max_sigma_M: float = 0.01,
    min_sigma_p: float = 0.05,
    max_sigma_p: float = 0.05,
):
    r"""Score the similarity between two isotopes.

    :param M: theoretical mass values.
    :param p: theoretical abundances.
    :param Mq:  query Mass values
    :param pq: query abundances.
    :param min_sigma_M: minimum mass standard deviation
    :param max_sigma_M: maximum mass standard deviation
    :param min_sigma_p: minimum abundance standard deviation.
    :param max_sigma_p: maximum abundance standard deviation.

    Returns
    -------
    score : float
        Number between 0 and 1. Higher values are related with similar envelopes.

    Notes
    -----
    The query envelope is compared against the theoretical envelope assuming
    a likelihood approach, similar to the described in [1]_. It is assumed
    that the theoretical mass and abundance is a normal random variable,
    with mean values defined by `M` and `p` and standard deviation computed as
    follows:

    .. math::

        \sigma_{M,i} = p_{i} \sigma_{M}^{\textrm{max}} + (1 - p_{i}) \sigma_{M}^{\textrm{min}}

    Where :math:`\sigma_{M,i}` is the standard deviation for the i-th element
    of `M`, :math:`p_{i}` is the i-th element of `p`, :math:`\sigma_{M}^{\textrm{max}}`
    is `max_sigma_M` and :math:`\sigma_{M}^{\textrm{min}}` is `min_sigma_M`. An
    analogous computation is done to compute the standard deviation for each
    abundance. Using this values, the likelihood of generating the values `Mq` and
    `pq` from `M` and `p` is computed using the error function.

    References
    ----------
    ..  [1] Sebastian Böcker, Matthias C. Letzel, Zsuzsanna Lipták, Anton
        Pervukhin, SIRIUS: decomposing isotope patterns for metabolite
        identification, Bioinformatics, Volume 25, Issue 2, 15 January 2009,
        Pages 218–224, https://doi.org/10.1093/bioinformatics/btn603

    """
    mz_sigma = max_sigma_M + (min_sigma_M - max_sigma_M) * pq
    sp_sigma = max_sigma_p + (min_sigma_p - max_sigma_p) * pq
    M = M[: Mq.size]
    p = p[: Mq.size]
    # normalize again the candidate intensity to 1
    p = p / p.sum()

    # corrects overestimation of the first peak area. This is done computing
    # an offset factor to subtract to the first peak. This correction is applied
    # only if the offset is positive. The offset value is computed in a way to
    # satisfy two conditions: the abundance of the first peak is equal to the
    # abundance of the candidate peak and the total area is normalized to one.
    # offset = (spq[0] - sp[0]) / (1 - sp[0])
    # offset = max(0, offset)
    norm = (pq[0] - 1) / (p[0] - 1)
    # spq = spq / (1 - offset)
    if norm < 1:
        pq = pq / norm
        pq[0] = p[0]

    # add a max offset parameter

    Mq = Mq + M[0] - Mq[0]
    dmz = np.abs(M - Mq) / (np.sqrt(2) * mz_sigma)
    dmz = dmz[pq > 0]
    dsp = np.abs(p - pq) / (np.sqrt(2) * sp_sigma)
    score = erfc(dmz).prod() * erfc(dsp).prod()
    return score


def make_formula_coefficients_envelopes(
    bounds: dict[str, tuple[int, int]],
    coefficients: FormulaCoefficients,
    cache: HomoAtomicEnvelopeCache,
    table: PeriodicTable,
    max_length: int,
) -> Envelope:
    """Compute the isotopic envelopes for coefficient formulas."""
    # initialize envelopes
    rows = coefficients.coefficients.shape[0]
    M_arr = np.zeros((rows, max_length))
    p_arr = np.zeros((rows, max_length))
    p_arr[:, 0] = 1
    env = Envelope(M=M_arr, p=p_arr)

    for k, isotope in enumerate(coefficients.isotopes):
        # if a symbol is passed in bounds, e.g. "C", the column in coefficients
        # will be the isotope "12C". to find the abundance, the element symbol
        # is used in bounds. If an isotope is specified, it is assumed that it
        # has no envelope.
        if isotope.to_str() in bounds:
            lb, ub = bounds[isotope.to_str()]
            k_env = cache.make_envelope_array(isotope, lb, ub, max_length)
        elif isotope.symbol in bounds:
            lb, ub = bounds[isotope.symbol]
            element = table.get_element(isotope.z)
            k_env = cache.make_envelope_array(element, lb, ub, max_length)
        else:
            # ignore dummy elements used to solve the formula generation problem
            # This occurs only in cases where there are only isotopes with positive
            # or negative mass defects.
            continue
        # Create copies of each row based on the values in the coefficient arrays
        # using lb correct index values in cases when 0 is not the lower bound
        k_env.M = k_env.M[coefficients.coefficients[:, k] - lb, :]
        k_env.p = k_env.p[coefficients.coefficients[:, k] - lb, :]
        env = combine_envelopes(env, k_env)
    return env


def _find_result_envelopes(
    fg: FormulaGenerator,
    pos_env: Envelope,
    neg_env: Envelope,
    c_env: Envelope,
) -> Envelope | None:
    length = pos_env.M.shape[1]
    shape = (fg.get_n_results(), length)
    M = np.zeros(shape, dtype=float)
    p = np.zeros(shape, dtype=float)
    start = 0
    for kp_index, kn_index, kc_index in fg.get_results().values():
        k_size = kp_index.size
        if k_size > 0:
            kp_env = Envelope(M=pos_env.M[kp_index], p=pos_env.p[kp_index])
            kn_env = Envelope(M=neg_env.M[kn_index], p=neg_env.p[kn_index])
            kc_env = Envelope(M=c_env.M[kc_index], p=c_env.p[kc_index])

            k_env = combine_envelopes(kp_env, kn_env)
            k_env = combine_envelopes(k_env, kc_env)

            end = start + k_size
            M[start:end] = k_env.M
            p[start:end] = k_env.p
            start = end

    envelopes = Envelope(M=M, p=p)
    if M.size < 2:
        return None
    envelopes.crop(M.size)
    return envelopes


def _create_12c_envelope(
    formula_generator: FormulaGenerator, table: PeriodicTable, cache: HomoAtomicEnvelopeCache, length: int
) -> Envelope:
    c12 = table.get_isotope("12C")
    c = table.get_element("C")
    c12_bounds = formula_generator.bounds.bounds.get(c12)
    if c12_bounds is None:
        nc_min, nc_max = 0, 0
    else:
        nc_min, nc_max = c12_bounds.lower, c12_bounds.upper
    return cache.make_envelope_array(c, nc_min, nc_max, length)
