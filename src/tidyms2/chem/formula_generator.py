"""Functions to calculate compatible molecular formulas within a given tolerance."""

from __future__ import annotations

import numpy as np
import pydantic

from ..core.utils.numpy import IntArray, cartesian_product_from_ranges
from .atoms import Isotope
from .config import FormulaGeneratorConfiguration
from .context import PTABLE

# name conventions used:
# M is used for molecular mass
# m is used for nominal mass
# d is used for mass defect
# e.g. for (12C)(1H)2(16O)2, M = 46.0055; m = 46; d = 0.0055
# dp is used for the contribution to d of isotopes with positive mass defect
# dn is used for the contribution to d of isotopes with negative mass defect
# d = dp + dn
# for (12C)(1H)2(16O)2 dp = 0.0157, dn = -0.0102.
# q and r are used for the quotient and remainder of the division of m by 12
# for (12C)(1H)2(16O)2 q = 3; r = 10
# mp is used for the contribution to m of isotopes with positive mass defect
# mn is used for the contribution to m of isotopes with negative mass defect
# qp and rp are the quotient and remainder of the division of mp by 12
# qn and rn are the quotient and remainder of the division of mn by 12
# mc is used fot the contribution to m of 12C.
# qc is the quotient of division of mc by 12.
# rc is not used but it is always zero.


class FormulaGenerator:
    """Generate sum formulas based on exact mass values.

    Refer to the :ref:`user guide <generating-formulas-guide>` for usage instructions.

    """

    def __init__(self, config: FormulaGeneratorConfiguration):
        """FormulaGenerator constructor.

        :param config: The formula generator configuration.

        """
        bounds = FormulaCoefficientBounds.from_isotope_str(config.bounds)
        self.bounds = bounds.bounds_from_mass(config.max_M, 0.0)

        dp_min, dp_max, dn_min, dn_max = self.bounds.get_defect_bounds()

        self._min_defect = dn_min + dp_min
        self._max_defect = dp_max + dn_max

        c12 = PTABLE.get_isotope("12C")
        self._has_carbon = c12 in self.bounds.bounds
        self._query = None
        self._results = None
        self._n_results = None

        # build Coefficients
        pos_restrictions, neg_restrictions = self.bounds.split_pos_neg()
        # add a dummy negative (or positive) isotope when only positive
        # (negative) isotopes are being used. This is a hack that prevents
        # making changes in the formula generation code...
        _add_dummy_isotope(pos_restrictions, neg_restrictions)
        self.pos = pos_restrictions.make_coefficients(config.max_M, reverse=True)
        self.neg = neg_restrictions.make_coefficients(config.max_M)

    def __repr__(self):
        return f"FormulaGenerator(bounds={self.bounds.bounds})"

    def get_results(self) -> dict:
        """Retrieve the formula generator results in the internal format.

        :return: a mapping of nominal masses of the results to a tuple of three arrays:
            1. the row index of positive coefficients.
            2. the row index of negative coefficients.
            3. the number of 12C in the formula.

        """
        if self._results is None:
            raise ValueError("No results found. This method must be called after `generate_formulas`")
        return self._results

    def get_n_results(self) -> int:
        """Retrieve the number of formulas found after a query."""
        if self._n_results is None:
            raise ValueError("No results found. This method must be called after `generate_formulas`")
        return self._n_results

    def generate_formulas(
        self, M: float, tolerance: float, min_defect: float | None = None, max_defect: float | None = None
    ):
        """Compute formulas compatibles with the given query mass.

        The formulas are computed assuming neutral species. If charged species are used, mass
        values must be corrected using the electron mass.

        Results are stored in an internal format, use :py:func:`results_to_array` to
        obtain the compatible formulas.

        :param M: query mass used for formula generation
        :param tolerance: mass tolerance to search compatible formulas
        :param min_defect: if provided, filter formulas with mass defects lower than this value
        :param max_defect: if provided, filter formulas with mass defects greater than this value

        >>> from tidyms2.chem import FormulaGenerator
        >>> fg_bounds = {"C": (0, 5), "H": (0, 10), "O": (0, 4)}
        >>> fg = FormulaGenerator(fg_bounds)
        >>> fg.generate_formulas(46.042, 0.005)

        """
        if M <= 0.0:
            msg = "`M` must be a positive number. Got {}".format(M)
            raise ValueError(msg)

        if tolerance <= 0.0:
            msg = "`tolerance` must be a positive number. Got {}".format(tolerance)
            raise ValueError(msg)

        min_defect = self._min_defect if min_defect is None else min_defect
        max_defect = self._max_defect if max_defect is None else max_defect

        self._results, self._n_results = _generate_formulas(
            M, tolerance, self.bounds, self.pos, self.neg, min_defect=min_defect, max_defect=max_defect
        )

    def results_to_array(self) -> tuple[np.ndarray, list[Isotope], np.ndarray]:
        """Convert results to an array of coefficients.

        :return: tuple containing a 2D array wit rows of formula coefficients, a list
            of isotopes associated with each coefficient and a 1D array with the exact
            mass of each formula.

        >>> import tidyms as ms
        >>> fg_bounds = {"C": (0, 5), "H": (0, 10), "O": (0, 4)}
        >>> fg = ms.chem.FormulaGenerator(fg_bounds)
        >>> fg.generate_formulas(46.042, 0.005)
        >>> coeff, isotopes, M = fg.results_to_array()

        """
        if self._results is None or self._n_results is None:
            raise ValueError("No results computed.")
        return _results_to_array(self._results, self._n_results, self.pos, self.neg, self._has_carbon)


class IsotopeCoeffBounds(pydantic.BaseModel):
    """Store a lower and upper bounds for formula coefficients bounds."""

    lower: pydantic.NonNegativeInt
    upper: pydantic.NonNegativeInt


class MassBounds(pydantic.BaseModel):
    """Store lower and upper bounds for mass."""

    lower: float
    upper: float


class MassDefectBounds(pydantic.BaseModel):
    """Store positive and negative mass defect bounds build from formula coefficients."""

    pos: MassBounds
    neg: MassBounds


class FormulaCoefficientBounds(pydantic.BaseModel):
    """Mapping from isotopes to upper and lower bounds."""

    bounds: dict[Isotope, IsotopeCoeffBounds]
    """A mapping from isotopes to coefficient bounds"""

    def __getitem__(self, item):
        return self.bounds[item]

    def bounds_from_mass(self, M: float, tol: float) -> FormulaCoefficientBounds:
        """Compute the mass-based bounds for each isotope.

        The bounds are refined using the values for each isotope.

        """
        bounds = dict()
        for i, b in self.bounds.items():
            lower = max(0, b.lower)
            upper = min(int((M + tol) / i.m), b.upper)
            bounds[i] = IsotopeCoeffBounds(lower=lower, upper=upper)
        return FormulaCoefficientBounds(bounds=bounds)

    def bound_negative_positive_defect(self, defect: float, tolerance: float) -> MassDefectBounds:
        """Bound positive and negative contributions to mass defect.

        :param defect: mass defect value of the Query.
        :param tolerance: mass tolerance

        """
        min_p, max_p, min_n, max_n = self.get_defect_bounds()
        max_pos = float(min(defect - min_n, max_p) + tolerance)
        min_neg = float(max(defect - max_p, min_n) - tolerance)
        min_pos = float(max(defect - max_n, min_p) - tolerance)
        max_neg = float(min(defect - min_p, max_n) + tolerance)
        return MassDefectBounds(
            pos=MassBounds(lower=min_pos, upper=max_pos), neg=MassBounds(lower=min_neg, upper=max_neg)
        )

    def get_nominal_defect_candidates(self, M: float, tol: float) -> tuple[list[int], list[float]]:
        """Split mass into possible values of nominal mass and mass defect."""
        dp_min, dp_max, dn_min, dn_max = self.get_defect_bounds()
        m_min = int(M - tol - dn_max - dp_max) + 1
        m_max = int(M + tol - dp_min - dn_min) + 1
        m_candidates = list(range(m_min, m_max))
        d_candidates = [M - x for x in m_candidates]
        return m_candidates, d_candidates

    def split_pos_neg(self) -> tuple[FormulaCoefficientBounds, FormulaCoefficientBounds]:
        """Split bounds into a two new instances, one with positive defect isotopes and other with negative defects."""
        pos_bounds = dict()
        neg_bounds = dict()
        for isotope, bounds in self.bounds.items():
            defect = isotope.d
            if defect > 0:
                pos_bounds[isotope] = bounds
            elif defect < 0:
                neg_bounds[isotope] = bounds
        pos = FormulaCoefficientBounds(bounds=pos_bounds)
        neg = FormulaCoefficientBounds(bounds=neg_bounds)
        return pos, neg

    def get_defect_bounds(self) -> tuple[float, float, float, float]:
        """Compute the minimum and maximum value of the mass defect."""
        min_positive, max_positive, min_negative, max_negative = 0, 0, 0, 0
        for isotope, isotope_bounds in self.bounds.items():
            defect = isotope.d
            min_tmp = defect * isotope_bounds.lower
            max_tmp = defect * isotope_bounds.upper
            if defect > 0:
                min_positive += min_tmp
                max_positive += max_tmp
            else:
                min_negative += max_tmp
                max_negative += min_tmp
        return min_positive, max_positive, min_negative, max_negative

    def make_coefficients(self, max_M: float, reverse: bool = False) -> FormulaCoefficients:
        """Generate coefficients for FormulaGenerator."""
        return FormulaCoefficients(self, max_M, True, reverse)

    @staticmethod
    def from_isotope_str(bounds: dict[str, tuple[int, int]]) -> FormulaCoefficientBounds:
        """Create a _Bounds instance from a list of isotope strings."""
        res = dict()
        for i, (lb, ub) in bounds.items():
            try:
                element = PTABLE.get_element(i)
                isotope = element.mmi
            except KeyError:
                isotope = PTABLE.get_isotope(i)
            res[isotope] = IsotopeCoeffBounds(lower=lb, upper=ub)
        return FormulaCoefficientBounds(bounds=res)


class FormulaCoefficients:
    """Named tuple with elements with positive/negative mass defect.

    :param coefficients: np.array[int]
        Formula coefficients. Each row is a formula, each column is an isotope.
    :param isotopes : List[Isotopes]
        element associated to each column of coefficients.
    :param M: monoisotopic mass associated to each row of coefficients.
    :param q: quotient between the nominal mass and 12.
    :param r_to_index: Maps remainders of division of m by 12 to rows of `coefficients`.
    :param r_to_d: Maps remainders of division of m by 12 to mass defect  values. Each
        value match to the corresponding index in `r_to_index`.

    """

    def __init__(
        self,
        bounds: FormulaCoefficientBounds,
        max_mass: float,
        return_sorted: bool,
        return_reversed: bool,
    ):
        self.isotopes = list(bounds.bounds)
        i_M = np.array([isotope.m for isotope in self.isotopes])
        i_m = np.array([isotope.a for isotope in self.isotopes])
        i_d = np.array([isotope.d for isotope in self.isotopes])

        # create coefficients array
        range_list = [list(range(b.lower, b.upper + 1)) for b in bounds.bounds.values()]
        coefficients = cartesian_product_from_ranges(*range_list)

        # sort coefficients by mass defect
        d = np.matmul(coefficients, i_d)
        if return_sorted:
            sorted_index = np.argsort(d)
            if return_reversed:
                sorted_index = sorted_index[::-1]
            coefficients = coefficients[sorted_index, :]
            d = d[sorted_index]

        # remove coefficients with mass higher than the maximum mass
        M = np.matmul(coefficients, i_M)
        valid_M = M <= max_mass
        coefficients = coefficients[valid_M, :]
        d = d[valid_M]
        M = M[valid_M]

        # Compute nominal mass, quotient and remainder
        m = np.matmul(coefficients, i_m)
        q, r = np.divmod(m, 12)

        # group mass defects and coefficient row index by remainder value
        r_to_d = _make_remainder_arrays(d, r)
        r_to_index = _make_remainder_arrays(np.arange(d.size), r)

        self.M = M
        self.coefficients = coefficients
        self.q = q
        self.r_to_index = r_to_index
        self.r_to_d = r_to_d


class MassQuery(pydantic.BaseModel):
    """Stores values for a Mass query."""

    m: pydantic.PositiveInt
    """nominal mass of the query."""

    d: float
    """mass defect of the query."""

    q: pydantic.NonNegativeInt
    """quotient of the division of `m` by 12."""

    r: pydantic.NonNegativeInt
    """remainder of the division of `m` by 12."""

    tol: pydantic.PositiveFloat
    """Mass tolerance for the query"""

    carbon_bounds: IsotopeCoeffBounds
    """Minimum and maximum number of 12C in the formula"""

    defect_bounds: MassDefectBounds
    """Mass defect bounds for valid formulas"""

    @classmethod
    def from_coeffs(cls, m: int, d: float, tol: float, bounds: FormulaCoefficientBounds):
        """Create a mass query from formula bounds."""
        q, r = divmod(m, 12)
        d_bounds = bounds.bound_negative_positive_defect(d, tol)

        c12 = PTABLE.get_isotope("12C")
        if c12 in bounds.bounds:
            c12_bounds = bounds.bounds[c12].model_copy()
        else:
            c12_bounds = IsotopeCoeffBounds(lower=0, upper=0)

        return MassQuery(m=m, d=d, q=q, r=r, defect_bounds=d_bounds, carbon_bounds=c12_bounds, tol=tol)


def _add_dummy_isotope(pos: FormulaCoefficientBounds, neg: FormulaCoefficientBounds):
    """Add dummy isotopes to empty bounds.

    This is to positive or negative elements to solve the mass defect problem in cases that there aren't any
    positive/negative isotopes.

    """
    if len(pos.bounds) == 0:
        h1 = PTABLE.get_isotope("1H")
        pos.bounds[h1] = IsotopeCoeffBounds(lower=0, upper=0)

    if len(neg.bounds) == 0:
        o16 = PTABLE.get_isotope("16O")
        neg.bounds[o16] = IsotopeCoeffBounds(lower=0, upper=0)


def _results_to_array(
    results: dict,
    n: int,
    pos: FormulaCoefficients,
    neg: FormulaCoefficients,
    has_carbon: bool,
) -> tuple[np.ndarray, list[Isotope], np.ndarray]:
    """Convert results from guess_formula to a numpy array of coefficients."""
    isotopes = pos.isotopes + neg.isotopes
    if has_carbon:
        isotopes = [PTABLE.get_isotope("12C")] + isotopes
        ic = 0
        i_pos = 1
        i_neg = i_pos + len(pos.isotopes)
    else:
        ic = 0
        i_pos = 0
        i_neg = i_pos + len(pos.isotopes)
    res = np.zeros((n, len(isotopes)), dtype=int)
    mass = np.zeros(n)
    start = 0
    for k in results:
        end = start + results[k][0].size
        res[start:end, i_pos:i_neg] = pos.coefficients[results[k][0], :]
        res[start:end, i_neg:] = neg.coefficients[results[k][1], :]
        if has_carbon:
            res[start:end, ic] = results[k][2]
            mass[start:end] = pos.M[results[k][0]] + 12 * np.array(results[k][2]) + neg.M[results[k][1]]
        else:
            mass[start:end] = pos.M[results[k][0]] + neg.M[results[k][1]]
        start = end
    res, isotopes = _remove_dummy_isotopes(pos, neg, has_carbon, isotopes, res)
    return res, isotopes, mass


def _remove_dummy_isotopes(
    pos: FormulaCoefficients,
    neg: FormulaCoefficients,
    has_carbon: bool,
    isotopes: list[Isotope],
    res: IntArray,
) -> tuple[IntArray, list[Isotope]]:
    has_pos = pos.coefficients.max() > 0
    has_neg = neg.coefficients.max() > 0

    if not has_pos:
        pop_ind = 1 if has_carbon else 0
        isotopes.pop(pop_ind)
        res = np.delete(res, pop_ind, axis=1)

    if not has_neg:
        pop_ind = res.shape[1] - 1
        isotopes.pop(pop_ind)
        res = np.delete(res, pop_ind, axis=1)

    return res, isotopes


def _make_remainder_arrays(x, r) -> dict[int, np.ndarray]:
    """Create a dictionary where each key is a value of r and the values are the corresponding x values.

    Auxiliary function of _make_coefficients.

    :param x: array
    :param r: array of remainders.

    Returns
    -------
    r_to_x : Dict

    """
    r_to_x = dict()
    for k in range(12):
        xk = x[r == k]
        r_to_x[k] = xk
    return r_to_x


def _generate_formulas(
    M: float,
    tol: float,
    bounds: FormulaCoefficientBounds,
    pos: FormulaCoefficients,
    neg: FormulaCoefficients,
    min_defect: float,
    max_defect: float,
):
    """Find formulas compatible with a given mass."""
    bounds = bounds.bounds_from_mass(M, tol)
    # possible values of nominal mass and mass defect based on the coefficient bounds.
    m_candidates, d_candidates = bounds.get_nominal_defect_candidates(M, tol)
    res = dict()
    n = 0  # number of valid formulas
    for m, d in zip(m_candidates, d_candidates):
        if (d + tol < min_defect) or (d - tol > max_defect):
            continue

        query = MassQuery.from_coeffs(m, d, tol, bounds)

        pos_index = list()
        neg_index = list()
        qc = list()
        for i in range(12):
            results = _generate_formulas_i(i, query, pos, neg)
            if results is not None:
                pos_index_i, neg_index_i, qc_i = results
                n += pos_index_i.size
                pos_index.append(pos_index_i)
                neg_index.append(neg_index_i)
                qc.append(qc_i)
        if pos_index:
            res[m] = (np.hstack(pos_index), np.hstack(neg_index), np.hstack(qc))
    return res, n


def _generate_formulas_i(
    i: int,
    query: MassQuery,
    pos: FormulaCoefficients,
    neg: FormulaCoefficients,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Solve the mass defect problem for fixed rp and rn values.

    Auxiliary function to _generate_formulas.
    """
    # solves the mass defect problem for fixe rp and rn values.
    # Finds all positive coefficients with rp == i and their matching negative
    # such that |d - dp - dn | <= tol.
    # These values are filtered taking into account the number of 12C, qc, as
    # follows: min_nc <= qc <= max_nc
    # the results are organized into three arrays
    # p_index contains the index to a row of pos.coeff
    # n_index contains the index to a row of neg.coeff
    # q_c contains the number of 12C in the formula

    rp = i
    rn = (query.r - i) % 12

    # find valid positive mass defect values
    rel_dp = query.d - pos.r_to_d[rp]
    rel_dp_bounds = query.d - query.defect_bounds.pos.upper, query.d - query.defect_bounds.pos.lower
    p_start, p_end = np.searchsorted(rel_dp, rel_dp_bounds)
    rel_dp = rel_dp[p_start:p_end]

    # filter values based on valid number of C
    p_index = pos.r_to_index[rp][p_start:p_end]
    qp = pos.q[p_index]
    valid_qp = (query.q - qp) >= query.carbon_bounds.lower
    rel_dp = rel_dp[valid_qp]
    p_index = p_index[valid_qp]

    # find valid negative mass defect values
    dn = neg.r_to_d[rn]
    n_index = neg.r_to_index[rn]
    n_start = np.searchsorted(dn, rel_dp - query.tol)
    n_end = np.searchsorted(dn, rel_dp + query.tol)

    # create three arrays, where each element corresponds to an index of valid
    # positive coeff, negative coeff and number of 12C atoms.
    n_index_size = n_end - n_start
    valid_dn = n_index_size > 0
    n_start = n_start[valid_dn]
    n_end = n_end[valid_dn]
    n_index_size = n_index_size[valid_dn]
    p_index = p_index[valid_dn]
    if p_index.size:
        p_index = np.repeat(p_index, n_index_size)
        n_index = np.hstack([n_index[s:e] for s, e in zip(n_start, n_end)])

        qp = pos.q[p_index]
        qn = neg.q[n_index]
        extra_c = int((rp + rn) >= 12)
        qc = query.q - qp - qn - extra_c

        # valid results
        valid_qc = (qc >= query.carbon_bounds.lower) & (qc <= query.carbon_bounds.upper)
        p_index = p_index[valid_qc]
        n_index = n_index[valid_qc]
        qc = qc[valid_qc]
        if p_index.size:
            results = p_index, n_index, qc
        else:
            results = None
    else:
        results = None
    return results
