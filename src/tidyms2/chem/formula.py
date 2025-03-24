"""Tools for working with chemical formulas."""

from __future__ import annotations

import string
from collections import Counter
from copy import copy
from typing import overload

from ..core.models import IsotopicEnvelope
from .atoms import EM, Isotope
from .context import DEFAULT_CONTEXT, ChemicalContext
from .envelope_utils import Envelope, HomoAtomicEnvelopeCache, combine_envelopes
from .exceptions import InvalidFormula, IsotopeError
from .table import PeriodicTable


class Formula:
    """Represent a chemical formula as a mapping from isotopes to formula coefficients.

    Refer to the :ref:`user guide <working-with-formulas-guide>` for usage instructions.

    :param formula: a string representation of a formula or a mapping from isotopes to formula coefficients
    :param charge: the numerical charge of the formula
    :param context: the chemical context where data is fetch from. In the majority of cases this parameter
        may be ignored. This parameter should be set only if working with custom isotopic abundances.

    >>> Formula("H2O")
    Formula(H2O)
    >>> Formula("(13C)O2")
    Formula((13C)O2)
    >>> Formula("[Cr(H2O)6]3+")
    Formula([H12CrO6]3+)
    >>> Formula("CH3CH2CH3")
    Formula(C3H8)
    >>> Formula({"C": 1, "17O": 2})
    Formula(C(17O)2)

    """

    @overload
    def __init__(self, formula: dict[Isotope, int], charge: int | None, context: None = None): ...

    @overload
    def __init__(self, formula: dict[str, int], charge: int | None, context: ChemicalContext | None = None): ...

    @overload
    def __init__(self, formula: str, charge: None = None, context: ChemicalContext | None = None): ...

    def __init__(self, formula, charge=None, context: ChemicalContext | None = None):
        self.context = DEFAULT_CONTEXT if context is None else context
        if isinstance(formula, str):
            formula_str, charge = _parse_charge(formula)
            composition = _parse_formula(formula_str, self.context.table)
        else:
            f_composition = formula
            charge = 0 if charge is None else charge
            composition: Counter[Isotope] = Counter()
            for k, v in f_composition.items():
                if isinstance(k, str):
                    isotope = self.context.table.get_isotope(k)
                elif isinstance(k, Isotope):
                    isotope = k
                else:
                    msg = "Composition keys must be a string representation of an isotope or an isotope object"
                    raise IsotopeError(msg)

                if not isinstance(v, int) or (v < 1):
                    msg = "Formula coefficients must be positive integers"
                    raise ValueError(msg)
                composition[isotope] = v

            if not isinstance(charge, int):
                msg = "Charge must be an integer"
                raise ValueError(msg)
        self.charge = charge
        self.composition = composition

    def __add__(self, other: Formula) -> Formula:
        sum_composition = copy(self.composition)
        sum_composition.update(other.composition)
        sum_charge = self.charge + other.charge
        return Formula(sum_composition, sum_charge)

    def __sub__(self, other: Formula) -> Formula:
        comp = copy(self.composition)
        comp.subtract(other.composition)
        charge = self.charge - other.charge
        min_coeff = min(comp.values())
        if min_coeff < 0:
            msg = "subtraction cannot generate negative coefficients"
            raise ValueError(msg)
        comp = Counter({k: v for k, v in comp.items() if v > 0})
        diff_f = Formula(comp, charge)
        return diff_f

    def __eq__(self, other) -> bool:
        return (self.charge == other.charge) and (self.composition == other.composition)

    def get_exact_mass(self) -> float:
        """Compute the exact mass of the formula.

        Returns
        -------
        exact_mass: float

        Examples
        --------
        >>> import tidyms as ms
        >>> f = ms.chem.Formula("H2O")
        >>> f.get_exact_mass()
        18.010564684

        """
        exact_mass = sum(x.m * k for x, k in self.composition.items())
        exact_mass -= EM * self.charge
        return exact_mass

    def get_nominal_mass(self) -> int:
        """Compute the nominal mass of the formula.

        >>> import tidyms as ms
        >>> f = ms.chem.Formula("H2O")
        >>> f.get_nominal_mass()
        18

        """
        nominal_mass = sum(x.a * k for x, k in self.composition.items())
        return nominal_mass

    def get_isotopic_envelope(self, n: int = 10, min_p: float = 1e-10) -> IsotopicEnvelope:
        """Compute the isotopic envelope of the formula.

        The natural abundance is assumed for each monoisotope, i.e., the most
        abundant isotope. If others isotopes are present in the formula, they
        are assumed to have abundance equal to 1. See the examples for a
        clarification of this.

        :param n: the number of isotopes to include in the results.
        :param min_p: isotopes are included until the abundance is lower than this value

        If no isotopes are specified in the formula, the natural abundance is used.

        >>> import tidyms as ms
        >>> f = ms.chem.Formula("C6H6")
        >>> print(f)
        C6H6
        >>> f.get_isotopic_envelope(n=3)
        (array([78.04695019, 79.05033578, 80.05373322]),
         array([0.93686877, 0.06144402, 0.00168606]))

        Using isotopes other than the monoisotope are treated as if they have
        an abundance equal to one.

        >>> f = ms.chem.Formula("(13C)6(2H)6")
        >>> print(f)
        (13C)6(2H)6
        >>> f.get_isotopic_envelope()
        (array([90.10473971]), array([1.]))

        """
        envelope = _find_formula_envelope(
            self.composition, self.context.envelope_cache, self.context.table, n, min_p=min_p
        )
        M = envelope.M - EM * self.charge
        return IsotopicEnvelope(mz=M.tolist(), p=envelope.p.tolist())  # type: ignore

    def __repr__(self):
        return "Formula({})".format(str(self))

    def __str__(self):
        return _get_formula_str(self.composition, self.charge, self.context.table)


_matching_parenthesis = {"(": ")", "[": "]"}


def _multiply_formula_coefficients(composition: Counter, multiplier: int):
    if multiplier != 1:
        for k in composition:
            composition[k] *= multiplier


def _parse_charge(formula: str) -> tuple[str, int]:
    """Compute the charge state of a formula and remove the charge from the formula string.

    :param formula: molecular formula string

    Returns
    -------
    formula_without_charge, charge: str, int

    """
    # check if there's a charge in the formula
    if formula[-1] == "+":
        sign = 1
    elif formula[-1] == "-":
        sign = -1
    else:
        sign = 0

    # for charge with an absolute value greater than 1 enforces the use
    # of parenthesis to prevent ambiguity with a formula coefficient
    if sign and (formula[-2] in string.digits):
        try:
            matching = _matching_parenthesis[formula[0]]
            start = 1
            end = formula.rfind(matching)
            charge = sign * int(formula[end + 1 : -1])
            return formula[start:end], charge
        except (KeyError, ValueError):
            raise InvalidFormula
    elif sign:
        return formula[:-1], sign
    else:
        return formula, 0


def _get_token_type(formula: str, ind: int) -> int:
    """Assign 0 to elements, 1 to isotopes and 2 to expressions.

    Return the token type and a matching parenthesis if necessary
    """
    c = formula[ind]
    if c in string.ascii_uppercase:
        token_type = 0
    elif c in "(":
        if formula[ind + 1] in string.digits:
            token_type = 1
        else:
            token_type = 2
    elif c == "[":
        token_type = 2
    else:
        raise InvalidFormula
    return token_type


def _find_matching_parenthesis(formula: str, ind: int):
    parenthesis_open = formula[ind]
    parenthesis_close = _matching_parenthesis[parenthesis_open]
    match_ind = ind + 1
    level = 1
    try:
        while level > 0:
            c = formula[match_ind]
            if c == parenthesis_open:
                level += 1
            elif c == parenthesis_close:
                level -= 1
            match_ind += 1
        return match_ind - 1
    except IndexError:
        msg = "Formula string has non-matching parenthesis"
        raise InvalidFormula(msg)


def _get_coefficient(formula: str, ind: int):
    """Traverses a formula string to compute a coefficient.

    `ind` is a position after an element or expression.

    coefficient : int
    new_ind : int, new index to continue parsing the formula
    """
    length = len(formula)
    if (ind >= length) or (formula[ind] not in string.digits):
        coefficient = 1
        new_ind = ind
    else:
        end = ind + 1
        while (end < length) and (formula[end] in string.digits):
            end += 1
        coefficient = int(formula[ind:end])
        new_ind = end
    return coefficient, new_ind


def _tokenize_element(formula: str, ind: int, table: PeriodicTable):
    length = len(formula)
    if (ind < length - 1) and (formula[ind + 1] in string.ascii_lowercase):
        end = ind + 2
    else:
        end = ind + 1
    symbol = formula[ind:end]
    isotope = table.get_isotope(symbol)
    coefficient, end = _get_coefficient(formula, end)
    token = {isotope: coefficient}
    return token, end


def _tokenize_isotope(formula: str, ind: int, table: PeriodicTable):
    """Convert an isotope substring starting at `ind` index into a token.

    Returns
    -------
    token, new_ind

    """
    end = _find_matching_parenthesis(formula, ind)
    isotope = table.get_isotope(formula[ind + 1 : end])
    coefficient, end = _get_coefficient(formula, end + 1)
    token = {isotope: coefficient}
    return token, end


def _parse_formula(formula: str, table: PeriodicTable) -> Counter[Isotope]:
    """Parse a formula string into a Counter that maps isotopes to formula coefficients."""
    ind = 0
    n = len(formula)
    composition = Counter()
    while ind < n:
        token_type = _get_token_type(formula, ind)
        if token_type == 0:
            token, ind = _tokenize_element(formula, ind, table)
        elif token_type == 1:
            token, ind = _tokenize_isotope(formula, ind, table)
        else:
            # expression type evaluated recursively
            exp_end = _find_matching_parenthesis(formula, ind)
            token = _parse_formula(formula[ind + 1 : exp_end], table)
            exp_coefficient, ind = _get_coefficient(formula, exp_end + 1)
            _multiply_formula_coefficients(token, exp_coefficient)
        composition.update(token)
    return composition


# functions to get a formula string from a Formula


def _arg_sort_elements(symbol_list: list[str], mass_number_list: list[int]):
    """Return the sorted index for a list of elements symbols and mass numbers.

    If there are repeated elements, they are sorted by mass number.
    """
    zipped = list(zip(symbol_list, mass_number_list))
    return sorted(range(len(symbol_list)), key=lambda x: zipped[x])


def _get_formula_str(composition: Counter[Isotope], charge: int, table: PeriodicTable):
    # get C str
    c12 = table.get_isotope("12C")
    c13 = table.get_isotope("13C")
    h1 = table.get_isotope("1H")
    h2 = table.get_isotope("2H")
    ch = [c12, c13, h1, h2]

    # ADD C and H to formula str
    f_str = ""
    for i in ch:
        if i in composition:
            f_str += _isotope_coeff_to_f_str(i, composition[i], table.is_monoisotope(i))

    # add other elements, sorted alphabetically
    isotopes = set(composition)
    isotopes = isotopes.difference(ch)
    for i in sorted(isotopes, key=lambda x: x.symbol + str(x.a)):
        f_str += _isotope_coeff_to_f_str(i, composition[i], table.is_monoisotope(i))

    if charge:
        charge_str = _get_charge_str(charge)
        f_str = "[{}]{}".format(f_str, charge_str)

    return f_str


def _isotope_coeff_to_f_str(isotope: Isotope, coeff: int, is_monoisotope: bool) -> str:
    coeff_str = str(coeff) if coeff > 1 else ""
    symbol_str = isotope.symbol if is_monoisotope else f"({isotope.a}{isotope.symbol})"
    return f"{symbol_str}{coeff_str}"


def _get_charge_str(q: int) -> str:
    qa = abs(q)
    q_sign = "+" if q > 0 else "-"
    q_str = str(qa) if qa > 1 else ""
    q_str = q_str + q_sign
    return q_str


def _find_formula_envelope(
    composition: Counter[Isotope],
    cache: HomoAtomicEnvelopeCache,
    table: PeriodicTable,
    length: int,
    min_p: float = 1e-10,
) -> Envelope:
    """Compute the isotopic envelope for a formula."""
    # initialize an empty envelope for the formula
    env = cache.get_empty(length)

    for isotope, coeff in composition.items():
        if table.is_monoisotope(isotope):
            element = table.get_element(isotope.z)
            isotope_envelope = cache.get_envelope(element, coeff)
        else:
            isotope_envelope = cache.get_envelope(isotope, coeff)
        env = combine_envelopes(env, isotope_envelope)

    min_p_filter = env.p >= min_p
    env.p = env.p[min_p_filter].flatten()
    norm = env.p.sum()
    if norm > 0.0:
        env.p /= norm
    env.M = env.M[min_p_filter].flatten()
    return env
