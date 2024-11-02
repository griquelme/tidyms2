"""Periodic table implementation."""

import json
import pathlib
from string import digits

from .atoms import Element, Isotope
from .exceptions import IsotopeError


class PeriodicTable:
    """Store information from elements and their isotopes.

    :param custom_abundances: override natural abundances of isotopes.
        Custom abundances is a dictionary that maps atomic numbers to a dictionary
        of isotopes mass numbers and their abundance. If an abundance is not
        provided for a particular isotope it is assumed to be zero. Abundances from
        a set of isotopes must be normalized to one.

    """

    def __init__(self, custom_abundances: dict[str, dict[int, float]] | None = None):
        self._symbol_to_element = _make_periodic_table(custom_abundances)
        self._z_to_element = {v.z: v for v in self._symbol_to_element.values()}
        self._za_to_isotope = dict()
        self._str_to_isotope = dict()
        for el_str in self._symbol_to_element:
            el = self._symbol_to_element[el_str]
            for isotope in el.isotopes:
                self._za_to_isotope[(isotope.z, isotope.a)] = isotope
                self._str_to_isotope[str(isotope.a) + el_str] = isotope

    def get_element(self, element: str | int) -> Element:
        """Fetch an element object using its symbol or atomic number.

        >>> import tidyms as ms
        >>> ptable = ms.chem.PeriodicTable()
        >>> h = ptable.get_element("H")
        >>> c = ptable.get_element(6)

        """
        if isinstance(element, int):
            res = self._z_to_element[element]
        else:
            res = self._symbol_to_element[element]
        return res

    def __iter__(self):
        for el in self._symbol_to_element.values():
            yield el

    def get_isotope(self, symbol: str, copy: bool = False) -> Isotope:
        """Fetch an isotope object from a string representation.

        :param symbol: a string representation of the isotope. If only the symbol
            is provided in the string, the monoisotope is returned.
        :param copy: If set to ``True`` a new isotope instance is created.

        >>> import tidyms as ms
        >>> ptable = ms.chem.PeriodicTable()
        >>> d = ptable.get_isotope("2H")
        >>> cl35 = ptable.get_isotope("Cl")

        """
        try:
            if symbol[0] in digits:
                isotope = self._str_to_isotope[symbol]
            else:
                isotope = self.get_element(symbol).monoisotope

            if copy:
                isotope = isotope.model_copy()

            return isotope
        except KeyError as e:
            msg = f"{symbol} is not a valid element or isotope symbol."
            raise IsotopeError(msg) from e

    def is_monoisotope(self, isotope: Isotope) -> bool:
        """Check if an isotope is the most abundant."""
        element = self.get_element(isotope.symbol)
        return element.monoisotope.a == isotope.a


def _make_periodic_table(custom_abundances: dict[str, dict[int, float]] | None = None) -> dict[str, Element]:
    data_dir = pathlib.Path(__file__).parent
    elements_file = data_dir / "elements.json"
    elements_dict = json.loads(elements_file.read_text())

    isotopes_file = data_dir / "isotopes.json"
    isotopes_dict = json.loads(isotopes_file.read_text())

    if custom_abundances is not None:
        _patch_isotope_abundances(isotopes_dict, custom_abundances)

    periodic_table = dict()
    for symbol, isotopes_list in isotopes_dict.items():
        name = elements_dict[symbol]
        isotopes = tuple(Isotope(symbol=symbol, **x) for x in isotopes_list)
        z = isotopes_list[0]["z"]
        element = Element(symbol=symbol, name=name, z=z, isotopes=isotopes)
        periodic_table[symbol] = element

    return periodic_table


def _patch_isotope_abundances(isotopes: dict[str, list[dict]], custom_abundances: dict[str, dict[int, float]]) -> None:
    for symbol, element_isotopes in isotopes.items():
        isotope_patches = custom_abundances.get(symbol)
        if isotope_patches is None:
            continue
        for i in element_isotopes:
            i["p"] = isotope_patches.get(i["a"], 0.0)


PTABLE = PeriodicTable()
