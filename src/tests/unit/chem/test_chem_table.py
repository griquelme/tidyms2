from tidyms2.chem import PTABLE


class TestPeriodicTable:
    def test_get_element_using_symbol(self):
        c = PTABLE.get_element("C")
        assert c.z == 6
        assert c.symbol == "C"

    def test_get_element_using_z(self):
        p = PTABLE.get_element(15)
        assert p.symbol == "P"
        assert p.z == 15

    def test_get_isotope_using_symbol(self):
        cl37 = PTABLE.get_isotope("37Cl")
        assert cl37.a == 37
        assert cl37.symbol == "Cl"

    def test_get_isotope_return_an_isotope_copy(self):
        isotope_str = "37Cl"
        cl37_copy = PTABLE.get_isotope(isotope_str, copy=True)
        cl37 = PTABLE.get_isotope(isotope_str)
        assert cl37.a == cl37_copy.a
        assert cl37.m == cl37_copy.m
        assert cl37.z == cl37_copy.z
        assert cl37 is not cl37_copy


def test_Element_get_monoisotope():
    element = PTABLE.get_element("B")
    monoisotope = element.monoisotope
    assert monoisotope.a == 11


def test_Element_get_mmi():
    element = PTABLE.get_element("B")
    mmi = element.mmi
    assert mmi.a == 10
