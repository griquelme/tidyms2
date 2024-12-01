.. _chemistry-user-guide:

Working with chemical data
==========================

The `chem` package provide utilities to work with chemical data such as isotopes, elements and
formulas. It also allows to generate formulas from exact mass, score isotopic envelopes and search
isotopic envelope candidates from a list of m/z values.

Elements and isotopes
---------------------

:class:`~tidyms2.chem.PeriodicTable` contains element and isotope information, retrieved from the
`Blue Obelisk Data Repository <https://github.com/BlueObelisk/bodr>`_. The
:py:func:`~tidyms2.chem.PeriodicTable.get_element` method returns a :class:`~tidyms2.chem.Element` instance:

.. code-block:: python

    from tidyms2.chem import PeriodicTable
    ptable = PeriodicTable()
    oxygen = ptable.get_element("O")
    print(oxygen)
    # Element(O, z=8)

An element contain basic element and isotope information:

.. code-block:: python

    print(oxygen.z)
    # 8
    print(oxygen.name)
    # "Oxygen"
    print(oxygen.isotopes)
    # (Isotope(z=8, a=16, symbol='O'), Isotope(z=8, a=17, symbol='O'), Isotope(z=8, a=18, symbol='O'))
    print(oxygen.monoisotope)
    # Isotope(z=8, a=16, symbol='O')
    print(oxygen.get_abundances())
    # ([16, 17, 18], [15.9949, 16.9991, 17.9991], [0.9976, 0.0004, 0.0020])

:class:`~tidyms2.chem.Isotope` store exact mass, nominal mass and abundance of each isotope:

.. code-block:: python

    o16 = oxygen.monoisotope
    print(o16)
    # z=8 a=16 symbol='O'


.. _working-with-formulas-guide:

Chemical formulas
-----------------

The :class:`~tidyms2.chem.Formula` class allows to create chemical formulas:

.. code-block:: python

    from tidyms2.chem import Formula
    water = Formula("H2O")
    print(water)
    # H2O

Formula objects can be used to compute a formula mass and its isotopic envelope:

.. code-block:: python

    print(water.get_exact_mass())
    # 18.010564684
    envelope = water.get_isotopic_envelope()
    print(envelope.mz)
    # [18.01056468, 19.01555724, 20.01481138, 21.02108788]
    print(envelope.p)
    # [9.97340572e-01, 6.09327319e-04, 2.04962911e-03, 4.71450803e-07]

Formulas can be created by passing a dictionary of element or isotopes to a formula coefficient
and the numerical charge of the formula. Formulas are implemented as dictionaries of isotopes
to formula coefficients, so if an element is passed, it is assumed that it is the most abundant
isotope.

.. code-block:: python

    f = Formula({"C": 1, "13C": 1, "O": 4}, 0)
    print(f)
    # C(13C)O4

Isotopes can also be specified in the string format:

.. code-block:: python

    f = Formula("[C(13C)2H2O4]2-")
    print(f)
    # [C(13C)2H2O4]2-
    print(f.charge)
    # -2


.. _generating-formulas-guide:

Sum formula generation
----------------------

The :class:`~tidyms2.chem.FormulaGenerator` generates sum formulas from a mass value. To generate
formulas, the formula space must be defined by passing a dictionary of elements and their associated
minimum and maximum allowed formula coefficients:

.. code-block:: python

    from tidyms2.chem import Formula, FormulaGenerator, FormulaGeneratorConfiguration

    config = FormulaGeneratorConfiguration(
        bounds={"C": (0, 20), "H": (0, 40), "O": (0, 10), "N": (0, 5)},
        max_M=1000.0,
    )
    formula_generator = FormulaGenerator(config)

To generate formulas, an exact mass value must be passed, along with a tolerance to find compatible
formulas. In the following code example, first a exact mass value is computed from a formula and
then compatible formulas are generated:

.. code-block:: python

    f = Formula("C5H10O2")
    M = f.get_exact_mass()
    tolerance = 0.005
    formula_generator.generate_formulas(M, tolerance)
    coefficients, isotopes, M_coeff = formula_generator.results_to_array()

    print(coefficients)
    # [[ 0 10  2  4]
    #  [ 3  8  3  1]
    #  [ 5 10  0  2]]
    print(isotopes)
    # [
    #   Isotope(z=6, a=12, symbol='C'),
    #   Isotope(z=1, a=1, symbol='H'),
    #   Isotope(z=7, a=14, symbol='N'),
    #   Isotope(z=8, a=16, symbol='O')
    # ]

`coefficients` is a 2D Numpy array where each row are matching formulas coefficients. The isotope associated
with each coefficient column are stored in `isotopes`. Finally, a third Numpy array stores the exact mass of
each matching formula.

The :py:func:`~tidyms2.chem.FormulaGeneratorConfiguration.from_chnops` provides an simple way to create
pre-configured formula generators:

.. code-block:: python

    config = FormulaGeneratorConfiguration.from_chnops(1000)
    formula_generator = FormulaGenerator(config)


this method generates a formula space for the CHNOPS elements by finding the maximum formula coefficients of
molecules in the `Human Metabolome DataBase <https://hmdb.ca>`_. Precomputed formula bounds are available
for molecules with maximum mass values of 500, 1000, 1500 and 2000. Other element can be added using the
:py:func:`~tidyms2.chem.FormulaGeneratorConfiguration.update_bounds` method:

.. code-block:: python

    config = FormulaGeneratorConfiguration.from_chnops(1000)
    config.update_bounds({"Cl": (0, 2)})
    formula_generator = FormulaGenerator(config)


.. _scoring-formulas-guide:

Scoring Isotopic envelopes
--------------------------

Scoring measured envelopes against theoretical values is a common strategy to establish a formula
candidate for an unknown compound. The :class:`~tidyms2.chem.EnvelopeScorer` ranks compatible
formulas based on the similarity with the measured envelope. As the envelope scorer uses a formula
generator to generate compatible formulas, we need to provide the formula bounds. In the same
way as the formula generator, the :py:func:`tidyms2.chem.EnvelopeScorerConfiguration.from_hmdb` provides
an easy way to create a configuration:

.. code-block:: python

    from tidyms2.chem import EnvelopeScorer, EnvelopeScorerConfiguration, Formula

    config = EnvelopeScorerConfiguration.from_chnops(500, max_length=5)
    envelope_scorer = EnvelopeScorer(config)

The `max_length` parameter sets the maximum length of the measured envelopes to compare against theoretical
values. The :py:func:`~tidyms2.chem.EnvelopeScorer.score` method takes a list of exact mass and abundances
corresponding to the measured isotopic envelope and scores against all compatible formulas. The results can
be obtained with the :meth:`~tidyms2.chem.EnvelopeScorer.get_top_results` method. The following example
uses the envelope of a known formula and scores compatible formulas with it:

.. code-block:: python

    f = Formula("C5H10O2")
    envelope = f.get_isotopic_envelope(5)
    mass_tolerance = 0.005

    envelope_scorer.score(envelope.mz, envelope.p, mass_tolerance)

    coeff, isotopes, score = envelope_scorer.get_top_results(10)
    print(coeff)
    # [[ 5 10  0  2  0  0]
    #  [ 3  8  3  1  0  0]
    #  [ 1 13  1  2  1  0]
    #  [ 0 10  2  4  0  0]
    #  [ 2 16  0  0  2  0]
    #  [ 2 14  0  2  0  1]
    #  [ 2 15  0  0  1  1]
    #  [ 1  6  6  0  0  0]
    #  [ 0 12  3  1  0  1]]
    print(isotopes)
    # [
    #   Isotope(z=6, a=12, symbol='C'),
    #   Isotope(z=1, a=1, symbol='H'),
    #   Isotope(z=7, a=14, symbol='N'),
    #   Isotope(z=8, a=16, symbol='O'),
    #   Isotope(z=15, a=31, symbol='P'),
    #   Isotope(z=16, a=32, symbol='S')
    # ]
    print(score)
    # [1.000 0.356 0.140  0.0568 0.0376 0.006 0.004 0.004 0.002]

By default, the :py:func:`tidyms2.chem.score_envelope` function is used. The parameters used by
this function can be modified by passing them to the envelope scorer constructor as keyword arguments.
It is also possible to use a custom scorer. Refer to :class:`~tidyms2.chem.EnvelopeScorer` API docs
for details on this.




