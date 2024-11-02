"""Chemistry utilities.

Provides:

- a formula object to compute the exact mass and isotopic distribution of molecular formulas.
- a periodic table with element and isotope information.
- a formula generator object to search molecular formulas based on exact mass values.
- an envelope scorer that scores the similarity between experimental and theoretical isotopic envelopes.
- an envelope validator that checks if a measured envelope may be generated from a combination of elements.

Refer to the :ref:`chemistry use guide <chemistry-user-guide>` for an introduction an examples on how to
use this package.

Constants
---------
- EM : the electron mass
- PTABLE : a periodic table instance.

"""

from .atoms import EM
from .config import EnvelopeScorerConfiguration, EnvelopeValidatorConfiguration, FormulaGeneratorConfiguration
from .context import DEFAULT_CONTEXT, ChemicalContext
from .envelope import EnvelopeScorer, EnvelopeValidator, score_envelope
from .formula import Formula
from .formula_generator import FormulaGenerator
from .table import PTABLE, PeriodicTable

__all__ = [
    "DEFAULT_CONTEXT",
    "EM",
    "PTABLE",
    "ChemicalContext",
    "EnvelopeScorerConfiguration",
    "EnvelopeValidatorConfiguration",
    "Formula",
    "FormulaGenerator",
    "FormulaGeneratorConfiguration",
    "PeriodicTable",
    "EnvelopeScorer",
    "EnvelopeValidator",
    "score_envelope",
]
