"""Utilities for working with chemical entities.

Refer to the :ref:`use guide <chemistry-user-guide>` for an introduction an examples on how to
use this package.

Constants
---------
- EM : the electron mass

"""

from .atoms import EM, Element, Isotope
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
    "Element",
    "EnvelopeScorerConfiguration",
    "EnvelopeValidatorConfiguration",
    "Formula",
    "FormulaGenerator",
    "FormulaGeneratorConfiguration",
    "Isotope",
    "PeriodicTable",
    "EnvelopeScorer",
    "EnvelopeValidator",
    "score_envelope",
]
