"""Set the current chemical context."""

from .config import ChemicalContextConfiguration
from .envelope_utils import HomoAtomicEnvelopeCache
from .table import PTABLE, PeriodicTable


class ChemicalContext:
    """Centralize chemical information used in the chem package."""

    def __init__(self, config: ChemicalContextConfiguration):
        self.envelope_cache = HomoAtomicEnvelopeCache()
        if config.custom_abundances is None:
            table = PTABLE
        else:
            table = PeriodicTable(custom_abundances=config.custom_abundances)
        self.table = table


DEFAULT_CONTEXT = ChemicalContext(ChemicalContextConfiguration())
