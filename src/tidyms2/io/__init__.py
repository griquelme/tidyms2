"""I/O utilities."""

from .msdata import MSData

# load MZMLReader into the registry
from .mzml import MZMLReader  # noqa

__all__ = ["MSData"]
