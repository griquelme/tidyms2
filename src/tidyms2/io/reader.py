"""Raw data reader interface."""

from __future__ import annotations

import pathlib
from typing import Protocol

from ..core.models import Chromatogram, MSSpectrum, Sample
from ..core.registry import Registry

reader_registry: Registry[Reader] = Registry("reader")


class Reader(Protocol):
    """Reader interface for raw data."""

    def __init__(self, src: pathlib.Path | Sample): ...

    def get_chromatogram(self, index: int) -> Chromatogram:
        """Retrieve a chromatogram from file."""
        ...

    def get_spectrum(self, index: int) -> MSSpectrum:
        """Retrieve a spectrum from file."""
        ...

    def get_n_chromatograms(self) -> int:
        """Retrieve the total number of chromatogram."""
        ...

    def get_n_spectra(self) -> int:
        """Retrieve the total number of spectra."""
        ...
