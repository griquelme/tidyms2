"""Algorithms to convert MS spectra into centroid mode."""

from pydantic import BaseModel

from ..core.models import MSSpectrum
from .signal import find_centroids


class PeakPickingCentroider(BaseModel):
    """Create a centroiding function."""

    min_snr: float = 3.0
    """The minimum :term:`snr` in a m/z peak"""

    min_distance: float = 0.01
    """minimum m/z distance between consecutive centroids"""

    def __call__(self, sp: MSSpectrum) -> MSSpectrum:
        """Apply :py:func:`~tidyms2.signal.find_centroids` to the spectrum."""
        mz, spint = find_centroids(sp.mz, sp.int, self.min_snr, self.min_distance)
        return MSSpectrum(index=sp.index, mz=mz, int=spint, ms_level=sp.ms_level, time=sp.time)
