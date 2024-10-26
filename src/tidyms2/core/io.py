"""General purpose raw data reader."""

from __future__ import annotations

import pathlib
from collections import OrderedDict
from typing import Generator, Protocol, Self

from typing_extensions import Callable

from .enums import MSDataMode
from .models import Chromatogram, MSSpectrum, Sample
from .registry import Registry

reader_registry: Registry[Reader] = Registry("reader")


class MSData:
    """Provide access to raw MS data.

    Data is read from disk in a lazy manner and cached in memory.

    Parameters
    ----------
    path : pathlib.Path
        Raw data source.
    reader : Reader or None, default=None
        The Reader class to read raw data. If ``None``, the reader is inferred using the file extension.
    centroid : MSDataMode, default=MSDataMode.CENTROID
        The mode in which the data is stored.
    centroider : Callable[[MSSpectrum], MSSpectrum] or None, default=None
        A function that takes a spectrum in profile mode and converts it to centroid mode.
    cache : int, default=-1
        The maximum cache size, in bytes. The cache will store spectrum data until it surpasses this value. At this
        point, old entries will be deleted from the cache. If set to``-1``, the cache can grow indefinitely. When
        working with data streams, the cache size is set to ``-1``, as there is no guarantee that the `seek` method,
        which is required for repeated access to the file object, is implemented.
    ms_level : int = 1
            Skip spectra without this MS level.
    start_time: float = 0.0
        Skip spectra with time lower than this value.
    end_time: float or None, default=None
        Skip spectra with time greater than this value.

    """

    def __init__(
        self,
        path: pathlib.Path,
        reader: type[Reader] | None = None,
        centroid: MSDataMode = MSDataMode.CENTROID,
        centroider: Callable[[MSSpectrum], MSSpectrum] | None = None,
        cache: int = -1,
        ms_level: int = 1,
        start_time: float = 0.0,
        end_time: float | None = None,
    ):
        if reader is None:
            reader = reader_registry.get(path.suffix[1:])

        self.ms_level = ms_level
        self.start_time = start_time
        self.end_time = end_time

        self._centroider = centroider
        self._reader = reader(path)
        self._centroid = centroid
        self._cache = MSDataCache(max_size=cache)
        self._n_spectra: int | None = None
        self._n_chromatogram: int | None = None

    def get_n_chromatograms(self) -> int:
        """Retrieve the total number of chromatograms stored in the source."""
        if self._n_chromatogram is None:
            self._n_chromatogram = self._reader.get_n_chromatograms()
        return self._n_chromatogram

    def get_n_spectra(self) -> int:
        """Retrieve the total number of spectra stored in the source."""
        if self._n_spectra is None:
            self._n_spectra = self._reader.get_n_spectra()
        return self._n_spectra

    def get_chromatogram(self, index: int) -> Chromatogram:
        """Retrieve a chromatogram by index."""
        return self._reader.get_chromatogram(index)

    def get_spectrum(self, index: int) -> MSSpectrum:
        """Retrieve a spectrum by index."""
        n_sp = self.get_n_spectra()
        if (index < 0) or (index >= n_sp):
            msg = f"`index` must be integer in the interval [0:{n_sp}). Got {index}."
            raise ValueError(msg)
        spectrum = self._cache.get(index)
        if spectrum is None:
            spectrum = self._reader.get_spectrum(index)
            if self._centroid == MSDataMode.PROFILE and self._centroider is not None:
                spectrum = self._centroider(spectrum)

            spectrum.centroid = self._centroid == MSDataMode.CENTROID
            self._cache.add(spectrum)
        return spectrum

    def __iter__(self) -> Generator[MSSpectrum, None, None]:
        """Iterate over all spectra in the data."""
        for k in range(self.get_n_spectra()):
            sp = self.get_spectrum(k)
            if (self.ms_level == sp.ms_level) and (self.start_time <= sp.time):
                if (self.end_time is None) or (self.end_time > sp.time):
                    yield sp

    @classmethod
    def from_sample(cls, sample: Sample) -> Self:
        """Create a new instance using sample information."""
        params = sample.model_dump(exclude={"id", "group", "order", "batch", "extra", "reader"})
        return cls(**params)


class MSDataCache:
    """Cache spectra data to avoid reading from disk.

    Old entries are deleted if the cache grows larger than total data size in bytes. The maximum size of the cache is
    defined by `max_size`. If set to ``-1``, the cache can grow indefinitely.

    """

    def __init__(self, max_size: int = -1):
        self.cache: OrderedDict[int, MSSpectrum] = OrderedDict()
        self.size = 0
        self.max_size = max_size

    def add(self, spectrum: MSSpectrum) -> None:
        """Store a spectrum."""
        self.cache[spectrum.index] = spectrum
        self.size += spectrum.get_nbytes()
        self.trim_cache()

    def get(self, index: int) -> MSSpectrum | None:
        """Retrieve a spectrum from the cache. If not found, returns ``None``."""
        spectrum = self.cache.get(index)
        if isinstance(spectrum, MSSpectrum):
            self.cache.move_to_end(index)
        return spectrum

    def trim_cache(self) -> None:
        """Delete entries until the cache size is lower than max_size."""
        if self.max_size > -1:
            while self.size > self.max_size:
                _, spectrum = self.cache.popitem(last=False)
                self.size -= spectrum.get_nbytes()


class Reader(Protocol):
    """Reader interface for raw data."""

    def __init__(self, src: pathlib.Path): ...

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
