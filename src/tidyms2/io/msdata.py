"""General purpose raw data reader."""

from __future__ import annotations

import pathlib
from collections import OrderedDict
from contextlib import contextmanager
from typing import Generator

from typing_extensions import Callable

from ..core.enums import MSDataMode
from ..core.models import Chromatogram, MSSpectrum, Sample
from .reader import Reader, reader_registry


class MSData:
    """Provide access to raw MS data.

    Data is read from disk in a lazy manner and cached in memory.

    :param src: raw data source. It may be the path to a raw data file or a sample model.
        If the latter is provided, the path to the data source is fetch from the
        :py:attr:`tidyms2.core.models.Sample.path` field.
    :param reader: the Reader to read raw data. If ``None``, the reader is inferred using the file extension.
        If `src` is a sample model and :py:attr:`~tidyms2.core.models.Sample.reader` is defined, the reader will
        be fetch fetched from the reader registry.
    :param mode: the mode in which the data is stored. If `src` is a sample instance, this parameter is ignored
        and is fetched from the sample data.
    :param centroider: a function that takes a spectrum in profile mode and converts it to centroid mode. Only
        used if :py:attr:`mode` is set to profile mode.
    :param cache: int, default=-1
        The maximum cache size, in bytes. The cache will store spectrum data until it surpasses this value. At this
        point, old entries will be deleted from the cache. If set to``-1``, the cache can grow indefinitely.
    :param ms_level: skip spectra without this MS level when iterating over spectra.  If `src` is a sample instance,
        this parameter is ignored and is fetched from the sample data.
    :param start_time: skip spectra with time lower than this value when iterating over data.  If `src` is a sample
        instance, this parameter is ignored and is fetched from the sample data.
    :param end_time: skip spectra with time greater than this value when iterating over data.  If `src` is a sample
        instance, this parameter is ignored and is fetched from the sample data.
    :param kwargs: keyword arguments passed to the reader.

    """

    def __init__(
        self,
        src: pathlib.Path | Sample,
        reader: type[Reader] | None = None,
        mode: MSDataMode = MSDataMode.CENTROID,
        centroider: Callable[[MSSpectrum], MSSpectrum] | None = None,
        cache: int = -1,
        ms_level: int = 1,
        start_time: float = 0.0,
        end_time: float | None = None,
        **kwargs,
    ):
        if isinstance(src, pathlib.Path):
            id_ = src.name
            src = Sample(
                id=id_, path=src, ms_level=ms_level, start_time=start_time, end_time=end_time, ms_data_mode=mode
            )
        self.sample = src

        if reader is None and isinstance(self.sample.reader, str):
            reader = reader_registry.get(self.sample.reader)
        elif reader is None:
            reader = reader_registry.get(self.sample.path.suffix[1:])

        self.centroider = centroider
        self._reader = reader(self.sample, **kwargs)
        self._cache = MSDataCache(max_size=cache)
        self._n_spectra: int | None = None
        self._n_chromatogram: int | None = None

    def get_sample(self) -> Sample:
        """Retrieve the sample associated with the data."""
        return self.sample

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

        if self._cache.check(index):
            spectrum = self._cache.get(index)
        else:
            spectrum = self._reader.get_spectrum(index)
            if self.sample.ms_data_mode is MSDataMode.PROFILE and self.centroider is not None:
                spectrum = self.centroider(spectrum)
            spectrum.centroid = self.sample.ms_data_mode is MSDataMode.CENTROID
            self._cache.add(spectrum)
        return spectrum

    def __iter__(self) -> Generator[MSSpectrum, None, None]:
        """Iterate over all spectra in the data."""
        for k in range(self.get_n_spectra()):
            sp = self.get_spectrum(k)
            if (self.sample.ms_level == sp.ms_level) and (self.sample.start_time <= sp.time):
                if (self.sample.end_time is None) or (self.sample.end_time > sp.time):
                    yield sp

    @contextmanager
    def using_tmp_config(
        self, ms_level: int | None = None, start_time: float | None = None, end_time: float | None = None
    ):
        """Context manager that temporarily modifies MS level and scans time range.

        :param ms_level: temporary value for the MS level. If set to ``None`` the original value is not modified.
        :param start_time: temporary value for the start time. If set to ``None`` the original value is not modified.
        :param end_time: temporary value for the end time. If set to ``None`` the original value is not modified.

        """
        sample = self.get_sample()
        tmp_config = {
            "ms_level": ms_level or sample.ms_level,
            "start_time": start_time or sample.start_time,
            "end_time": end_time or sample.end_time,
        }
        sample_with_tmp_config = sample.model_copy(update=tmp_config)
        Sample.model_validate(sample_with_tmp_config)
        try:
            self.sample = sample_with_tmp_config
            yield
        finally:
            self.sample = sample


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
        self._prune()

    def get(self, index: int) -> MSSpectrum:
        """Retrieve a spectrum from the cache. If not found, returns ``None``."""
        spectrum = self.cache[index]
        self.cache.move_to_end(index)
        return spectrum

    def check(self, index: int) -> bool:
        """Check if the provided index is in the cache."""
        return index in self.cache

    def _prune(self) -> None:
        """Delete entries until the cache size is lower than max_size."""
        if self.max_size > -1:
            while self.size > self.max_size:
                _, spectrum = self.cache.popitem(last=False)
                self.size -= spectrum.get_nbytes()
