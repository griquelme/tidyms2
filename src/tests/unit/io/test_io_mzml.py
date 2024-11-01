"""Test lcms and fileio functionality with real data."""

import pathlib

import pytest

from tidyms2.core.models import Chromatogram, MSSpectrum
from tidyms2.io import MSData


@pytest.mark.parametrize(
    "filename",
    [
        "centroid-data-indexed-uncompressed.mzML",
        "centroid-data-zlib-no-index-compressed.mzML",
        "centroid-data-zlib-indexed-compressed.mzML",
        "profile-data-zlib-indexed-compressed.mzML",
    ],
)
class TestMZMLReader:
    def test_get_n_spectra(self, raw_data_dir: pathlib.Path, filename: str):
        ms_data = MSData(raw_data_dir / filename)
        n = ms_data.get_n_spectra()
        assert isinstance(n, int)
        assert n > 0

    def test_get_n_chromatograms(self, raw_data_dir: pathlib.Path, filename: str):
        ms_data = MSData(raw_data_dir / filename)
        n = ms_data.get_n_chromatograms()
        assert isinstance(n, int)
        assert n > 0

    def test_get_spectrum(self, raw_data_dir: pathlib.Path, filename: str):
        ms_data = MSData(raw_data_dir / filename)
        sp = ms_data.get_spectrum(0)
        assert isinstance(sp, MSSpectrum)

    def test_get_all_spectra(self, raw_data_dir: pathlib.Path, filename: str):
        ms_data = MSData(raw_data_dir / filename)
        expected_n_spectra = ms_data.get_n_spectra()
        all_spectra = [ms_data.get_spectrum(x) for x in range(expected_n_spectra)]

        assert len(all_spectra) == expected_n_spectra
        assert all(isinstance(x, MSSpectrum) for x in all_spectra)

    def test_get_all_spectra_rebuild_index(self, raw_data_dir: pathlib.Path, filename: str):
        ms_data = MSData(raw_data_dir / filename, rebuild_index=True)
        expected_n_spectra = ms_data.get_n_spectra()
        all_spectra = [ms_data.get_spectrum(x) for x in range(expected_n_spectra)]

        assert len(all_spectra) == expected_n_spectra
        assert all(isinstance(x, MSSpectrum) for x in all_spectra)

    def test_get_chromatogram(self, raw_data_dir: pathlib.Path, filename: str):
        ms_data = MSData(raw_data_dir / filename)
        chromatogram = ms_data.get_chromatogram(0)
        assert isinstance(chromatogram, Chromatogram)

    def test_get_all_chromatograms(self, raw_data_dir: pathlib.Path, filename: str):
        ms_data = MSData(raw_data_dir / filename)
        expected_n_chromatograms = ms_data.get_n_chromatograms()
        all_spectra = [ms_data.get_chromatogram(x) for x in range(expected_n_chromatograms)]

        assert len(all_spectra) == expected_n_chromatograms
        assert all(isinstance(x, Chromatogram) for x in all_spectra)
