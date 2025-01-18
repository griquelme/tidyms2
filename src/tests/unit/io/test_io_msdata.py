import pytest

from tidyms2.core.models import Sample
from tidyms2.io.msdata import MSData, MSDataCache
from tidyms2.io.mzml import MZMLReader


class TestMSDataFromSample:
    @pytest.fixture
    def sample(self, raw_data_dir):
        sample_path = raw_data_dir / "centroid-data-zlib-indexed-compressed.mzML"
        return Sample(path=sample_path, id="example")

    def test_create_from_sample(self, sample):
        ms_data = MSData(sample)
        n_sp = ms_data.get_n_spectra()
        assert n_sp > 0

    def test_with_start_time(self, sample: Sample):
        expected = 10.0
        sample = sample.model_copy(update={"start_time": expected})
        ms_data = MSData(sample)
        for sp in ms_data:
            assert sp.time >= expected

    def test_with_end_time(self, sample: Sample):
        expected = 10.0
        sample = sample.model_copy(update={"end_time": expected})
        ms_data = MSData(sample)
        for sp in ms_data:
            assert sp.time <= expected

    def test_with_ms_level(self, sample: Sample):
        sample = sample.model_copy(update={"ms_level": 1})
        ms_data = MSData(sample)
        spectra = [sp for sp in ms_data]
        assert spectra

    def test_iterate_ignore_all_spectra_if_no_spectra_with_ms_level(self, sample: Sample):
        sample = sample.model_copy(update={"ms_level": 10})
        ms_data = MSData(sample)
        spectra = [sp for sp in ms_data]
        assert not spectra


class TestMSDataCache:
    @pytest.fixture
    def sample_path(self, raw_data_dir):
        return raw_data_dir / "profile-data-zlib-indexed-compressed.mzML"

    def test_cache_add_data_no_limits(self, sample_path):
        reader = MZMLReader(sample_path)
        cache = MSDataCache()

        index = 5
        spectrum = reader.get_spectrum(index)
        cache.add(spectrum)
        assert cache.size == spectrum.get_nbytes()

        stored = cache.get(index)
        assert spectrum == stored

    def test_cache_with_limit_delete_entries(self, sample_path):
        reader = MZMLReader(sample_path)

        index1 = 0
        index2 = 1
        sp1 = reader.get_spectrum(index1)
        sp2 = reader.get_spectrum(index2)
        size1 = sp1.get_nbytes()
        size2 = sp2.get_nbytes()
        max_size = max(size1, size2)
        cache = MSDataCache(max_size=max_size)
        cache.add(sp1)
        cache.add(sp2)

        assert cache.check(index2)
        assert not cache.check(index1)
