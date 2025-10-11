import pytest

from tidyms2.algorithms.centroiders import PeakPickingCentroider
from tidyms2.core.enums import MSDataMode
from tidyms2.core.models import Sample
from tidyms2.io import MSData


class TestMSDataCentroider:
    @pytest.fixture
    def sample(self, raw_data_dir):
        sample_path = raw_data_dir / "profile-data-zlib-indexed-compressed.mzML"
        return Sample(path=sample_path, id="example", ms_data_mode=MSDataMode.PROFILE)

    def test_ms_data_with_centoider(self, sample):
        centroider = PeakPickingCentroider()
        ms_data = MSData(sample, centroider=centroider)
        ms_data.get_spectrum(3)
