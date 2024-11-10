import pytest

from tidyms2.core.models import Sample
from tidyms2.lcms import operators
from tidyms2.lcms.models import MZTrace, Peak
from tidyms2.lcms.simulation import SimulatedLCMSSample
from tidyms2.storage.memory import OnMemorySampleStorage


@pytest.fixture
def sample(lcms_sample_factory) -> Sample:
    return lcms_sample_factory(id="sample")


@pytest.fixture
def n_features(sample: Sample) -> int:
    assert sample.extra is not None
    simulated_sample_spec = SimulatedLCMSSample(**sample.extra)
    return len(simulated_sample_spec.features)


@pytest.fixture
def sample_storage(sample) -> OnMemorySampleStorage[MZTrace, Peak]:
    return OnMemorySampleStorage(sample, MZTrace, Peak)


def test_mz_trace_extractor(sample_storage: OnMemorySampleStorage, n_features: int):
    op = operators.LCTraceExtractor(id="mz-trace-extractor")
    assert not sample_storage.get_n_rois()
    op.apply(sample_storage)
    assert sample_storage.get_n_rois() == n_features


def test_peak_extractor(sample_storage: OnMemorySampleStorage, n_features: int):
    # perform roi extraction
    operators.LCTraceExtractor(id="mz-trace-extractor").apply(sample_storage)

    op = operators.PeakExtractor(id="peak-extractor")
    assert not sample_storage.get_n_features()
    op.apply(sample_storage)
    assert sample_storage.get_n_rois() == n_features
