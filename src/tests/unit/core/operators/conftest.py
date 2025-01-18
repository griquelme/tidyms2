import pytest

from tidyms2.storage.memory import OnMemorySampleStorage

from ... import helpers
from ...helpers import ConcreteFeature, ConcreteRoi


@pytest.fixture
def sample_storage(tmp_path) -> OnMemorySampleStorage[ConcreteRoi, ConcreteFeature]:
    sample = helpers.create_sample(tmp_path, 1)
    return OnMemorySampleStorage(sample, ConcreteRoi, ConcreteFeature)


@pytest.fixture
def sample_storage_with_rois(sample_storage):
    roi_extractor = helpers.DummyRoiExtractor()
    roi_extractor.apply(sample_storage)
    return sample_storage


@pytest.fixture
def sample_storage_with_features(sample_storage_with_rois):
    feature_extractor = helpers.DummyFeatureExtractor()
    feature_extractor.apply(sample_storage_with_rois)
    return sample_storage_with_rois
