from __future__ import annotations

from uuid import UUID

import pytest

from tidyms2.core import models

from ..helpers import ConcreteFeature, ConcreteRoi, create_sample


@pytest.fixture
def expected_descriptor_names():
    return ["area", "height", "custom_descriptor", "mz"]


class TestRoi:
    @pytest.fixture(scope="class")
    def sample(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("tmp")
        return create_sample(tmp_path, 1)

    def test_id(self, sample):
        roi = ConcreteRoi(sample=sample)
        assert isinstance(roi.id, UUID)

    def test_serialization_deserialization_returns_equal_roi(self, sample):
        expected = ConcreteRoi(data=[1.0, 2.0], sample=sample)
        serialized = expected.to_str()
        actual = ConcreteRoi.from_str(serialized, sample=sample)

        assert actual.data == expected.data
        assert actual.id == expected.id


class TestFeature:
    @pytest.fixture(scope="class")
    def sample(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("tmp")
        return create_sample(tmp_path, 1)

    @pytest.fixture(scope="class")
    def roi(self, sample) -> ConcreteRoi:
        return ConcreteRoi(data=[1.0, 2.0], sample=sample)

    @pytest.fixture(scope="class")
    def feature(self, roi) -> ConcreteFeature:
        return ConcreteFeature(data_mz=1, roi=roi)

    def test_serialization(self, feature):
        serialized = feature.to_str()
        expected = ConcreteFeature.from_str(serialized, feature.roi, feature.annotation)
        assert expected.data_mz == feature.data_mz
        assert expected.roi == feature.roi
        assert expected.annotation is not None
        assert expected.annotation.id == feature.annotation.id

    def test_mz_equals_get_mz(self, feature):
        assert feature.get("mz") == feature.mz

    def test_area_equals_get_area(self, feature):
        assert feature.get("area") == feature.area

    def test_height_equals_get_height(self, feature):
        assert feature.get("height") == feature.height

    def test_custom_descriptor_equals_get_custom_descriptor(self, feature):
        assert feature.get("custom_descriptor") == feature.custom_descriptor

    def test_descriptor_names_are_feature_attributes(self, feature):
        all_descriptors = feature.descriptor_names()
        all_attr = feature.__dict__
        assert all(x in all_attr for x in all_descriptors)

    def test_describe(self, feature):
        descriptors = feature.describe()
        all_attr = feature.__dict__
        assert all(x in all_attr for x in descriptors)
        assert all(isinstance(x, float) for x in descriptors.values())
        assert feature.mz == descriptors["mz"]
        assert feature.height == descriptors["height"]
        assert feature.area == descriptors["area"]

    def test_is_feature_descriptor_in_valid_range_valid_range(self, feature):
        name = "height"
        value = feature.get(name)
        bounds = {name: (value - 5, value + 5)}
        assert feature.has_descriptors_in_range(**bounds)

    def test_is_feature_descriptor_in_valid_range_invalid_range(self, feature):
        name = "height"
        value = feature.get(name)
        bounds = {name: (value + 5, value + 10)}
        assert not feature.has_descriptors_in_range(**bounds)

    @pytest.fixture(scope="class")
    def ft1(self, roi):
        return ConcreteFeature(roi=roi, data_mz=1)

    @pytest.fixture(scope="class")
    def ft2(self, roi):
        return ConcreteFeature(roi=roi, data_mz=2)

    def test_order_lt(self, ft1, ft2):
        assert ft1 < ft2

    def test_le(self, ft1, ft2):
        assert ft1 <= ft2

    def test_order_ge(self, ft1):
        assert ft1 >= ft1

    def test_order_gt(self, ft1, ft2):
        assert ft2 > ft1

    def test_hash_is_equal_to_id_hash(self, ft1):
        assert hash(ft1) == hash(ft1.id)

    def test_get_invalid_descriptor_raises_error(self, ft1):
        with pytest.raises(ValueError):
            ft1.get("invalid_descriptor")


class TestSample:
    def test_serialization(self, tmp_path):
        sample_id = "my-sample"
        path = tmp_path / sample_id
        expected = models.Sample(id=sample_id, path=path)
        actual = models.Sample(**expected.model_dump())
        assert actual == expected

    def test_serialization_with_extra(self, tmp_path):
        sample_id = "my-sample"
        path = tmp_path / sample_id
        extra = {"extra-field-1": 0.25, "extra-field-2": 3, "extra-field-3": "extra"}
        expected = models.Sample(id=sample_id, path=path, meta=extra)  # type: ignore
        actual = models.Sample(**expected.model_dump())
        assert actual == expected
