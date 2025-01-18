"""Abstract sample operators."""

from __future__ import annotations

from abc import abstractmethod
from math import inf
from typing import Generic, Literal

import pydantic

from ..dataflow import SampleProcessStatus
from ..enums import OperatorType
from ..models import FeatureType, RoiType, Sample
from ..storage import SampleStorage
from .base import BaseOperator


class SampleOperator(BaseOperator[SampleProcessStatus], Generic[RoiType, FeatureType]):
    """Base operator for sample storage."""

    type: Literal[OperatorType.SAMPLE] = OperatorType.SAMPLE

    def apply(self, data: SampleStorage[RoiType, FeatureType]) -> None:
        """Apply the operator function to the data."""
        self.check_status(data.get_status())

        if hasattr(self, "pre_apply"):
            self.pre_apply()  # type: ignore

        self._apply_operator(data)

        if hasattr(self, "post_apply"):
            self.post_apply()  # type: ignore

        self.update_status(data.get_status())

    @abstractmethod
    def _apply_operator(self, data: SampleStorage) -> None: ...


class RoiExtractor(SampleOperator[RoiType, FeatureType]):
    """Extract ROIs from raw sample data."""

    def _apply_operator(self, data: SampleStorage[RoiType, FeatureType]) -> None:
        sample = data.get_sample()
        rois = self.extract_rois(sample)
        data.add_rois(*rois)

    @abstractmethod
    def extract_rois(self, sample: Sample) -> list[RoiType]: ...  # noqa

    def get_expected_status_in(self) -> SampleProcessStatus:
        """Get the expected status before performing ROI extraction."""
        return SampleProcessStatus()

    def get_expected_status_out(self) -> SampleProcessStatus:
        """Get the expected status after performing ROI extraction."""
        return SampleProcessStatus(roi_extracted=True)


class RoiTransformer(SampleOperator[RoiType, FeatureType]):
    """Transform ROIs from raw sample data.

    Must implement the `transform_roi` method, which takes a single ROI and transform it inplace.

    If `transform_roi` returns ``None``, the ROI is removed from the sample storage.

    """

    def _apply_operator(self, data: SampleStorage[RoiType, FeatureType]) -> None:
        rm_roi_list = list()
        for roi in data.list_rois():
            transformed = self.transform_roi(roi)
            if transformed is None:
                rm_roi_list.append(roi.id)
        data.delete_rois(*rm_roi_list)

    @abstractmethod
    def transform_roi(self, roi: RoiType) -> RoiType | None: ...  # noqa

    def get_expected_status_in(self) -> SampleProcessStatus:
        """Get the expected status before performing ROI extraction."""
        return SampleProcessStatus(roi_extracted=True)

    def get_expected_status_out(self) -> SampleProcessStatus:
        """Get the expected status after performing ROI extraction."""
        return SampleProcessStatus(roi_extracted=True)


class FeatureExtractor(SampleOperator[RoiType, FeatureType]):
    """Extract features from ROIs.

    Must implement the `extract_rois` method, which takes a single ROI and creates a list of features.

    Provides descriptor based filtering of features using the :py:attr:`bounds`.

    """

    bounds: dict[str, tuple[pydantic.NonNegativeFloat | None, pydantic.NonNegativeFloat | None]] = dict()
    """Define valid boundaries for each feature descriptor. Boundaries are expressed by mapping descriptor
    names to a tuple lower and upper bounds. If only a lower/upper bound is required, ``None`` must be used
    (e.g. ``(None, 10.0)`` to use only an upper bound)."""

    def _apply_operator(self, data: SampleStorage[RoiType, FeatureType]) -> None:
        bounds = dict()
        for descriptor, (lower, upper) in self.bounds.items():
            lower = lower if lower is not None else -inf
            upper = upper if upper is not None else inf
            bounds[descriptor] = lower, upper

        features: list[FeatureType] = list()
        for roi in data.list_rois():
            extracted = [x for x in self.extract_features(roi) if x.has_descriptors_in_range(**bounds)]
            features.extend(extracted)
        data.add_features(*features)

    @abstractmethod
    def extract_features(self, roi: RoiType) -> list[FeatureType]: ...  # noqa

    def get_expected_status_in(self) -> SampleProcessStatus:
        """Get the expected status before performing ROI extraction."""
        return SampleProcessStatus(roi_extracted=True)

    def get_expected_status_out(self) -> SampleProcessStatus:
        """Get the expected status after performing ROI extraction."""
        return SampleProcessStatus(roi_extracted=True, feature_extracted=True)


class FeatureTransformer(SampleOperator[RoiType, FeatureType]):
    """Apply a transformation to individual features.

    Must implement the `transform_feature` method, which takes a single feature and transform it inplace.

    """

    def _apply_operator(self, data: SampleStorage[RoiType, FeatureType]) -> None:
        rm_features = list()
        for feature in data.list_features():
            transformed = self.transform_feature(feature)
            if transformed is None:
                rm_features.append(feature.id)
        data.delete_features(*rm_features)

    @abstractmethod
    def transform_feature(self, feature: FeatureType) -> FeatureType | None: ...  # noqa

    def get_expected_status_in(self) -> SampleProcessStatus:
        """Get the expected status before performing ROI extraction."""
        return SampleProcessStatus(roi_extracted=True, feature_extracted=True)

    def get_expected_status_out(self) -> SampleProcessStatus:
        """Get the expected status after performing ROI extraction."""
        return SampleProcessStatus(roi_extracted=True, feature_extracted=True)
