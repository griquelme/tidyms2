"""TidyMS core operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import inf
from typing import Generic, Literal, Self, TypeVar, overload

import pydantic

from .dataflow import AssayProcessStatus, ProcessType, SampleProcessStatus, check_process_status, update_process_status
from .enums import MSInstrument, OperatorType, Polarity, SeparationMode
from .exceptions import PipelineConfigurationError, ProcessStatusError, RepeatedIdError
from .models import FeatureType, RoiType, Sample
from .storage import SampleStorage


class BaseOperator(ABC, pydantic.BaseModel, Generic[ProcessType]):
    """TidyMS base operator which all other operators inherit from.

    Provides functionality to:
    - set default parameters using instrument type, separation type and polarity.
    - set parameters using a dictionary.
    - get default parameters.

    """

    id: str = ""
    """The Operator id."""

    model_config = pydantic.ConfigDict(validate_assignment=True)

    @abstractmethod
    def get_expected_status_in(self) -> ProcessType:
        """Get the expected sample status before applying the operator."""
        ...

    @abstractmethod
    def get_expected_status_out(self) -> ProcessType:
        """Get the expected sample status after applying the operator."""
        ...

    def check_status(self, status: ProcessType) -> None:
        """Raise an exception if data status is not compatible with operator required status."""
        check_process_status(status, self.get_expected_status_in())

    def update_status(self, status_in: ProcessType):
        """Update the sample process status to the status after applying the operator."""
        return update_process_status(status_in, self.get_expected_status_out())

    def apply(self, data) -> None:
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

    @classmethod
    @abstractmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        """Create a new operator with sane defaults for the specified MS instrument, separation mode and polarity.

        :param instrument: The MS instrument used to measure the samples
        :param separation: The analytical method used for separation
        :param polarity: The polarity in which the samples where measured
        :return: A new operator instance

        """
        ...


class SampleOperator(BaseOperator[SampleProcessStatus], Generic[RoiType, FeatureType]):
    """Base operator for sample storage."""

    type: Literal[OperatorType.SAMPLE] = OperatorType.SAMPLE


class AssayOperator(BaseOperator[AssayProcessStatus], Generic[RoiType, FeatureType]):
    """Base operator for sample storage."""


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


OperatorItem = TypeVar("OperatorItem", SampleOperator, AssayOperator)


class Pipeline(pydantic.BaseModel, Generic[OperatorItem]):
    """Compose multiple operators of the same type into a single unit."""

    id: str
    operators: list[OperatorItem | Pipeline[OperatorItem]]

    @pydantic.field_validator("operators", mode="before")
    @classmethod
    def _ensure_non_empty_pipeline(cls, value: list) -> list:
        if not value:
            raise ValueError("Pipeline must contain at least one operator")
        return value

    @pydantic.field_validator("operators", mode="before")
    @classmethod
    def _ensure_unique_ids(cls, value: list) -> list:
        unique_ids = set()
        for op in value:
            if op.id in unique_ids:
                raise ValueError(f"Operators id must be unique. Got multiple operators with id {op.id}.")
            unique_ids.add(op.id)
        return value

    def add_operator(self, operator: OperatorItem | Pipeline[OperatorItem]):
        """Add a new operator to the pipeline.

        :param operator: the operator to add
        :raises PipelineConfigurationError: if the operator type differs from the pipeline type.

        """
        if operator.id in self.operators:
            msg = f"Pipeline {self.id} already contains an operator with id {operator.id}."
            raise RepeatedIdError(msg)
        self.operators.append(operator)

    def apply(self, data: SampleStorage) -> None:
        """Apply pipeline to the data."""
        for op in self.operators:
            op.apply(data)

    def validate_pipeline(self) -> None:
        """Check if the data status throughout the pipeline is valid."""
        first_operator = self.operators[0]
        while not isinstance(first_operator, BaseOperator):
            first_operator = first_operator.operators[0]

        initial_status = get_initial_process_status(first_operator)
        try:
            self._validate_recursion(initial_status)
        except ProcessStatusError as e:
            msg = "Check that all operators are of the same type and that the order is valid."
            raise PipelineConfigurationError(msg) from e

    def _validate_recursion(self, status):
        for op in self.operators:
            if isinstance(op, Pipeline):
                op._validate_recursion(status)
            else:
                op.check_status(status)
                op.update_status(status)


@overload
def get_initial_process_status(operator: SampleOperator) -> SampleProcessStatus: ...


@overload
def get_initial_process_status(operator: AssayOperator) -> AssayProcessStatus: ...


def get_initial_process_status(operator):
    """Create an initial data process status."""
    if isinstance(operator, SampleOperator):
        return SampleProcessStatus()
    elif isinstance(operator, AssayOperator):
        return AssayProcessStatus()
    else:
        raise NotImplementedError
