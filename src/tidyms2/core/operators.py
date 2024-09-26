"""TidyMS core operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import inf
from typing import Any, Generic, Literal, Self, overload

import pydantic

from .dataflow import AssayProcessStatus, ProcessType, SampleProcessStatus, check_process_status, update_process_status
from .enums import MSInstrument, OperatorType, Polarity, SeparationMode
from .exceptions import PipelineConfigurationError, ProcessStatusError, RepeatedIdError
from .models import AnnotationPatch, DescriptorPatch, FeatureType, FillValue, RoiType, Sample
from .registry import operator_registry
from .storage import AssayStorage, SampleStorage


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


class AssayOperator(BaseOperator[AssayProcessStatus]):
    """Base operator for sample storage."""

    def apply(self, data: AssayStorage) -> None:
        """Apply the operator function to the data."""
        self.check_status(data.get_process_status())

        if hasattr(self, "pre_apply"):
            self.pre_apply()  # type: ignore

        self._apply_operator(data)

        if hasattr(self, "post_apply"):
            self.post_apply()  # type: ignore

        self.update_status(data.get_process_status())

    @abstractmethod
    def _apply_operator(self, data: AssayStorage) -> None: ...


class AnnotationPatcher(AssayOperator):
    """Patches feature annotation data from an assay.

    Must Implement the compute_patches method, which takes an assay storage and returns
    a list of patches that will be applied to the assay.

    """

    def get_expected_status_in(self) -> AssayProcessStatus:
        """Get expected status of input data."""
        return AssayProcessStatus()

    def get_expected_status_out(self) -> AssayProcessStatus:
        """Get status of output data."""
        return AssayProcessStatus()

    def _apply_operator(self, data: AssayStorage) -> None:
        patches = self.compute_patches(data)
        data.patch_annotations(*patches)

    @abstractmethod
    def compute_patches(self, data: AssayStorage) -> list[AnnotationPatch]:
        """Compute feature descriptor patches."""
        ...


class DescriptorPatcher(AssayOperator):
    """Patches descriptor data from an assay.

    Must Implement the compute_patches method, which takes an assay storage and returns
    a list of patches that will be applied to the assay.

    """

    def get_expected_status_in(self) -> AssayProcessStatus:
        """Get expected status of input data."""
        return AssayProcessStatus()

    def get_expected_status_out(self) -> AssayProcessStatus:
        """Get status of output data."""
        return AssayProcessStatus()

    def _apply_operator(self, data: AssayStorage) -> None:
        patches = self.compute_patches(data)
        data.patch_descriptors(*patches)

    @abstractmethod
    def compute_patches(self, data: AssayStorage) -> list[DescriptorPatch]:
        """Compute feature descriptor patches."""
        ...


class MissingImputer(AssayOperator):
    """Add values that will be used as fill in missing data matrix entries.

    Must Implement the `add_fill_values` method, which takes an assay storage and returns
    a list of fill values.

    """

    def get_expected_status_in(self) -> AssayProcessStatus:
        """Get expected status of input data."""
        return AssayProcessStatus(feature_matched=True)

    def get_expected_status_out(self) -> AssayProcessStatus:
        """Get status of output data."""
        return AssayProcessStatus(missing_imputed=True)

    def _apply_operator(self, data: AssayStorage) -> None:
        fill_values = self.add_fill_values(data)
        data.add_fill_values(*fill_values)

    @abstractmethod
    def add_fill_values(self, data: AssayStorage) -> list[FillValue]:
        """Compute feature descriptor patches."""
        ...


class Pipeline:
    """Compose multiple operators of the same type into a single unit."""

    def __init__(self, id: str) -> None:
        self.id = id
        self.operators: list[SampleOperator | AssayOperator | Pipeline] = list()

    def __eq__(self, other) -> bool:
        equal_ids = self.id == other.id
        equal_operators = self.operators == other.operators
        return equal_ids and equal_operators

    @overload
    def add_operator(self, operator: AssayOperator) -> None: ...

    @overload
    def add_operator(self, operator: SampleOperator) -> None: ...

    @overload
    def add_operator(self, operator: Pipeline) -> None: ...

    def add_operator(self, operator):
        """Add a new operator to the pipeline.

        :param operator: the operator to add
        :raises PipelineConfigurationError: if the operator type differs from the pipeline type.

        """
        check_compatible_element(self, operator)

        if any(x.id == operator.id for x in self.operators):
            msg = f"Pipeline {self.id} already contains an operator with id {operator.id}."
            raise RepeatedIdError(msg)

        self.operators.append(operator)

    @overload
    def apply(self, data: SampleStorage[RoiType, FeatureType]) -> None: ...

    @overload
    def apply(self, data: AssayStorage[RoiType, FeatureType]) -> None: ...

    def apply(self, data) -> None:
        """Apply pipeline to the data."""
        for op in self.operators:
            op.apply(data)

    @classmethod
    def deserialize(cls, d: dict[str, Any]) -> Pipeline:
        """Deserialize a dictionary into a pipeline."""
        id_ = d.get("id")
        if not isinstance(id_, str):
            raise ValueError("`id` is a mandatory field and must be a string.")

        operators = d.get("operators")
        if not isinstance(operators, list):
            raise ValueError("`operators` is a mandatory field and must be a list of dictionaries.")

        pipe = Pipeline(id_)

        for d in operators:
            if not isinstance(d, dict):
                raise ValueError("`operators` element is not a dictionary.")
            op_type = d.pop("class", None)
            if op_type is None:
                op = Pipeline.deserialize(d)
            else:
                T = operator_registry.get(op_type)
                op = T(**d)
            pipe.add_operator(op)
        return pipe

    def serialize(self) -> dict:
        """Serialize pipeline into a JSON serializable dictionary."""
        operators = list()
        serialized = {"id": self.id, "operators": operators}
        for op in self.operators:
            if isinstance(op, Pipeline):
                d = op.serialize()
            else:
                d = op.model_dump(mode="json")
                d["class"] = op.__class__.__name__
            operators.append(d)
        return serialized

    def validate_dataflow(self) -> None:
        """Check if the data status throughout the pipeline is valid."""
        first_operator = self.operators[0]
        while not isinstance(first_operator, BaseOperator):
            first_operator = first_operator.operators[0]

        initial_status = get_initial_process_status(first_operator)
        try:
            self._validate_dataflow_recursion(initial_status)
        except ProcessStatusError as e:
            msg = "Check that all operators are of the same type and that the order is valid."
            raise PipelineConfigurationError(msg) from e

    def _validate_dataflow_recursion(self, status):
        for op in self.operators:
            if isinstance(op, Pipeline):
                op._validate_dataflow_recursion(status)
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


OT = SampleOperator | AssayOperator
PipelineElement = OT | Pipeline


def check_compatible_element(pipeline: Pipeline, element: PipelineElement) -> None:
    """Raise an exception if two pipeline elements are not compatible."""
    pipeline_first = get_first_operator(pipeline)
    element_first = get_first_operator(element)
    if element_first is None:
        raise PipelineConfigurationError("Nested pipelines cannot be empty")

    if pipeline_first is None:
        return None

    both_sample_operators = isinstance(pipeline_first, SampleOperator) and isinstance(element_first, SampleOperator)
    both_assay_operators = isinstance(pipeline_first, AssayOperator) and isinstance(element_first, AssayOperator)
    if not both_sample_operators or both_assay_operators:
        msg = "All pipeline elements must process the same data type (sample, assay or matrix)."
        raise PipelineConfigurationError(msg)


def get_first_operator(op: PipelineElement) -> OT | None:
    """Get the first element of a pipeline or raise an error if the pipeline is empty."""
    if not isinstance(op, Pipeline):
        return op
    elif not op.operators:
        return None
    return get_first_operator(op.operators[0])
