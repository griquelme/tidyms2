"""TidyMS core operators."""

from __future__ import annotations

from typing import Any, overload

from ..dataflow import AssayProcessStatus, SampleProcessStatus
from ..exceptions import PipelineConfigurationError, ProcessStatusError, RepeatedIdError
from ..models import FeatureType, RoiType
from ..registry import operator_registry
from ..storage import AssayStorage, SampleStorage
from .assay import AssayOperator
from .base import BaseOperator
from .sample import SampleOperator


class Pipeline:
    """Compose multiple operators of the same type into a single unit."""

    def __init__(self, id: str) -> None:
        self.id = id
        self.operators: list[SampleOperator | AssayOperator | Pipeline] = list()
        self._id_to_operator = dict()

    def copy(self) -> Pipeline:
        """Create a copy of the current instance."""
        cp = Pipeline(self.id)
        for op in self.operators:
            op_cp = op.copy() if isinstance(op, Pipeline) else op.model_copy(deep=True)
            cp.add_operator(op_cp)
        return cp

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

        if operator.id in self._id_to_operator:
            msg = f"Pipeline {self.id} already contains an operator with id {operator.id}."
            raise RepeatedIdError(msg)

        self.operators.append(operator)
        self._id_to_operator[operator.id] = operator

    def get_operator(self, id: str) -> AssayOperator | SampleOperator | Pipeline:
        """Get an operator by id.

        :param id: the id of an operator in the pipeline
        :raises ValueError: if the operator is not found

        """
        try:
            return self._id_to_operator[id]
        except KeyError:
            raise ValueError(f"Operator with id {id} not found in pipeline `{self.id}`.")

    def list_operator_ids(self) -> list[str]:
        """Retrieve the name of all operators in the pipeline."""
        return list(self._id_to_operator)

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
