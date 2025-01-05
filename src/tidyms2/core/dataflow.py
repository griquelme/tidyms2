"""The data flow models define valid workflows."""

from typing import Literal, TypeVar

import pydantic

from .enums import OperatorType
from .exceptions import ProcessStatusError


class BaseProcessStatus(pydantic.BaseModel):
    """Base model to store data storage status."""

    extra: dict[str, bool] = dict()
    """Extra fields added to any model."""

    def list_extras(self) -> list[str]:
        """List all extra fields."""
        return list(self.extra)

    def get_extra(self, field: str) -> bool:
        """Retrieve extra value. Return ``False`` if not found."""
        if field not in self.extra:
            return False
        return self.extra[field]

    def set_extra(self, field: str, value: bool) -> None:
        """Set extra value."""
        self.extra[field] = value


class SampleProcessStatus(BaseProcessStatus):
    """Report sample data process status.

    Each attribute is a flag representing the current status of the data through
    a processing pipeline.

    """

    type: Literal[OperatorType.SAMPLE] = OperatorType.SAMPLE

    roi_extracted: bool = False
    """Flags if ROI were extracted."""

    feature_extracted: bool = False
    """Flags if features were extracted."""

    isotopologue_annotated: bool = False
    """Flags if isotopologue features were annotated."""

    adduct_annotated: bool = False
    """Flags if adducts were annotated."""


class AssayProcessStatus(BaseProcessStatus):
    """Report assay data process status.

    Each attribute is a flag representing the current status of the data through
    a processing pipeline.

    """

    type: Literal[OperatorType.ASSAY] = OperatorType.ASSAY

    adduct_annotated: bool = False
    """Flags if adducts were annotated."""

    feature_matched: bool = False
    """Flags if features where matched"""

    features_group_created: bool = False
    """Flags if feature groups were created"""

    isotopologue_annotated: bool = False
    """Flags if isotopologue features were annotated."""

    missing_imputed: bool = False
    """Flags if missing values were imputed."""


class DataMatrixProcessStatus(BaseProcessStatus):
    """Report assay data process status.

    Each attribute is a flag representing the current status of the data through
    a processing pipeline.

    """

    type: Literal[OperatorType.MATRIX] = OperatorType.MATRIX

    normalized: bool = False
    """Flags if Sample values are normalized."""

    missing_imputed: bool = False
    """Flag if the matrix contain missing values."""


ProcessType = TypeVar("ProcessType", SampleProcessStatus, AssayProcessStatus, DataMatrixProcessStatus)


def check_process_status(actual: ProcessType, reference: ProcessType) -> None:
    """Check if sample status is compatible with a reference status.

    :param actual: The actual sample status
    :param reference: The status used as a reference
    :raises ProcessStatusError: if actual status is not compatible with the reference status

    """
    for field in reference.model_fields:
        if field == "type" or field == "extra":
            continue

        if not getattr(actual, field) and getattr(reference, field):
            msg = f"Expected {actual.type.value} status {field} to be True."
            raise ProcessStatusError(msg)

    for field in reference.list_extras():
        if reference.get_extra(field) and not actual.get_extra(field):
            msg = f"Expected extra check field {field} in {actual.type.value} to be True."
            raise ProcessStatusError(msg)


def update_process_status(actual: ProcessType, reference: ProcessType) -> None:
    """Update two compatible status using the reference.

    :param actual: the status to update
    :param reference: The status used as a reference

    """
    for field in reference.model_fields:
        if field == "type" or field == "extra":
            continue
        value = getattr(actual, field) or getattr(reference, field)
        setattr(actual, field, value)

    for extra in reference.list_extras():
        value = reference.get_extra(extra) or actual.get_extra(extra)
        actual.set_extra(extra, value)
