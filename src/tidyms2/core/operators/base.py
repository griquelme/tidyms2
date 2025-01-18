"""Abstract base operator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Self

import pydantic

from ..dataflow import ProcessType, check_process_status, update_process_status
from ..enums import MSInstrument, Polarity, SeparationMode


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
