"""Abstract assay operators."""

from __future__ import annotations

from abc import abstractmethod

from ..dataflow import AssayProcessStatus
from ..models import AnnotationPatch, DescriptorPatch, FillValue
from ..storage import AssayStorage
from .base import BaseOperator


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
        fill_values = self.compute_fill_values(data)
        data.add_fill_values(*fill_values)

    @abstractmethod
    def compute_fill_values(self, data: AssayStorage) -> list[FillValue]:
        """Compute fill values in a sample.

        :param sample: the sample to search missing values
        :param groups: the feature groups where the sample did not contribute a feature.

        """
        ...
