"""Abstract matrix operators."""

from __future__ import annotations

import concurrent.futures
from abc import abstractmethod
from multiprocessing import get_context

from ..dataflow import DataMatrixProcessStatus
from ..matrix import DataMatrix, FeatureVector, SampleVector
from ..utils.numpy import FloatArray
from .base import BaseOperator


class MatrixOperator(BaseOperator[DataMatrixProcessStatus]):
    """Base operator for data matrix transformation."""

    def apply(self, data: DataMatrix) -> None:
        """Apply the operator function to the data."""
        self.check_status(data.get_process_status())

        if hasattr(self, "pre_apply"):
            self.pre_apply(data)  # type: ignore

        self._apply_operator(data)

        if hasattr(self, "post_apply"):
            self.post_apply(data)  # type: ignore

        self.update_status(data.get_process_status())

    @abstractmethod
    def _apply_operator(self, data: DataMatrix) -> None: ...


class MatrixTransformer(MatrixOperator):
    """Apply an arbitrary transformation to a data matrix.

    MUST implement `transform_matrix` which takes a data matrix and produces a 2D array
    that will replace the matrix data.

    """

    def _apply_operator(self, data: DataMatrix) -> None:
        data.set_data(self._transform_matrix(data))

    @abstractmethod
    def _transform_matrix(self, data: DataMatrix) -> FloatArray: ...


class ColumnFilter(MatrixOperator):
    """Remove data matrix features based on a specific condition.

    MUST implement the method `_create_remove_list` that takes the data matrix and produce a
    list of feature groups to remove.

    """

    exclude: list[int] | None = None
    """A list of feature groups to ignore during filtering."""

    def _apply_operator(self, data: DataMatrix) -> None:
        exclude = self.exclude or set()
        data.remove_features(*(g for g in self._create_remove_list(data) if g not in exclude))

    @abstractmethod
    def _create_remove_list(self, data: DataMatrix) -> list[int]: ...


class RowFilter(MatrixOperator):
    """Remove data matrix samples based on a specific condition.

    MUST implement the method `_create_remove_list` that takes the data matrix and produce a
    list of feature groups to remove.

    """

    exclude: list[str] | None = None
    """A list of sample ids to ignore during filtering."""

    def _apply_operator(self, data: DataMatrix) -> None:
        exclude = set() if self.exclude is None else set(self.exclude)
        data.remove_samples(*(g for g in self._create_remove_list(data) if g not in exclude))

    @abstractmethod
    def _create_remove_list(self, data: DataMatrix) -> list[str]: ...


class MultiProcessTransformer(MatrixOperator):
    """Abstract operator with configuration for using multiprocess transformations."""

    max_workers: int | None = None
    """the maximum number of workers subprocesses"""

    spawn_context: str = "spawn"
    """the method used to create subprocesses, passed to :py:func:`multiprocess.get_context`"""


class ColumnTransformer(MultiProcessTransformer):
    """Transform a data matrix columns based on an arbitrary transformation.

    MUST implement the method `_transform_column` that takes a feature vector and creates a
    transformed vector.

    """

    exclude: list[int] | None = None
    """A list of feature groups to ignore during filtering."""

    def _apply_operator(self, data: DataMatrix) -> None:
        exclude = set() if self.exclude is None else set(self.exclude)
        include = [x.group for x in data.features if x.group not in exclude]

        transformed: list[FeatureVector] = list()
        mp_context = get_context(method=self.spawn_context)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
            futures = [executor.submit(self._transform_column, s) for s in data.get_columns(*include)]
            for fs in concurrent.futures.as_completed(futures):
                transformed.append(fs.result())

        data.set_columns(*((x.feature.group, x.data) for x in transformed))

    @abstractmethod
    def _transform_column(self, column: FeatureVector) -> FeatureVector: ...


class RowTransformer(MultiProcessTransformer):
    """Transform a data matrix rows based on an arbitrary transformation.

    MUST implement the method `_transform_row` that takes a sample vector and creates a
    transformed vector.

    """

    exclude: list[str] | None = None
    """A list of sample ids to exclude from the transformation"""

    def _apply_operator(self, data: DataMatrix) -> None:
        exclude = set() if self.exclude is None else set(self.exclude)
        include = [x.id for x in data.samples if x.id not in exclude]

        transformed: list[SampleVector] = list()
        mp_context = get_context(method=self.spawn_context)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_context) as executor:
            futures = [executor.submit(self._transform_row, s) for s in data.get_rows(*include)]
            for fs in concurrent.futures.as_completed(futures):
                transformed.append(fs.result())

        data.set_rows(*((x.sample.id, x.data) for x in transformed))

    @abstractmethod
    def _transform_row(self, row: SampleVector) -> SampleVector: ...
