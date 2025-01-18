"""Data matrix implementation."""

from __future__ import annotations

from typing import Generator, Sequence

import numpy
import pydantic

from ..utils.numpy import FloatArray, FloatArray1D
from . import exceptions
from .dataflow import DataMatrixProcessStatus
from .models import FeatureGroup, Sample


class DataMatrix:
    """Storage class for matrix data.

    :param samples: the list of samples in the data matrix. Each sample is associated with a matrix row.
    :param features: the list of features in the data matrix. Each feature is associated with a matrix column.
    :param data: A 2D numpy float array with matrix data. The number of rows and columns must match the
        `samples`  and `features` length respectively.

    """

    class Metrics:
        """Define data matrix metrics computation."""

        def __init__(self, matrix: DataMatrix) -> None:
            self.matrix = matrix

        def cv(self, *groupby: str, exclude_missing: bool = True):
            r"""Compute features coefficient of variation (CV).

            .. math::

                \textrm{CV} = \frac{\bar{X}}{S}

            where :math:`S` is the sample standard deviation.

            :param groupby: a list of sample metadata fields to compute the CV by group.
            :param exclude_missing: exclude samples that does not contain the metadata
                fields specified in `groupby`. If set to ``False`` it will raise an error.
            :raises ValueError: if a sample does not contain a metadata field defined in `groupby`.

            """

    def __init__(self, samples: Sequence[Sample], features: Sequence[FeatureGroup], data: FloatArray):
        validate_data_matrix(samples, features, data)
        self._status = DataMatrixProcessStatus()
        self.metrics = self.Metrics(self)
        self._data = data
        self._samples = [x for x in samples]
        self._features = [x for x in features]
        self._sort_by_sample_order()
        self._sort_by_feature_group()
        self._build_feature_index()
        self._build_sample_index()

    def get_n_features(self) -> int:
        """Retrieve the number of feature groups in the data matrix."""
        return len(self._features)

    def get_n_samples(self) -> int:
        """Retrieve the number of samples in the data matrix."""
        return self._data.shape[0]

    def get_process_status(self) -> DataMatrixProcessStatus:
        """Retrieve the current data matrix status."""
        return self._status

    def list_samples(self) -> list[Sample]:
        """List all samples in the data matrix."""
        return self._samples.copy()

    def list_features(self) -> list[FeatureGroup]:
        """List all features in the data matrix."""
        return self._features.copy()

    def add_columns(self, *columns: FeatureVector) -> None:
        """Add columns to the data matrix.

        :param features: the list of columns to add

        """
        ...

    def get_columns(self, *groups: int) -> list[FeatureVector]:
        """Retrieve columns from the data matrix.

        :param groups: the feature groups associated with each column. If no groups are provided then all
            groups are retrieved.

        """
        if not groups:
            groups = tuple(self._feature_group_to_index)

        columns = list()
        for g in groups:
            if g not in self._feature_group_to_index:
                raise exceptions.FeatureGroupNotFound(g)
            index = self._feature_group_to_index[g]
            row = FeatureVector(data=self._data[:, index], feature=self._features[index])
            columns.append(row)
        return columns

    def get_rows(self, *ids: str) -> list[SampleVector]:
        """Retrieve rows from the data matrix.

        :param ids: the sample ids associated with each row

        """
        if not ids:
            ids = tuple(self._sample_id_to_index)

        rows = list()
        for s in ids:
            if s not in self._sample_id_to_index:
                raise exceptions.SampleNotFound(s)
            index = self._sample_id_to_index[s]
            row = SampleVector(data=self._data[index], sample=self._samples[index])
            rows.append(row)
        return rows

    def get_data(self) -> FloatArray:
        """Retrieve the matrix data in numpy format.

        Each rows in the array is associated with a sample and each column is associated with a feature.

        """
        return self._data

    def get_feature_index(self, *groups: int) -> list[int]:
        """Retrieve the list of indices in the data associated with feature groups.

        :param groups: the list of feature groups to search

        """
        return [self._feature_group_to_index[x] for x in groups]

    def get_sample_index(self, *sample_ids: str) -> list[int]:
        """Retrieve the list of indices in the data associated with samples.

        :param sample_ids: the list of samples to search

        """
        return [self._sample_id_to_index[x] for x in sample_ids]

    def has_feature(self, group: int) -> bool:
        """Check if a feature group is stored in the matrix.

        :param group: the feature group to check

        """
        return group in self._feature_group_to_index

    def has_sample(self, sample_id: str) -> bool:
        """Check if a sample is stored in the matrix.

        :param sample_id: the sample id to check

        """
        return sample_id in self._sample_id_to_index

    def set_columns(self, *pairs: tuple[int, FloatArray1D]) -> None:
        """Set column values in the data matrix.

        :param pairs: a tuple consisting of in a feature group and the corresponding column data.

        """
        n_rows = self.get_n_samples()
        for group, col in pairs:
            if not self.has_feature(group):
                raise exceptions.FeatureGroupNotFound(group)

            if not col.size == n_rows:
                raise ValueError("The size of columns must be equal to the number of rows in the matrix.")

            if col.dtype.kind != "f":
                raise TypeError("Data array must be of float dtype.")

        for group, col in pairs:
            ind = self._feature_group_to_index[group]
            self._data[:, ind] = col

    def set_data(self, data: FloatArray) -> None:
        """Set all values in the data matrix."""
        validate_data_matrix(self._samples, self._features, data)
        self._data = data.copy()

    def set_rows(self, *pairs: tuple[str, FloatArray1D]) -> None:
        """Set row values in the data matrix.

        :param pairs: a tuple consisting of in a sample id and the corresponding column data.

        """
        n_cols = self.get_n_features()
        for sample_id, row in pairs:
            if not self.has_sample(sample_id):
                raise exceptions.SampleNotFound(sample_id)

            if not row.size == n_cols:
                raise ValueError("Row size must be equal to the number of columns in the matrix.")

            if row.dtype.kind != "f":
                raise TypeError("Row data array must be of float dtype.")

        for sample_id, row in pairs:
            ind = self._sample_id_to_index[sample_id]
            self._data[ind] = row

    def query(self, stmt: str):
        """Query data matrix using SQL syntax.

        :param stmt: the SQL statement to query data.

        """
        # TODO: use polars.LazyFrame + SQL API
        ...

    def remove_features(self, *groups: int) -> None:
        """Remove feature groups based on their groups labels.

        :param groups: the group labels to remove

        """
        if not groups:
            return

        for g in groups:
            if g not in self._feature_group_to_index:
                raise exceptions.FeatureGroupNotFound(g)

        if len(groups) == self.get_n_features():
            raise exceptions.EmptyDataMatrix("Removing selected features will result in an empty data matrix.")

        self._data = numpy.delete(self._data, groups, axis=1)
        group_set = set(groups)
        self._features = [x for x in self._features if x.group not in group_set]
        self._build_feature_index()

    def remove_samples(self, *ids: str) -> None:
        """Remove samples with based on their ids.

        :param ids: the list of sample ids to remove

        """
        if not ids:
            return

        indices = list()
        for i in ids:
            if i not in self._sample_id_to_index:
                raise exceptions.SampleNotFound(i)
            indices.append(self._sample_id_to_index[i])

        if len(indices) == self.get_n_samples():
            raise exceptions.EmptyDataMatrix("Removing the selected samples will result in an empty data matrix.")

        self._data = numpy.delete(self._data, indices, axis=0)
        id_set = set(ids)
        self._samples = [x for x in self._samples if x.id not in id_set]
        self._build_sample_index()

    def split(self, *by: str) -> Generator[tuple[tuple, DataMatrix], None, None]:
        """Split data matrix into submatrices using sample metadata.

        :param by: the sample metadata fields to

        """
        for group, group_indices in self._create_sample_group_to_indices(by).items():
            group_data = self._data[group_indices].copy()
            group_samples = [self._samples[i] for i in group_indices]
            validate_data_matrix(group_samples, self._features, group_data)
            yield group, self.__class__(group_samples, self._features, group_data)

    @classmethod
    def combine(cls, *matrices: DataMatrix) -> DataMatrix:
        """Combine multiple matrices into a single data matrix.

        All matrices are assumed to have the same feature groups.

        """
        if not matrices:
            raise ValueError("At least one matrix is required to perform matrix join.")
        samples = list()
        for m in matrices:
            samples.extend(m.list_samples())
        data = numpy.vstack([m.get_data() for m in matrices])
        features = matrices[0].list_features()
        return cls(samples, features, data)

    def _sort_by_sample_order(self) -> None:
        sorted_index = [k for k, _ in sorted(enumerate(self._samples), key=lambda x: x[1].meta.order)]
        self._samples = [self._samples[x] for x in sorted_index]
        self._data = self._data[sorted_index]

    def _sort_by_feature_group(self) -> None:
        sorted_index = [k for k, _ in sorted(enumerate(self._features), key=lambda x: x[1].group)]
        self._features = [self._features[x] for x in sorted_index]
        self._data = self._data[:, sorted_index]

    def _build_sample_index(self) -> None:
        self._sample_id_to_index = {x.id: k for k, x in enumerate(self._samples)}

    def _build_feature_index(self) -> None:
        self._feature_group_to_index = {x.group: k for k, x in enumerate(self._features)}

    def _create_sample_group_to_indices(self, groups: Sequence[str]) -> dict[tuple, list[int]]:
        group_to_indices = dict()
        for idx, sample in enumerate(self._samples):
            group_key = self._sample_to_group_key(sample, groups)
            group_indices = group_to_indices.setdefault(group_key, list())
            group_indices.append(idx)
        return group_to_indices

    @staticmethod
    def _sample_to_group_key(sample: Sample, groups: Sequence[str]) -> tuple:
        key = list()
        for g in groups:
            try:
                g_key = sample.meta.group if g == "group" else getattr(sample.meta, g)
            except AttributeError:
                raise exceptions.SampleMetadataNotFound(f"{sample.id}.meta.{g}")
            key.append(g_key)
        return tuple(key)


def validate_data_matrix(samples: Sequence[Sample], features: Sequence[FeatureGroup], data: FloatArray) -> None:
    r"""Perform sanity check on matrix data.

    :param samples: the data matrix samples
    :param features: the data matrix features
    :param data: the data matrix data
    :raises EmptyDataMatrix: if an empty samples or feature list is provided
    :raises ValueError: if the the data shape is not :math:`n_{samples} \times n_{features}` or if the data
        dtype is not float.
    :raises RepeatedIDError: if samples with repeated id are provided.
    :raises RepeatedSampleOrder: if samples with repeated sample order are provided

    """
    if not samples:
        raise exceptions.EmptyDataMatrix("Data matrix must contain at least one sample.")

    if not features:
        raise exceptions.EmptyDataMatrix("Data matrix must contain at least one feature group.")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array.")

    if data.dtype.kind != "f":
        raise ValueError("data dtype must be of floating type.")

    n_rows, n_cols = data.shape
    n_samples = len(samples)
    n_features = len(features)

    if n_samples != n_rows:
        msg = f"The number of samples ({n_samples}) does not match the number of rows in the data ({n_rows})."
        raise ValueError(msg)

    if n_features != n_cols:
        msg = f"The number of features ({n_features}) does not match the number of columns in the data ({n_cols})."
        raise ValueError(msg)

    if len({x.group for x in features}) < n_features:
        msg = "Features must have a unique group label."
        raise exceptions.RepeatedIdError(msg)

    if len({x.id for x in samples}) < n_samples:
        msg = "Samples must have a unique id."
        raise exceptions.RepeatedIdError(msg)

    if len({x.meta.order for x in samples}) < n_samples:
        msg = "Samples must have a unique order."
        raise exceptions.RepeatedSampleOrder(msg)


class BaseVector(pydantic.BaseModel):
    """A container for 1D vector data."""

    model_config = pydantic.ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data: FloatArray1D = pydantic.Field(repr=False)
    """The vector data."""


class FeatureVector(BaseVector):
    """Data matrix column view."""

    feature: FeatureGroup
    """The feature information associated with the matrix column."""


class SampleVector(BaseVector):
    """Data matrix row."""

    sample: Sample
    """The sample associated with the row."""
