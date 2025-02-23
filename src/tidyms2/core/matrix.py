"""Data matrix implementation."""

from __future__ import annotations

from functools import cache
from typing import Generator, Self, Sequence

import numpy
import pydantic

from . import exceptions
from .dataflow import DataMatrixProcessStatus
from .enums import SampleType
from .models import FeatureGroup, Sample
from .utils.metrics import cv, detection_rate, dratio
from .utils.numpy import FloatArray, FloatArray1D


class DataMatrix:
    """Storage class for matrix data.

    :param samples: the list of samples in the data matrix. Each sample is associated with a matrix row.
    :param features: the list of features in the data matrix. Each feature is associated with a matrix column.
    :param data: A 2D numpy float array with matrix data. The number of rows and columns must match the
        `samples`  and `features` length respectively.
    :param validate: If set to ``True`` will assume that the input data is sanitized. Otherwise, will validate
        and normalize data before creating the data matrix. Set to ``False`` by default.

    """

    class Metrics:
        """Define data matrix metrics computation."""

        def __init__(self, matrix: DataMatrix) -> None:
            self.matrix = matrix

        def cv(self, *groupby: str, robust: bool = False) -> dict[Sequence[str], FloatArray1D]:
            r"""Compute features coefficient of variation (CV).

            .. math::

                \textrm{CV} = \frac{\bar{X}}{S}

            where :math:`S` is the sample standard deviation and :math:`\bar{X}` is the sample mean

            :param groupby: a list of sample metadata fields to compute the CV by group.
            :param robust: If set to ``True`` will use the sample median absolute deviation and
                median instead of the standard deviation and mean.
            :return: a dictionary where the keys are group values for each matrix partition and
                the values are the CV estimation for each feature in the group. NaN values will be
                obtained if all values in the column are zero or NaN. If `robust` is set to ``False``
                and the number less than two values in the column are not NaN, a NaN value will also
                be obtained.
            :raises ValueError: if a sample does not contain a metadata field defined in `groupby`.

            """
            result = dict()
            for group, sub in self.matrix.split(*groupby):
                result[group] = cv(sub.get_data(), robust=robust)
            return result

        def detection_rate(
            self,
            *groupby: str,
            threshold: float | FloatArray1D = 0.0,
        ) -> dict[Sequence[str], FloatArray1D]:
            """Compute the detection rate of features (DR)."""
            result = dict()
            for group, sub in self.matrix.split(*groupby):
                result[group] = detection_rate(sub.get_data(), threshold=threshold)
            return result

        def dratio(
            self,
            sample_groups: list[str] | None = None,
            qc_groups: list[str] | None = None,
            robust: bool = False,
        ) -> FloatArray:
            r"""Compute the D-ratio metric for columns.

            The D-ratio is defined as the quotient between the standard deviation of QC data, or data that
            is expected to exhibit instrumental variation only and the standard deviation of sample data
            or data that presents `biological` variation.

            .. math::

                \textrm{D-Ratio} = \frac{S_{\textrm{QC}}}{S_{\textrm{sample}}}

            where :math:`S_{\textrm{sample}` is the sample standard deviation and :math:`S_{\textrm{QC}` is
            the QC standard deviation.

            NaN values in the sample or QC data will be ignored in the computation of the standard deviation.

            :param sample_groups: a list of sample groups with biological variation. If not provided, uses
                all samples with :term:`sample type` :py:class:`SampleType.SAMPLE`.
            :param qc_groups: a list of sample groups with instrumental variation only. If not provided, uses
                all samples with :term:`sample type` :py:class:`SAMPLETYPE.TECHNICAL_QC`.
            :param robust: if set to ``True`` estimate the D-ratio using the median absolute deviation instead.
            :return: an 1D array with the D-ratio of each column. Columns with constant sample values will
                result in ``Inf``. If both sample and QC columns are contant, the result will be ``NaN``.
                If `robust` is set to ``True`` and there are less than two non ``NaN`` values in either `Xqc`
                or `Xs` columns, NaN values will also be obtained.
            """
            all_samples = self.matrix.list_samples()
            if sample_groups is None:
                sample_ids = [x.id for x in all_samples if x.meta.type is SampleType.SAMPLE]
            else:
                sample_ids = [x.id for x in all_samples if x.meta.group in sample_groups]

            if qc_groups is None:
                qc_sample_ids = [x.id for x in all_samples if x.meta.type is SampleType.TECHNICAL_QC]
            else:
                qc_sample_ids = [x.id for x in all_samples if x.meta.group in qc_groups]

            Xs = self.matrix.filter_rows(*sample_ids).get_data()
            Xqc = self.matrix.filter_rows(*qc_sample_ids).get_data()
            return dratio(Xs, Xqc, robust=robust)

        def pca(self):
            """Compute the PCA scores and loading."""
            ...

        def correlation(self, field): ...

    def __init__(
        self,
        samples: Sequence[Sample],
        features: Sequence[FeatureGroup],
        data: FloatArray,
        validate: bool = True,
    ):
        self._data = data
        self._samples = [x for x in samples]
        self._features = [x for x in features]
        self._status = DataMatrixProcessStatus()
        self.metrics = self.Metrics(self)
        if validate:
            self.validate()

    def get_n_features(self) -> int:
        """Retrieve the number of feature groups in the data matrix."""
        return len(self._features)

    def get_n_samples(self) -> int:
        """Retrieve the number of samples in the data matrix."""
        return len(self._samples)

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
            groups = tuple(self._feature_index())

        columns = list()
        for g in groups:
            if g not in self._feature_index():
                raise exceptions.FeatureGroupNotFound(g)
            index = self._feature_index()[g]
            row = FeatureVector(data=self._data[:, index], feature=self._features[index])
            columns.append(row)
        return columns

    def get_rows(self, *ids: str) -> list[SampleVector]:
        """Retrieve rows from the data matrix.

        :param ids: the sample ids associated with each row

        """
        if not ids:
            ids = tuple(self._sample_index())

        rows = list()
        for s in ids:
            if s not in self._sample_index():
                raise exceptions.SampleNotFound(s)
            index = self._sample_index()[s]
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
        return [self._feature_index()[x] for x in groups]

    def get_sample_index(self, *sample_ids: str) -> list[int]:
        """Retrieve the list of indices in the data associated with samples.

        :param sample_ids: the list of samples to search

        """
        return [self._sample_index()[x] for x in sample_ids]

    def has_feature(self, group: int) -> bool:
        """Check if a feature group is stored in the matrix.

        :param group: the feature group to check

        """
        return group in self._feature_index()

    def has_sample(self, sample_id: str) -> bool:
        """Check if a sample is stored in the matrix.

        :param sample_id: the sample id to check

        """
        return sample_id in self._sample_index()

    def set_columns(self, *pairs: tuple[int, FloatArray1D]) -> None:
        """Set column values in the data matrix.

        :param pairs: a tuple consisting of in a feature group and the corresponding column data.

        """
        n_rows = self.get_n_samples()
        # TODO: check data non-negative
        for group, col in pairs:
            if not self.has_feature(group):
                raise exceptions.FeatureGroupNotFound(group)

            if not col.size == n_rows:
                raise ValueError("The size of columns must be equal to the number of rows in the matrix.")

            if col.dtype.kind != "f":
                raise TypeError("Data array must be of float dtype.")

        for group, col in pairs:
            ind = self._feature_index()[group]
            self._data[:, ind] = col

    def set_data(self, data: FloatArray) -> None:
        """Set all values in the data matrix."""
        validate_data_matrix(self._samples, self._features, data)
        self._data = data.copy()

    def set_rows(self, *pairs: tuple[str, FloatArray1D]) -> None:
        """Set row values in the data matrix.

        :param pairs: a tuple consisting of in a sample id and the corresponding column data.

        """
        # TODO: check data non-negative
        n_cols = self.get_n_features()
        for sample_id, row in pairs:
            if not self.has_sample(sample_id):
                raise exceptions.SampleNotFound(sample_id)

            if not row.size == n_cols:
                raise ValueError("Row size must be equal to the number of columns in the matrix.")

            if row.dtype.kind != "f":
                raise TypeError("Row data array must be of float dtype.")

        for sample_id, row in pairs:
            ind = self._sample_index()[sample_id]
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
            if g not in self._feature_index():
                raise exceptions.FeatureGroupNotFound(g)

        if len(groups) == self.get_n_features():
            raise exceptions.EmptyDataMatrix("Removing selected features will result in an empty data matrix.")

        self._data = numpy.delete(self._data, groups, axis=1)
        group_set = set(groups)
        self._features = [x for x in self._features if x.group not in group_set]
        self._feature_index.cache_clear()

    def remove_samples(self, *ids: str) -> None:
        """Remove samples with based on their ids.

        :param ids: the list of sample ids to remove

        """
        if not ids:
            return

        indices = list()
        for i in ids:
            if i not in self._sample_index():
                raise exceptions.SampleNotFound(i)
            indices.append(self._sample_index()[i])

        if len(indices) == self.get_n_samples():
            raise exceptions.EmptyDataMatrix("Removing the selected samples will result in an empty data matrix.")

        self._data = numpy.delete(self._data, indices, axis=0)
        id_set = set(ids)
        self._samples = [x for x in self._samples if x.id not in id_set]
        self._sample_index.cache_clear()

    def validate(self) -> None:
        """Perform a sanity check and normalization of the data matrix."""
        validate_data_matrix(self._samples, self._features, self._data)
        samples, data = sort_matrix_rows(self._samples, self._data)
        features, data = sort_matrix_columns(self._features, data)
        self._data = data
        self._samples = samples
        self._features = features

    @classmethod
    def combine(cls, *matrices: Self) -> Self:
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

    def filter_columns(self, *groups: int) -> Self:
        """Create a submatrix using a subset of rows.

        :param groups: the list of feature groups to include in the sub-matrix

        """
        idx = self.get_feature_index(*groups)
        data = self._data[:, idx].copy()
        features = [self._features[x] for x in idx]
        return self.__class__(self._samples, features, data, validate=False)

    def filter_rows(self, *ids: str) -> Self:
        """Create a submatrix using a subset of rows.

        :param ids: the list of sample ids to include in the sub-matrix

        """
        idx = self.get_sample_index(*ids)
        data = self._data[idx].copy()
        samples = [self._samples[i] for i in idx]
        return self.__class__(samples, self._features, data, validate=False)

    def split(self, *by: str) -> Generator[tuple[tuple, Self], None, None]:
        """Split data matrix into submatrices using sample metadata.

        :param by: the sample metadata fields to

        """
        for group, group_ids in self._create_sample_split_group_to_id(by).items():
            yield group, self.filter_rows(*group_ids)

    def _create_sample_split_group_to_id(self, groups: Sequence[str]) -> dict[tuple, list[str]]:
        group_to_ids = dict()
        for sample in self._samples:
            group_key = self._sample_to_group_key(sample, groups)
            group_ids = group_to_ids.setdefault(group_key, list())
            group_ids.append(sample.id)
        return group_to_ids

    @staticmethod
    def _sample_to_group_key(sample: Sample, groups: Sequence[str]) -> tuple:
        key = list()
        for g in groups:
            try:
                g_value = getattr(sample.meta, g)
            except AttributeError:
                raise exceptions.SampleMetadataNotFound(f"{sample.id}.meta.{g}")
            key.append(g_value)
        return tuple(key)

    @cache
    def _sample_index(self) -> dict[str, int]:
        """Map sample ids to indices in the data matrix rows."""
        return {x.id: k for k, x in enumerate(self._samples)}

    @cache
    def _feature_index(self) -> dict[int, int]:
        """Map feature groups to indices in the data matrix columns."""
        return {x.group: k for k, x in enumerate(self._features)}


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


def sort_matrix_columns(features: Sequence[FeatureGroup], data: FloatArray) -> tuple[list[FeatureGroup], FloatArray]:
    """Sort data using feature group label."""
    sorted_index = [k for k, _ in sorted(enumerate(features), key=lambda x: x[1].group)]
    return [features[x] for x in sorted_index], data[:, sorted_index]


def sort_matrix_rows(samples: Sequence[Sample], data: FloatArray) -> tuple[list[Sample], FloatArray]:
    """Sort data using sample run order."""
    sorted_index = [k for k, _ in sorted(enumerate(samples), key=lambda x: x[1].meta.order)]
    return [samples[x] for x in sorted_index], data[sorted_index]


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
