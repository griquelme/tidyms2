"""Data matrix implementation."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Self, Sequence

import numpy
import pydantic
from typing_extensions import deprecated

from . import exceptions
from .dataflow import DataMatrixProcessStatus
from .enums import CorrelationMethod, NormalizationMethod, SampleType, ScalingMethod
from .models import FeatureGroup, Sample
from .utils import metrics
from .utils.numpy import FloatArray, FloatArray1D


class DataMatrix:
    """Storage class for matrix data.

    :param samples: the list of samples in the data matrix. Each sample is associated with a matrix row.
    :param features: the list of features in the data matrix. Each feature is associated with a matrix column.
    :param data: A 2D numpy float array with matrix data. The number of rows and columns must match the
        `samples`  and `features` length respectively.
    :param validate: If set to ``True`` will assume that the input data is sanitized. Otherwise, will validate
        and normalize data before creating the data matrix. Set to ``True`` by default.

    """

    class Metrics:
        """Define data matrix metrics computation."""

        def __init__(self, matrix: DataMatrix) -> None:
            self.matrix = matrix

        def cv(self, robust: bool = False) -> FloatArray1D:
            r"""Compute features coefficient of variation (CV).

            .. math::

                \textrm{CV} = \frac{\bar{X}}{S}

            where :math:`S` is the sample standard deviation and :math:`\bar{X}` is the sample mean

            :param robust: If set to ``True`` will use the sample median absolute deviation and
                median instead of the standard deviation and mean.
            :return: a dictionary where the keys are group values for each matrix partition and
                the values are the CV estimation for each feature in the group. NaN values will be
                obtained if all values in the column are zero or NaN. If `robust` is set to ``False``
                and the number less than two values in the column are not NaN, a NaN value will also
                be obtained.
            :raises ValueError: if a sample does not contain a metadata field defined in `groupby`.

            """
            return metrics.cv(self.matrix.get_data(), robust=robust)

        def detection_rate(
            self,
            threshold: float | FloatArray1D = 0.0,
        ) -> FloatArray1D:
            """Compute the detection rate of features (DR)."""
            return metrics.detection_rate(self.matrix.get_data(), threshold=threshold)

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

            a D-ratio of 0.0 means that the technical variance is zero, and all observed variance can be
            attributed to a biological cause. On the other hand, a D-Ratio of 1.0 or larger, means that
            the observed variation is mostly technical.

            NaN values in the sample or QC data will be ignored in the computation of the standard deviation.

            :param sample_groups: a list of sample groups with biological variation. If not provided, uses
                all samples with :term:`sample type` :py:class:`SampleType.SAMPLE`.
            :param qc_groups: a list of sample groups with instrumental variation only. If not provided, uses
                all samples with :term:`sample type` :py:class:`SampleType.TECHNICAL_QC`.
            :param robust: if set to ``True`` estimate the D-ratio using the median absolute deviation instead.
            :return: an 1D array with the D-ratio of each column. Columns with constant sample values will
                result in ``Inf``. If both sample and QC columns are constant, the result will be ``NaN``.
                If `robust` is set to ``True`` and there are less than two non ``NaN`` values in either `Xqc`
                or `Xs` columns, NaN values will also be obtained.
            """
            if sample_groups is None:
                sample_query = self.matrix.query.filter(type=SampleType.SAMPLE)
            else:
                sample_query = self.matrix.query.filter(group=sample_groups)

            sample_query_results = sample_query.fetch_sample_ids()
            if not sample_query_results:
                raise ValueError("No samples found in sample group.")
            sample_ids = sample_query_results[0][1]

            if qc_groups is None:
                qc_query = self.matrix.query.filter(type=SampleType.TECHNICAL_QC)
            else:
                qc_query = self.matrix.query.filter(group=qc_groups)

            qc_sample_query_results = qc_query.fetch_sample_ids()
            if not qc_sample_query_results:
                raise ValueError("No samples found in qc group.")
            qc_sample_ids = qc_sample_query_results[0][1]

            Xs = self.matrix.get_data(sample_ids=sample_ids)
            Xqc = self.matrix.get_data(sample_ids=qc_sample_ids)
            return metrics.dratio(Xs, Xqc, robust=robust)

        def lod(self) -> FloatArray:
            r"""Compute the limit of detection (LOD) using blank samples.

            The limit of detection is defined as:

            .. math::

                \textrm{LOD} = \bar{X}_{\textrm{blank}} + 3 * \bar{S}_{\textrm{blank}}

            where :math:`\bar{X}_{\textrm{blank}}` is the feature mean in the blank samples and
            :math:`\bar{S}_{\textrm{blank}}` is the sample standard deviation of blanks.

            :return: an array with the LOD of each feature. If the LOD cannot be estimated because
                there are no blank samples or the blank contains only missing values, it with
                return zero instead.

            """
            blank_query = self.matrix.query.filter(type=SampleType.SAMPLE).fetch_sample_ids()
            if not blank_query:
                return numpy.zeros(shape=(self.matrix.get_n_features(),), dtype=float)
            blank_samples = blank_query[0][1]
            blank_data = self.matrix.get_data(sample_ids=blank_samples)
            return metrics.lod(blank_data)

        def loq(self) -> FloatArray:
            r"""Compute the limit of quantification (LOQ) using blank samples.

            The limit of quantification is defined as:

            .. math::

                \textrm{LOQ} = \bar{X}_{\textrm{blank}} + 10 * \bar{S}_{\textrm{blank}}

            where :math:`\bar{X}_{\textrm{blank}}` is the feature mean in the blank samples and
            :math:`\bar{S}_{\textrm{blank}}` is the sample standard deviation of blanks.

            :return: an array with the LOQ of each feature. If the LOQ cannot be estimated because
                there are no blank samples or the blank contains only missing values, it with
                return zero instead.

            """
            blank_query = self.matrix.query.filter(type=SampleType.SAMPLE).fetch_sample_ids()
            if not blank_query:
                return numpy.zeros(shape=(self.matrix.get_n_features(),), dtype=float)
            blank_samples = blank_query[0][1]
            blank_data = self.matrix.get_data(sample_ids=blank_samples)
            return metrics.loq(blank_data)

        def pca(
            self,
            *,
            n_components: int = 2,
            normalization: NormalizationMethod | str | None = None,
            scaling: ScalingMethod | str | None = None,
            return_loadings: bool = False,
            return_variance: bool = False,
        ):
            """Compute the PCA scores and loading of the data matrix.

            :param n_components: the number of Principal Components to compute.
            :param scaling: the scaling method applied to `X` columns. Refer to
                :py:class:`~tidyms2.core.enums.ScalingMethod` for the list of available scaling methods. If set to
                ``None``, no scaling is applied.
            :param normalization: One of the available normalization methods. Refer to
                :py:class:`~tidyms2.core.enums.NormalizationMethod` for the list of available normalization methods.
                If set to ``None``, do not perform row normalization.
            :param return_loadings: wether to return the PCA feature loadings.
            :param return_variance: wether to return the PC variance.
            :param kwargs: params passed to :py:func:`~tidyms2.core.matrix.DataMatrix.select_samples` method to
                perform a PCA analysis with a subset of samples.
            :return: A 2D array with PCA scores. If `return_loadings` is set to ``True`` also include a 2D array with
                the PCA loadings. If `return_pc_variance` is set to ``True`` also include a 1D array with PC variances.


            """
            if not self.matrix.get_process_status().missing_imputed:
                raise ValueError("PCA cannot be computed on matrices with NaN values.")

            return metrics.pca(
                self.matrix.get_data(),
                n_components=n_components,
                normalization=normalization,
                scaling=scaling,
                return_loadings=return_loadings,
                return_variance=return_variance,
            )

        def correlation(
            self,
            field: str,
            method: CorrelationMethod | str = CorrelationMethod.PEARSON,
        ) -> FloatArray1D:
            """Compute the correlation coefficient between features and a sample metadata field."""
            X = self.matrix.get_data()
            y = numpy.array(self.matrix.list_sample_field(field))
            if y.dtype.kind not in ["i", "u", "f"]:
                raise ValueError("Sample metadata field must contain numeric data.")
            return metrics.correlation(X, y, method)

    class Query:
        """Query API for selecting sample subsets using sample metadata.

        Note that the `filter` and `group_by` methods are implemented using using pure Python.
        If performance is required, consider using the
        :py:func:`~tidyms2.core.matrix.DataMatrix.Metrics.sql` method, which allows to query
        sample metadata and feature using a DuckDB SQL backend.

        """

        def __init__(self, matrix: DataMatrix):
            self.matrix = matrix
            self._filter_by = None
            self._group_by = None

        def _reset(self):
            self._filter_by = None
            self._group_by = None

        def filter(self, **kwargs) -> Self:
            """Select samples based on metadata fields.

            :param kwargs: key-value pairs used to select samples. Keys must be a
                :py:class:`~tidyms2.core.models.SampleMetadata` field. If a scalar value is passed,
                it is compared for equality with each sample metadata. If an list or tuple is passed,
                then the metadata field is checked for membership in the iterable. If multiple
                key-value pairs are provided, samples must pass checks for all pairs.

            """
            self._filter_by = kwargs
            return self

        def group_by(self, *args: str) -> Self:
            """Group samples based on metadata fields.

            :param args: the list of :py:class:`~tidyms2.core.models.SampleMetadata` fields used to
                create groups.

            """
            self._group_by = args
            return self

        def fetch_sample_ids(self) -> list[tuple[Sequence[str], list[str]]]:
            """Execute a sample query."""
            filter_by = self._filter_by or dict()
            filtered = self._filter_samples(self.matrix.samples, **filter_by)

            group_by = self._group_by or tuple()
            grouped = self._group_by_samples(filtered, *group_by)
            self._reset()
            return [(group, [x.id for x in group_samples]) for group, group_samples in grouped.items()]

        def sql(self, stmt: str):
            """Query data matrix metadata using SQL syntax.

            :param stmt: the SQL statement to query data.

            """
            # TODO: use DuckDB + python
            raise NotImplementedError

        @staticmethod
        def _filter_samples(samples: Sequence[Sample], **kwargs) -> list[Sample]:
            """Select a subset of samples using metadata."""
            filtered = list()
            for sample in samples:
                include = True
                for field, filter_value in kwargs.items():
                    if not hasattr(sample.meta, field):
                        msg = f"Sample id: {sample.id}. Field: {field}"
                        raise exceptions.SampleMetadataNotFound(msg)

                    value = getattr(sample.meta, field, None)

                    if include and isinstance(filter_value, (list, tuple)):
                        include = value in filter_value
                    elif include:
                        include = value == filter_value

                    if not include:
                        break

                if include:
                    filtered.append(sample)
            return filtered

        @staticmethod
        def _group_by_samples(samples: list[Sample], *args) -> dict[Sequence[str], list[Sample]]:
            """Select a subset of samples using metadata."""
            group_to_ids = dict()
            for sample in samples:
                group = list()
                for meta_field in args:
                    if not hasattr(sample.meta, meta_field):
                        msg = f"Sample id: {sample.id}. Field: {meta_field}"
                        raise exceptions.SampleMetadataNotFound(msg)
                    meta_value = getattr(sample.meta, meta_field)

                    group.append(meta_value)
                group = tuple(group)
                group_ids = group_to_ids.setdefault(group, list())
                group_ids.append(sample)
            return group_to_ids

    class IO:
        """Manage export and import of a data matrix in a variety of formats."""

        def __init__(self, matrix: DataMatrix):
            self.matrix = matrix

        def features_to_dict(self) -> dict:
            """Export feature metadata into a dataframe-friendly dictionary format."""
            raise NotImplementedError

        def features_to_csv(self, path: Path) -> None:
            """Write feature metadata into a csv file."""
            raise NotImplementedError

        def matrix_to_dict(self) -> dict:
            """Export data matrix into a dataframe-friendly dictionary format."""
            raise NotImplementedError

        def matrix_to_csv(self, path: Path) -> None:
            """Write data matrix into a csv file."""
            raise NotImplementedError

        def samples_to_dict(self) -> dict:
            """Export sample metadata into a dataframe-friendly dictionary format."""
            raise NotImplementedError

        def samples_to_csv(self, path: Path) -> None:
            """Write sample metadata into a csv file."""
            raise NotImplementedError

        @classmethod
        def from_csv(cls, samples_csv: Path, matrix_csv: Path, features_csv: Path) -> DataMatrix:
            """Create a data matrix instance from csv data."""
            raise NotImplementedError

    def __init__(
        self,
        samples: Sequence[Sample],
        features: Sequence[FeatureGroup],
        data: FloatArray,
        validate: bool = True,
    ):
        self._data = data
        self._samples = tuple(x for x in samples)
        self._features = tuple(x for x in features)
        self._status = DataMatrixProcessStatus()
        self._metrics = self.Metrics(self)
        self._query = self.Query(self)
        self._io = self.IO(self)
        if validate:
            self.validate()
            self.check_status()

    @property
    def metrics(self) -> Metrics:
        """Matrix metrics method getter."""
        return self._metrics

    @property
    def io(self) -> IO:
        """Matrix IO methods getter."""
        return self._io

    @property
    def query(self) -> Query:
        """Matrix query methods getter."""
        return self._query

    @property
    def samples(self) -> Sequence[Sample]:
        """The list of samples in the matrix."""
        return self._samples

    @property
    def features(self) -> Sequence[FeatureGroup]:
        """The list of features in the matrix."""
        return self._features

    def get_n_features(self) -> int:
        """Retrieve the number of feature groups in the data matrix."""
        return len(self._features)

    def get_n_samples(self) -> int:
        """Retrieve the number of samples in the data matrix."""
        return len(self._samples)

    def get_process_status(self) -> DataMatrixProcessStatus:
        """Retrieve the current data matrix status."""
        return self._status

    @deprecated("Use the 'samples' property  instead.")
    def list_samples(self) -> Sequence[Sample]:
        """List all samples in the data matrix."""
        return self.samples

    def list_sample_field(self, field: str) -> list:
        """Retrieve the field value from all samples.

        If a sample does not contain the queried field, it returns ``None``.

        :param field: the field name to fetch

        """
        return [getattr(x.meta, field, None) for x in self._samples]

    @deprecated("Use the `features` property instead.")
    def list_features(self) -> Sequence[FeatureGroup]:
        """List all features in the data matrix."""
        return self.features

    def add_columns(self, *columns: FeatureVector) -> None:
        """Add columns to the data matrix.

        :param features: the list of columns to add

        """
        ...

    def check_status(self) -> None:
        """Check and update the data matrix status."""
        self._status.missing_imputed = numpy.all(~numpy.isnan(self.get_data())).item()

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
            row = FeatureVector(data=self._data[:, index], feature=self._features[index], index=index)
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
            row = SampleVector(data=self._data[index], sample=self._samples[index], index=index)
            rows.append(row)
        return rows

    def get_data(
        self,
        sample_ids: list[str] | None = None,
        feature_groups: list[int] | None = None,
    ) -> FloatArray:
        """Retrieve the matrix data in numpy format.

        Each rows in the array is associated with a sample and each column is associated with a feature.

        :param sample_ids: if provided, return a copy of the data array using the subset of samples provided.
        :param feature_groups: if provided, return a copy of the data array using the subset of feature provided.

        """
        if sample_ids is None and feature_groups is None:
            return self._data
        elif sample_ids is None and feature_groups is not None:
            feature_idx = self.get_feature_index(*feature_groups)
            return self._data[:, feature_idx].copy()
        elif sample_ids is not None and feature_groups is None:
            sample_idx = self.get_sample_index(*sample_ids)
            return self._data[sample_idx].copy()
        else:
            assert feature_groups is not None
            assert sample_ids is not None
            feature_idx = self.get_feature_index(*feature_groups)
            sample_idx = self.get_sample_index(*sample_ids)
            return self._data[sample_idx, feature_idx].copy()

    def get_feature_index(self, *groups: int) -> list[int]:
        """Retrieve the list of indices in the data associated with feature groups.

        :param groups: the list of feature groups to search

        """
        try:
            return [self._feature_index()[x] for x in groups]
        except KeyError as e:
            raise exceptions.FeatureGroupNotFound from e

    def get_sample_index(self, *sample_ids: str) -> list[int]:
        """Retrieve the list of indices in the data associated with samples.

        :param sample_ids: the list of samples to search

        """
        try:
            return [self._sample_index()[x] for x in sample_ids]
        except KeyError as e:
            raise exceptions.SampleNotFound from e

    def get_feature(self, group: int) -> FeatureGroup:
        """Retrieve a group from the data matrix.

        :param group: the group label of the feature to retrieve
        :raise FeatureGroupNotFound: if the feature is not found in the data matrix.

        """
        if not self.has_feature(group):
            raise exceptions.FeatureGroupNotFound(group)
        idx = self._feature_index()[group]
        return self._features[idx]

    def get_sample(self, sample_id: str) -> Sample:
        """Retrieve a sample from the data matrix.

        :param sample_id: the id of the sample to retrieve
        :raises SampleNotFound: if no sample with the provided id exists in the matrix

        """
        if not self.has_sample(sample_id):
            raise exceptions.SampleNotFound(sample_id)
        idx = self._sample_index()[sample_id]
        return self._samples[idx]

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
        self.check_status()

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
            samples.extend(m.samples)
        data = numpy.vstack([m.get_data() for m in matrices])
        features = matrices[0].features
        return cls(samples, features, data, validate=True)

    def create_submatrix(
        self,
        sample_ids: list[str] | None = None,
        feature_groups: list[int] | None = None,
    ) -> Self:
        """Create a submatrix using a subset of samples and/or features."""
        if sample_ids is None and feature_groups is None:
            submatrix_samples = self.samples
            submatrix_features = self.features
            data = self.get_data()
        elif sample_ids is None and feature_groups is not None:
            submatrix_samples = self.samples
            submatrix_features = [self.get_feature(x) for x in feature_groups]
            data = self.get_data(feature_groups=feature_groups)
        elif sample_ids is not None and feature_groups is None:
            submatrix_samples = [self.get_sample(x) for x in sample_ids]
            submatrix_features = self.features
            data = self.get_data(sample_ids=sample_ids)
        else:
            assert feature_groups is not None
            assert sample_ids is not None
            submatrix_samples = [self.get_sample(x) for x in sample_ids]
            submatrix_features = [self.get_feature(x) for x in feature_groups]
            data = self.get_data(feature_groups=feature_groups, sample_ids=sample_ids)

        return self.__class__(submatrix_samples, submatrix_features, data, validate=False)

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

    Sample.validate_samples(*samples)


def sort_matrix_columns(
    features: Sequence[FeatureGroup], data: FloatArray
) -> tuple[Sequence[FeatureGroup], FloatArray]:
    """Sort data using feature group label."""
    sorted_index = [k for k, _ in sorted(enumerate(features), key=lambda x: x[1].group)]
    return tuple(features[x] for x in sorted_index), data[:, sorted_index]


def sort_matrix_rows(samples: Sequence[Sample], data: FloatArray) -> tuple[Sequence[Sample], FloatArray]:
    """Sort data using sample run order."""
    sorted_index = [k for k, _ in sorted(enumerate(samples), key=lambda x: x[1].meta.order)]
    return tuple(samples[x] for x in sorted_index), data[sorted_index]


class BaseVector(pydantic.BaseModel):
    """A container for 1D vector data."""

    model_config = pydantic.ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data: FloatArray1D = pydantic.Field(repr=False)
    """The vector data."""

    index: int
    """The data matrix index associated with the vector."""


class FeatureVector(BaseVector):
    """Data matrix column view."""

    feature: FeatureGroup
    """The feature information associated with the matrix column."""


class SampleVector(BaseVector):
    """Data matrix row."""

    sample: Sample
    """The sample associated with the row."""
