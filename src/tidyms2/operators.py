"""Data matrix operators."""

from abc import abstractmethod
from typing import Generator, Self

import numpy
import pydantic

from tidyms2.core.dataflow import DataMatrixProcessStatus
from tidyms2.core.enums import MSInstrument, Polarity, SeparationMode

from .core.enums import AggregationMethod, CorrelationMethod, SampleType
from .core.matrix import DataMatrix
from .core.operators.matrix import ColumnFilter, MatrixTransformer, RowFilter
from .core.utils.numpy import FloatArray, FloatArray1D
from .core.utils.transformation import aggregate


class SampleFilter(RowFilter):
    """Remove samples using metadata properties.

    The :py:func:`~tidyms2.core.matrix.DataMatrix.Query.filter` query function is used to
    select which samples to remove. All samples matching the filter criteria
    will be removed from the data matrix.

    **Examples**

    Remove samples from `group-c` or `group-d`:

    .. code-block:: python

        from tidyms2.operators.SampleFilter

        op = SampleFilter(group=["group-c", "group-d"]})

    Remove all samples from the second analytical batch:

    .. code-block:: python

        from tidyms2.operators.SampleFilter

        op = SampleFilter(batch=2})

    Remove all blank samples:

    .. code-block:: python

        from tidyms2.operators.SampleFilter

        op = SampleFilter(type="blank")

    """

    model_config = pydantic.ConfigDict(extra="allow")

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()

    def _create_remove_list(self, data: DataMatrix) -> list[str]:
        filters = self.model_extra or dict()
        query = data.query.filter(**filters).fetch_sample_ids()
        return query[0][1] if query else list()


class FeatureFilter(ColumnFilter):
    """Remove feature groups based on descriptor values.

    This filter compares feature descriptors against lower and upper bounds and remove all features
    outside these values.

    **Examples**

    Remove all features with retention time lower than 30 s:

    .. code-block:: python

        from math import inf

        from tidyms2.operators import FeatureFilter

        op = FeatureFilter(rt=(30.0, inf))

    Remove all features with peak widths larger than 20 s:

    .. code-block:: python

        from tidyms2.operators import FeatureFilter

        op = FeatureFilter(width=(0.0, 20.0))

    """

    model_config = pydantic.ConfigDict(extra="allow")

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()

    @pydantic.model_validator(mode="after")
    def _ensure_tuple_in_filters(self):
        if self.model_extra is None:
            return self

        for field, value in self.model_extra.items():
            assert isinstance(value, tuple), "filter bounds must be a pair"
            assert len(value) == 2
            lb, ub = value
            assert lb <= ub, "The values in the filter should be a pair of lower and upper bounds."
        return self

    def _create_remove_list(self, data: DataMatrix) -> list[int]:
        filters = self.model_extra or dict()
        return [x.group for x in data.features if not x.has_descriptors_in_range(**filters)]


class BlankCorrector(MatrixTransformer):
    """Removes blank contribution from samples.

    The blank contribution is estimated as the aggregation of all samples that match the `blank_group`.
    If `blank_group` is not defined, all samples with blank type are used. The aggregation operation
    is defined by the `aggregation` parameter. By default, the blank contribution is estimated as the
    sample mean of all blank samples.

    Samples or features may be included from the correction by using the parameters `exclude_samples`
    and `exclude_features` respectively.

    """

    aggregation: AggregationMethod = AggregationMethod.MEAN
    """The method used to compute the blank contribution."""

    blank_groups: list[str] | None = None
    """The sample groups used to compute the blank contribution. If not specified, all samples
    from with BLANK :term:`sample type` will be used compute the blank contribution."""

    exclude_samples: list[str] | None = None
    """A list of sample ids to exclude from blank correction."""

    exclude_features: list[int] | None = None
    """A list of feature groups to exclude from blank correction."""

    apply_to_blanks: bool = False
    """Wether to apply the blank correction to samples used to compute the blank contribution."""

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()

    def _find_blank_sample_ids(self, data: DataMatrix) -> list[str]:
        if self.blank_groups is None:
            blank_samples_query = data.query.filter(type=SampleType.BLANK)
        else:
            blank_samples_query = data.query.filter(group=self.blank_groups)

        blank_sample_query_results = blank_samples_query.fetch_sample_ids()
        if not blank_sample_query_results:
            raise ValueError("No samples found in the blank group.")
        return blank_sample_query_results[0][1]

    def _transform_matrix(self, data: DataMatrix) -> FloatArray:
        blank_ids = self._find_blank_sample_ids(data)
        blank_data = data.get_data(sample_ids=blank_ids)
        blank_contribution = aggregate(blank_data, self.aggregation, axis=0)

        include_samples = list()
        exclude_samples = self.exclude_samples or list()
        for sample in data.samples:
            if sample.id in exclude_samples:
                continue
            if not self.apply_to_blanks and sample.id in blank_ids:
                continue
            include_samples.append(sample.id)
        include_samples_index = data.get_sample_index(*include_samples)

        include_features = list()
        exclude_features = self.exclude_features or list()
        for feature in data.features:
            if feature.group in exclude_features:
                continue
            include_features.append(feature.group)
        include_features_index = data.get_feature_index(*include_features)

        X = data.get_data().copy()
        for idx in include_samples_index:
            X[idx, include_features_index] -= blank_contribution[include_features_index]
        return numpy.maximum(X, 0.0)


class MetricsFilter(ColumnFilter):
    """Abstract base class for all metric-based filters."""

    lb: float = 0.0
    """The filter lower bound. Features with metric values lower than this value will be removed.
    Must be lower than `ub`."""

    ub: float = 0.25
    """The filter upper bound. Features with metric values greater than this value will be removed."""

    group_by: list[str] | None = None
    """How to group samples using sample metadata fields. If this parameter is specified, the feature
    metric will be computed for all groups and then aggregated using the `agg` parameter. For more details
    refer to the :py:func:`~tidyms2.core.matrix.DataMatrix.Query.group_by` method."""

    filter: dict | None = None
    """How to filter samples using sample metadata fields. This parameter is used to define which samples
    to include when computing the metric value. If not defined, no filtering is applied. For more details
    refer to the :py:func:`~tidyms2.core.matrix.DataMatrix.Query.filter` method."""

    aggregation: AggregationMethod = AggregationMethod.MIN
    """When the `group_by` parameter is defined, this parameter defines the aggregation method used to
    compute the feature value that will be compared against the filter lower and upper bounds. If
    `group_by` is not specified, this parameter is ignored."""

    @pydantic.model_validator(mode="after")
    def _ensure_lb_lower_than_ub(self) -> Self:
        if self.ub < self.lb:
            raise ValueError("Lower bound must be lower than the upper bound.")
        return self

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        return cls()

    @abstractmethod
    def _get_filter(self) -> dict | None: ...

    @abstractmethod
    def _get_group_by(self) -> list[str] | None: ...

    def _iter_groups(self, data: DataMatrix) -> Generator[DataMatrix, None, None]:
        filters = self._get_filter() or dict()
        groups = self._get_group_by() or tuple()
        for _, ids in data.query.filter(**filters).group_by(*groups).fetch_sample_ids():
            yield data.create_submatrix(sample_ids=ids)

    @abstractmethod
    def _compute_metric(self, data: DataMatrix) -> FloatArray1D: ...

    def aggregate_metric(self, data: DataMatrix) -> FloatArray:
        metric_list = list()
        for matrix in self._iter_groups(data):
            metric_list.append(self._compute_metric(matrix))

        if not metric_list:
            raise ValueError("No sample groups found.")

        metric_array = numpy.vstack(metric_list)
        return aggregate(metric_array, self.aggregation)

    def _create_remove_list(self, data: DataMatrix) -> list[int]:
        metric = self.aggregate_metric(data)
        ind = numpy.where((metric < self.lb) | (metric > self.ub))[0]
        return [data.features[x.item()].group for x in ind]


class CVFilter(MetricsFilter):
    """Filter features using the coefficient of variation (CV) metric.

    This filter allows to compute the CV on multiple sample subsets by using the matrix
    :py:class:`~tidyms2.core.matrix.DataMatrix.Query` utilities. The parameters `group_by`
    and `filter` are used to this end. Refer to the :ref:`matrix-query-guide` guide for
    more details.

    If neither the `group_by` and `filter` parameters are defined, the CV is computed on
    all samples labelled with the QC :term:`sample type`. If QC samples are found, a
    ``ValuerError`` is raised.

    If multiple sample subsets are used, an aggregated CV value is obtained
    for each feature by using the `aggregation` parameter, which defines the aggregation
    operation. By default, the minimum CV of all groups is kept. The aggregated CV
    value is then compared against the filter lower and upper bounds, defined by the
    `lb` and `ub` parameters. Features with values outside these bounds are removed from
    the data matrix.

    The :ref:`data-curation-guide` guide covers several examples of this filter.

    .. seealso::
        :py:func:`~tidyms2.core.matrix.DataMatrix.Metrics.cv`: The CV metric computation
        method.

    """

    robust: bool = True
    """Compute the CV using a robust estimator."""

    def _get_filter(self) -> dict | None:
        if self.filter is None and self.group_by is None:
            return {"type": SampleType.TECHNICAL_QC}
        return self.filter

    def _get_group_by(self) -> list[str] | None:
        return self.group_by

    def _compute_metric(self, data: DataMatrix) -> FloatArray1D:
        return data.metrics.cv(robust=self.robust)


class DRatioFilter(MetricsFilter):
    """Filter features using the D-ratio metric.

    D-ratio is computed for each feature and features are removed if they fall outside the
    filters lower and upper bounds. By default, the sample variance is estimated using all
    samples with sample type. The technical variation is estimated all QC samples.

    The :ref:`data-curation-guide` guide covers several examples of this filter.

    .. seealso::
        :py:func:`~tidyms2.core.matrix.DataMatrix.Metrics.dratio`: The D-Ratio metric computation
        method.

    """

    lb: pydantic.NonNegativeFloat = 0.0
    ub: pydantic.NonNegativeFloat = 1.0

    robust: bool = True
    """Compute the D-ratio using a robust estimator."""

    sample_groups: list[str] | None = None
    """A list of :term:`sample group` used to estimate the biological variation. If not defined, all samples
    with sample type are used."""

    qc_groups: list[str] | None = None
    """A list of :term:`sample group` used to estimate the technical variation. If not defined, all samples
    with QC type are used."""

    def _get_filter(self):
        return None

    def _get_group_by(self):
        return None

    @pydantic.field_validator("filter", "group_by", mode="before")
    @classmethod
    def _ensure_none(cls, val, info: pydantic.ValidationInfo):
        if val is not None:
            raise ValueError(f"{info.field_name} usage is not supported for this filter.")
        return val

    def _compute_metric(self, data: DataMatrix) -> FloatArray1D:
        return data.metrics.dratio(self.sample_groups, self.qc_groups, robust=self.robust)


class DetectionRateFilter(MetricsFilter):
    """Filter features using the detection rate (DR) metric.

    This filter allows to compute the DR on multiple sample subsets by using the matrix
    :py:class:`~tidyms2.core.matrix.DataMatrix.Query` utilities. The parameters `group_by`
    and `filter` are used to this end. Refer to the :ref:`matrix-query-guide` guide for
    more details.

    If neither the `group_by` and `filter` parameters are defined, samples are grouped
    by :term:`sample group` and filtered by :term:`sample type`, keeping only samples
    with sample type, before computing the DR on each group. If no samples with this type
    are found, a ``ValuerError`` is raised.

    By default, the maximum DR value of all groups is kept. The aggregated DR
    value is then compared against the filter lower and upper bounds, defined by the
    `lb` and `ub` parameters. Features with values outside these bounds are removed from
    the data matrix.

    The :ref:`data-curation-guide` guide covers the usage of this filter.

    .. seealso::
        :py:func:`~tidyms2.core.matrix.DataMatrix.Metrics.detection_rate`: The detection
        rate metric computation method.

    """

    lb: pydantic.NonNegativeFloat = 0.9
    ub: pydantic.NonNegativeFloat = 1.0

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    threshold: float | FloatArray1D | None = None
    """The threshold value to define a feature as detected. If a scalar is provided, then
    the same threshold is applied to all features. Using an array allows to use an unique
    threshold for each feature. If the threshold is set to ``None``, the threshold value
    will be estimated using :py:func:`~tidyms2.core.matrix.DataMatrix.Metrics.lod`."""

    def _get_filter(self) -> dict | None:
        if self.group_by is None and self.filter is None:
            return {"type": SampleType.SAMPLE}
        return self.filter

    def _get_group_by(self) -> list[str] | None:
        if self.group_by is None and self.filter is None:
            return ["group"]
        return self.group_by

    def _compute_metric(self, data: DataMatrix) -> FloatArray1D:
        threshold = self.threshold or data.metrics.lod()
        return data.metrics.detection_rate(threshold=threshold)


class CorrelationFilter(MetricsFilter):
    """Filter features based on the correlation with sample metadata fields.

     This filter allows to compute the correlation with a sample metadata field on multiple
     sample subsets by using the matrix :py:class:`~tidyms2.core.matrix.DataMatrix.Query`
     utilities. The parameters `group_by` and `filter` are used to this end. Refer to the
     :ref:`matrix-query-guide` guide for more details.

    If neither the `group_by` and `filter` parameters are defined, the correlation is
    computed using all samples.

    This filter will fail if the data contain NaN values.

    **Examples**

    Filter features using the correlation between the run order and intensity on QC samples:

    .. code-block:: python

        from tidyms2.operators import CorrelationFilter

        op = CorrelationFilter(filter={"type": "qc"}, field="order")

    Filter features using the correlation between the dilution factor and intensity of diluter QC samples:

    .. code-block:: python

        from tidyms2.operators import CorrelationFilter

        op = CorrelationFilter(filter={"type": "dqc"}, field="dilution")

    """

    field: str
    """The sample metadata field used to compute the correlation."""

    method: CorrelationMethod = CorrelationMethod.PEARSON
    """The method used to compute the correlation."""

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus(missing_imputed=True)

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus(missing_imputed=True)

    def _get_filter(self):
        return None

    def _get_group_by(self):
        return None

    def _compute_metric(self, data: DataMatrix) -> FloatArray1D:
        return data.metrics.correlation(self.field, self.method)
