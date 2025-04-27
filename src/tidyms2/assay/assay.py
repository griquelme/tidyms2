"""A manager class for processing multiple samples."""

from collections import OrderedDict
from logging import getLogger
from typing import Generic, Iterable, Literal, overload
from uuid import UUID

from ..core import exceptions
from ..core.matrix import DataMatrix
from ..core.models import Annotation, FeatureType, RoiType, Sample
from ..core.operators.assay import AssayOperator, MissingImputer
from ..core.operators.pipeline import Pipeline
from ..core.storage import AssayStorage, SampleStorage
from .executors import SampleProcessor
from .utils import create_feature_groups, create_matrix_data

logger = getLogger("assay")


class Assay(Generic[RoiType, FeatureType]):
    """The assay class.

    :param id: an identifier for the assay
    :param assay_storage: the storage class for assay data.

    """

    class _PipelineContainer:
        def __init__(self, id: str):
            self.sample = Pipeline(f"{id}-sample-pipeline")
            self.assay = Pipeline(f"{id}-assay-pipeline")

    def __init__(
        self,
        id: str,
        assay_storage: AssayStorage[RoiType, FeatureType],
        sample_processor: SampleProcessor[RoiType, FeatureType],
        sample_storage_type: type[SampleStorage[RoiType, FeatureType]],
        sample_storage_config: dict | None = None,
    ):
        self.id = id
        self._sample_queue: OrderedDict[str, Sample] = OrderedDict()
        self._storage = assay_storage
        self._sample_processor = sample_processor
        self._sample_storage_type = sample_storage_type
        self._sample_storage_config = sample_storage_config or dict()
        self.pipes = self._PipelineContainer(id)

    def add_samples(self, *samples: Sample) -> None:
        """Add samples to the assay sample processing queue.

        :param sample: the samples to add
        """
        # first we validate new samples
        Sample.validate_samples(*samples)

        # then we also include processed samples
        all_samples = self._storage.list_samples() + list(samples)
        try:
            Sample.validate_samples(*all_samples)
        except exceptions.RepeatedIdError as e:
            msg = "Cannot add samples. Processed samples with the same id already exist."
            raise exceptions.RepeatedIdError(msg) from e
        except exceptions.RepeatedSampleOrder as e:
            msg = "Cannot add samples. Processed samples with the run order already exist."
            raise exceptions.RepeatedIdError(msg) from e

        for sample in samples:
            logger.info(f"Added sample `{sample.id}` with path `{sample.path}` to {self.id}.")
            self._sample_queue[sample.id] = sample

    def create_data_matrix(self, matrix_value: str | None = None) -> DataMatrix:
        """Create a data matrix using data from the assay pipeline.

        :param matrix_value: the descriptor name used to build the matrix data. If not
            provided, the descriptor `area` will be used as the matrix value.

        """
        matrix_value = matrix_value or "area"
        descriptors = self.fetch_feature_table()
        if matrix_value not in descriptors:
            raise ValueError(f"{matrix_value} is not a valid feature descriptor.")
        values = descriptors[matrix_value]
        groups = self._storage.fetch_feature_groups()
        samples = self.fetch_samples()
        annotations = self.fetch_feature_annotations()
        fill_values = self._storage.fetch_fill_values()

        data = create_matrix_data(values, annotations, samples, fill_values)
        return DataMatrix(samples, groups, data)

    def fetch_samples(self, queued: bool = False) -> list[Sample]:
        """Retrieve a list of queued or processed samples in the assay.

        :param queued: if set to ``True``, return queued samples. Otherwise, fetch processed samples.

        """
        if queued:
            return [x for x in self._sample_queue.values()]

        return self._storage.list_samples()

    def fetch_feature_annotations(self) -> list[Annotation]:
        """Retrieve a list with feature annotations."""
        return self._storage.fetch_annotations()

    def fetch_feature_table(self) -> dict[str, list[float]]:
        """Fetch the feature descriptors table."""
        return self._storage.fetch_descriptors()

    @overload
    def fetch_features(self, by: Literal["id"], keys: Iterable[UUID]) -> list[FeatureType]: ...

    @overload
    def fetch_features(self, by: Literal["sample"], keys: str) -> list[FeatureType]: ...

    @overload
    def fetch_features(self, by: Literal["group"], keys: int) -> list[FeatureType]: ...

    def fetch_features(self, by: str, keys) -> list[FeatureType]:
        """Fetch extracted features."""
        if by == "id":
            return self._storage.fetch_features_by_id(*keys)
        elif by == "sample":
            return self._storage.fetch_features_by_sample(keys)
        elif by == "group":
            return self._storage.fetch_features_by_group(keys)
        else:
            raise ValueError(f"Valid values for `by` are `sample`, 'group' or `id`. Got {by}.")

    @overload
    def fetch_rois(self, by: Literal["id"], keys: Iterable[UUID]) -> list[RoiType]: ...

    @overload
    def fetch_rois(self, by: Literal["sample"], keys: str) -> list[RoiType]: ...

    def fetch_rois(self, by: str, keys):
        """Fetch extracted ROIs.

        :param by: The criteria to fetch ROIs, can be either ``"id"`` or ``"sample"``.
        :param keys: the keys used to fetch ROIs. If `by` is set to ``"id"``, then keys must be a list of
            ROI ids. If `by` is set to ``"sample"`` it must be a single sample id.

        """
        if by == "sample":
            return self._storage.fetch_rois_by_sample(keys)
        elif by == "id":
            return self._storage.fetch_rois_by_id(*keys)
        else:
            raise ValueError(f"Valid values for parameter `by` are `sample` or `id`. Got {by}.")

    def process_samples(self, sample_ids: list[str] | None = None) -> None:
        """Apply sample pipeline to queued samples.

        :param sample_ids: process only a subset of samples in the queue. This parameter is useful
            to explore sample processing parameters. If not provided, process all samples in the
            queue.

        """
        if not self._sample_queue:
            logger.warning("No samples to process: sample queue is empty.")
            return
        if sample_ids is None:
            sample_ids = list(self._sample_queue)

        samples = list()
        for id_ in sample_ids:
            try:
                sample = self._sample_queue[id_]
                sample_storage = self._sample_storage_type(
                    sample,
                    self._storage.get_roi_type(),
                    self._storage.get_feature_type(),
                    **self._sample_storage_config,
                )
                samples.append(sample_storage)
            except KeyError:
                raise exceptions.SampleNotFound(id_)

        logger.info("Starting sample processing pipeline with operators:")
        for k, op in enumerate(self.pipes.sample.operators, start=1):
            logger.info(f"{k}. {op.id}: {op}")
        self._sample_processor.execute(self._storage, self.pipes.sample, *samples)
        self._remove_from_sample_queue()
        logger.info("Successfully processed all samples.")

    def process_assay(self) -> None:
        """Apply assay pipeline to assay data."""
        if self._sample_queue:
            msg = "There are unprocessed samples in the sample queue. Run `process_samples` to fix this error."
            raise exceptions.UnprocessedSampleError(msg)

        feature_groups_created = False
        for op in self.pipes.assay.operators:
            assert isinstance(op, AssayOperator)  # helping pyright
            if isinstance(op, MissingImputer):
                self._create_feature_groups()
                feature_groups_created = True
            logger.info(f"Applying {op.id} to assay...")
            op.apply(self._storage)

        if not feature_groups_created:
            self._create_feature_groups()

    def _remove_from_sample_queue(self, ids: list[str] | None = None) -> None:
        if ids is None:
            self._sample_queue.clear()
        else:
            for id_ in ids:
                self._sample_queue.pop(id_)

    def _create_feature_groups(self) -> None:
        descriptors = self.fetch_feature_table()
        annotations = self.fetch_feature_annotations()
        feature_groups = create_feature_groups(descriptors, annotations)
        self._storage.add_feature_groups(*feature_groups)
