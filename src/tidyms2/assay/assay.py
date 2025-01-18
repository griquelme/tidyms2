"""A manager class for processing multiple samples."""

from collections import OrderedDict
from logging import getLogger
from typing import Generic, Iterable, Literal, overload
from uuid import UUID

from ..core.exceptions import UnprocessedSampleError
from ..core.models import Annotation, FeatureType, RoiType, Sample
from ..core.operators.assay import AssayOperator, MissingImputer
from ..core.operators.pipeline import Pipeline
from ..core.storage import AssayStorage
from .executors import SampleProcessor

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
        sample_processor: SampleProcessor,
    ):
        self.id = id
        self._sample_queue: OrderedDict[str, Sample] = OrderedDict()
        self._storage = assay_storage
        self._sample_processor = sample_processor
        self.pipes = self._PipelineContainer(id)

    def add_samples(self, *samples: Sample) -> None:
        """Add samples to the assay sample processing queue.

        :param sample: the samples to add
        """
        for sample in samples:
            logger.info(f"Added sample `{sample.id}` with path `{sample.path}` to {self.id}.")
            self._sample_queue[sample.id] = sample

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

    def process_samples(self) -> None:
        """Apply sample pipeline to queued samples."""
        if self._sample_queue:
            samples = list(self._sample_queue.values())
            self._sample_processor.execute(self._storage, self.pipes.sample, *samples)
            self._empty_sample_queue()
        else:
            logger.warning("No samples to process: sample queue is empty.")

    def process_assay(self) -> None:
        """Apply assay pipeline to assay data."""
        if self._sample_queue:
            raise UnprocessedSampleError(", ".join(x for x in self._sample_queue))

        for op in self.pipes.assay.operators:
            assert isinstance(op, AssayOperator)
            if isinstance(op, MissingImputer):
                pass
            op.apply(self._storage)

    def _empty_sample_queue(self) -> None:
        self._sample_queue = OrderedDict()
