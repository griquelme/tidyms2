"""Utilities to trigger sample pipeline execution."""

import concurrent.futures
from logging import getLogger
from typing import Generic, Protocol

import pydantic

from ..core.exceptions import SampleProcessorError
from ..core.models import FeatureType, RoiType
from ..core.operators.pipeline import Pipeline
from ..core.storage import AssayStorage, SampleStorage, delete_empty_rois

logger = getLogger(__name__)


class SampleProcessor(Protocol, Generic[RoiType, FeatureType]):
    """Base sample executor class."""

    def execute(
        self,
        storage: AssayStorage[RoiType, FeatureType],
        pipe: Pipeline,
        *samples: SampleStorage[RoiType, FeatureType],
    ) -> None:
        """Apply pipeline to multiple samples and store results into an assay storage."""
        ...


class SequentialSampleProcessor(pydantic.BaseModel, Generic[RoiType, FeatureType]):
    """Execute a sample pipeline."""

    remove_empty_roi: bool = True
    """Flag to delete ROIs that do not contain any features."""

    def execute(
        self,
        storage: AssayStorage[RoiType, FeatureType],
        pipe: Pipeline,
        *samples: SampleStorage[RoiType, FeatureType],
    ) -> None:
        """Apply a pipeline to a sample."""
        n_samples = len(samples)
        for k, sample in enumerate(samples, start=1):
            logger.info(f"Processing sample `{sample.get_sample().id}` ({k}/{n_samples}).")
            pipe.apply(sample)

            if self.remove_empty_roi:
                delete_empty_rois(sample)

            storage.add_sample_data(sample)


class ParallelSampleProcessor(pydantic.BaseModel, Generic[RoiType, FeatureType]):
    """Execute a sample pipeline."""

    remove_empty_roi: bool = True
    """Flag to delete ROIs that do not contain any features."""

    max_workers: pydantic.PositiveInt = 2
    """The maximum number of process spawned simultaneously to process samples."""

    def execute(
        self,
        storage: AssayStorage[RoiType, FeatureType],
        pipe: Pipeline,
        *samples: SampleStorage[RoiType, FeatureType],
    ) -> None:
        """Apply a pipeline to multiple samples and store the results in an assay storage."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            n_samples = len(samples)
            futures = [executor.submit(_sample_executor_worker, pipe.copy(), sample) for sample in samples]
            for k, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                data = future.result()
                logger.info(f"Processed sample `{data.get_sample().id}` ({k}/{n_samples}).")
                if self.remove_empty_roi:
                    delete_empty_rois(data)
                storage.add_sample_data(data)


def _sample_executor_worker(
    pipeline: Pipeline, data: SampleStorage[RoiType, FeatureType]
) -> SampleStorage[RoiType, FeatureType]:
    """Apply pipeline to a sample data instance."""
    try:
        pipeline.apply(data)
        delete_empty_rois(data)
    except Exception as e:
        raise SampleProcessorError(f"Failed to process sample {data.get_sample().id}") from e
    return data
