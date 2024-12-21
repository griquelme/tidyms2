"""Utilities to trigger sample pipeline execution."""

import concurrent.futures
from logging import getLogger
from typing import Generator, Generic, Protocol

import pydantic

from ..core.exceptions import SampleProcessorError
from ..core.models import FeatureType, RoiType, Sample
from ..core.operators import Pipeline
from ..core.storage import AssayStorage
from ..storage import OnMemorySampleStorage

logger = getLogger(__name__)


class SampleProcessor(Protocol, Generic[RoiType, FeatureType]):
    """Base sample executor class."""

    def execute(self, storage: AssayStorage[RoiType, FeatureType], pipe: Pipeline, *samples: Sample) -> None:
        """Apply pipeline to multiple samples and store results into an assay storage."""
        ...


class SequentialSampleProcessor(Generic[RoiType, FeatureType]):
    """Execute a sample pipeline."""

    def execute(self, storage: AssayStorage, pipe: Pipeline, *samples: Sample) -> None:
        """Apply a pipeline to a sample."""
        n_samples = len(samples)
        for k, sample in enumerate(samples):
            logger.info(f"Processing `{sample.id}` ({k}/{n_samples}).")
            data = OnMemorySampleStorage(sample, storage.get_roi_type(), storage.get_feature_type())
            pipe.apply(data)
            storage.add_sample_data(data)


class ParallelSampleProcessor(pydantic.BaseModel, Generic[RoiType, FeatureType]):
    """Execute a sample pipeline."""

    max_workers: pydantic.PositiveInt = 2
    """The maximum number of process spawned simultaneously to process samples."""

    def execute(self, storage: AssayStorage[RoiType, FeatureType], pipe: Pipeline, *samples: Sample) -> None:
        """Apply a pipeline to multiple samples and store the results in an assay storage."""

        def iterator() -> Generator[tuple[Pipeline, OnMemorySampleStorage[RoiType, FeatureType]], None, None]:
            """Provide independent pipeline instances to subprocess worker."""
            for sample in samples:
                yield pipe.copy(), OnMemorySampleStorage(sample, storage.get_roi_type(), storage.get_feature_type())

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            n_samples = len(samples)
            futures = [executor.submit(_sample_executor_worker, pipe, data) for pipe, data in iterator()]
            for k, future in enumerate(concurrent.futures.as_completed(futures)):
                data = future.result()
                logger.info(f"Processing `{data.get_sample().id}` ({k}/{n_samples}).")
                storage.add_sample_data(data)


def _sample_executor_worker(
    pipeline: Pipeline, data: OnMemorySampleStorage[RoiType, FeatureType]
) -> OnMemorySampleStorage[RoiType, FeatureType]:
    """Apply pipeline to a sample data instance."""
    try:
        pipeline.apply(data)
    except Exception as e:
        raise SampleProcessorError(f"Failed to process sample {data.get_sample().id}") from e
    return data
