"""Assay configuration."""

from pathlib import Path

import pydantic

from ..core.models import Sample


class AssayConfiguration(pydantic.BaseModel):
    """The assay configuration model."""

    samples: list[Sample]
    """The list of samples included in the assay."""

    sample_root: Path | None = None
    """The base directory to search for sample paths."""

    storage_type: str
    """the name of a registered storage backend."""

    storage_config: dict
    """The configuration passed to the storage backend."""

    sample_processor_type: str
    """the name of a registered sample processor backend."""

    sample_processor_config: dict
    """The configuration passed to the storage processor."""

    sample_pipeline: list[dict]
    """The configuration of sample operators."""

    assay_pipeline: list[dict]
    """The configuration of assay operators."""

    matrix_pipeline: list[dict]
    """The configuration of the matrix operators."""
