"""TidyMS core exceptions."""


class UnprocessedSampleError(ValueError):
    """Exception raised when an assay processing pipeline is applied on an assay with queued sample data."""


class FeatureGroupNotFound(ValueError):
    """Exception raised when a feature group is not found in an assay storage."""


class FeatureNotFound(ValueError):
    """Exception raised when a Feature is not found in a sample storage or assay storage."""


class InvalidFeatureDescriptor(ValueError):
    """Exception raised when a non existing feature descriptor name is requested."""


class PipelineConfigurationError(ValueError):
    """Exception raised when an invalid configuration is set in a pipeline."""


class RepeatedIdError(ValueError):
    """Exception raised when trying to add a resource with an existing id."""


class RoiNotFound(ValueError):
    """Exception raised when a Roi is not found in a sample storage or assay storage."""


class ProcessStatusError(ValueError):
    """Exception raised when an action cannot be performed on sample data due to incorrect processing status."""


class RegistryError(ValueError):
    """Exception raised when an entry is not found in a registry."""


class SampleNotFound(ValueError):
    """Exception raised when a sample is not found in assay/matrix storage."""


class SnapshotNotFound(ValueError):
    """Exception raised when a snapshot is not found in the sample or assay storage."""


class SnapshotError(ValueError):
    """Exception raised from snapshot data related errors."""


class DatasetNotFound(ValueError):
    """Exception raised when a dataset is not found."""


class InvalidSeparationMode(ValueError):
    """Exception raised when an invalid separation mode is passed."""


class SampleProcessorError(ValueError):
    """Exception raised when a sample processor fails to process a sample."""


class RepeatedSampleOrder(ValueError):
    """Exception raised when multiple samples have the same order."""


class EmptyDataMatrix(ValueError):
    """Exception raised when an empty data matrix is created or a transformation results in an empty matrix."""


class MatrixQueryError(ValueError):
    """Exception raised when an invalid sample query is created."""


class SampleMetadataNotFound(ValueError):
    """Exception raises when a sample does not contain required metadata."""
