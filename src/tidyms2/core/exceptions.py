"""TidyMS core exceptions."""


class FeatureGroupNotFound(ValueError):
    """Exception raised when a feature group is not found in an assay storage."""


class FeatureNotFound(ValueError):
    """Exception raised when a Feature is not found in a sample storage or assay storage."""


class InvalidFeatureDescriptor(ValueError):
    """Exception raised when a non existing feature descriptor name is requested."""


class PipelineConfigurationError(ValueError):
    """Exception raised when an invalid configuration is set in a pipeline."""


class ReaderNotFound(ValueError):
    """Exception raised when a reader is not found for a specific format."""


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
