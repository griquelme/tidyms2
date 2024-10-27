"""TidyMS constants."""

import enum


class OperatorType(str, enum.Enum):
    """Available operators types."""

    SAMPLE = "sample"
    """Apply operations on sample storage."""

    ASSAY = "assay"
    """Apply operations on assay storage."""

    MATRIX = "matrix"
    """Apply operations on data matrix."""


class DataType(str, enum.Enum):
    """Data types that processors operates on."""

    SAMPLE = "sample"
    ASSAY = "assay"
    MATRIX = "matrix"


class SeparationMode(str, enum.Enum):
    """Analytical method separation platform."""

    DART = "DART"
    HPLC = "HPLC"
    UPLC = "UPLC"


class MSInstrument(enum.Enum):
    """Available MS instrument types."""

    QTOF = "qtof"
    ORBITRAP = "orbitrap"


class Polarity(str, enum.Enum):
    """Scan polarity."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class MSDataMode(str, enum.Enum):
    """Raw data mode."""

    PROFILE = "profile"
    CENTROID = "centroid"


class IDFeatureDescriptors(str, enum.Enum):
    """ID related Feature descriptors added to a feature table."""

    ID = "id"
    ROI_ID = "roi_id"
    SAMPLE_ID = "sample_id"
    FEATURE_GROUP = "group"
