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


class SampleType(str, enum.Enum):
    """Sample types in an untargeted metabolomics assay."""

    SAMPLE = "sample"
    """A test subject sample."""

    TECHNICAL_QC = "QC"
    """A technical QC sample"""

    EXPERIMENTAL_QC = "eQC"
    """An experimental QC sample"""

    DILUTED_QC = "dQC"
    """A diluted QC sample"""

    BLANK = "blank"
    """A blank QC sample"""

    OTHER = "other"
    """Samples that do not belong to any of the other categories."""
