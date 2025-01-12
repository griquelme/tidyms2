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
