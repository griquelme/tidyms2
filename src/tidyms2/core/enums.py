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

    TECHNICAL_QC = "qc"
    """A technical QC sample"""

    EXPERIMENTAL_QC = "eqc"
    """An experimental QC sample"""

    DILUTED_QC = "dqc"
    """A diluted QC sample"""

    BLANK = "blank"
    """A blank QC sample"""

    OTHER = "other"
    """Samples that do not belong to any of the other categories."""


class ScalingMethod(str, enum.Enum):
    """Available feature scaling methods."""

    AUTOSCALING = "autoscaling"
    """Scale features to unitary sample variance"""

    PARETO = "pareto"
    """Scale features using the square root of the sample standard deviation"""

    RESCALING = "rescaling"
    """Scale features to the range :math:`[0, 1]`"""


class NormalizationMethod(str, enum.Enum):
    """Available sample normalization methods."""

    SUM = "sum"
    """Normalize samples using sum of all features."""

    MAX = "max"
    """Normalize samples using the maximum value of all features"""

    EUCLIDEAN = "euclidean"
    """Normalize samples using the row 2-norm."""

    FEATURE = "feature"
    """Normalize samples using the value of a feature."""


class CorrelationMethod(str, enum.Enum):
    """Avilable correlation methods."""

    PEARSON = "pearson"
    """The Pearson's correlation coefficient."""

    SPEARMAN = "spearman"
    """The Spearman rank coefficient."""
