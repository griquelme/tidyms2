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

    DART = "dart"
    """direct analysis in real time ion source"""

    HPLC = "hplc"
    """High Performance Liquid Chromatography separation"""

    UPLC = "uplc"
    """ultra performance liquid chromatography separation"""


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
    """Available feature scaling methods.

    All scaling methods support matrix with NaN values. NaN values will be obtained when scaling columns
    with constant values.

    """

    AUTOSCALING = "autoscaling"
    r"""Scale features to unitary population variance:

    .. math:

            X_{j}^{\textrm{(scaled)}} = \frac{X_{j} - \bar{X}_{j}}{S_{j}}

    where :math:`X_{j}` is the j-th column of the matrix, :math:`\bar{X}_{j}` is the column average and
    :math:`S_{j}` is the column population standard deviation.

    """

    PARETO = "pareto"
    r"""Mean center and scale features using the square root of the population standard deviation:

    .. math:

            X_{j}^{\textrm{(scaled)}} = \frac{X_{j} - \bar{X}_{j}}{\sqrt{S_{j}}}

    where :math:`X_{j}` is the j-th column of the matrix, :math:`\bar{X}_{j}` is the column average and
    :math:`S_{j}` is the column population standard deviation.

    """

    RESCALING = "rescaling"
    r"""Scale features to the range :math:`[0, 1]`:

    .. math:

            X_{j}^{\textrm{(scaled)}} = \frac{X_{j} - \min{X}_{j}}{\max{X_{j}} - \min{X_{j}}}

    where :math:`X_{j}` is the j-th column of the matrix, :math:`\max{X}_{j}` is the column maximum and
    :math:`\min{X}_{j}` is the column minimum.

    """


class AggregationMethod(str, enum.Enum):
    """Available sample aggregations methods."""

    LOD = "lod"
    """Compute the limit of detection, defined as the mean plus three times the sample standard
    deviation."""

    LOQ = "loq"
    """Compute the limit of quantification, defined as the mean plus ten times the sample standard
    deviation."""

    SUM = "sum"
    """sums the feature abundance from samples"""

    MEAN = "avg"
    """Computes the mean feature abundance from samples"""

    MEDIAN = "median"
    """Computes the median feature abundance across samples"""

    MIN = "min"
    """computes the minimum features value across samples"""

    MAX = "max"
    """computes the maximum features value across samples"""

    STD = "std"
    """compute the standard deviation of features across samples"""


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
    """Available correlation methods."""

    PEARSON = "pearson"
    """The Pearson's correlation coefficient."""

    SPEARMAN = "spearman"
    """The Spearman rank coefficient."""
