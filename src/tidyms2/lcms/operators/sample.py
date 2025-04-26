"""Tools to process LC-MS-based datasets."""

from __future__ import annotations

from typing import Self, assert_never

import pydantic
from numpy import minimum, zeros_like

from ...algorithms.raw import MakeRoiParameters, make_roi
from ...algorithms.signal import detect_peaks, estimate_baseline, estimate_noise, smooth
from ...core.enums import MSInstrument, Polarity, SeparationMode
from ...core.exceptions import InvalidSeparationMode
from ...core.models import MZTrace, Sample
from ...core.operators.sample import FeatureExtractor, RoiExtractor, RoiTransformer
from ...io import MSData
from ..models import Peak


class LCTraceExtractor(RoiExtractor[MZTrace, Peak], MakeRoiParameters):
    """Extracts regions-of-interest (ROI) from raw data represented as m/z traces.

    Traces are created by connecting values across consecutive scans based on the
    closeness in m/z.

    Refer to the :ref:`processing-lcms-datasets` guide for examples on how to use this operator.
    See the :ref:`algorithms-roi-extraction` for a description of the algorithm used.

    .. seealso:: lcms.MZTrace : Representation of a ROI using m/z traces.
    .. seealso:: lcms.MakeRoiParameters : Parameters used by the ROI extraction algorithm

    """

    def extract_rois(self, sample: Sample) -> list[MZTrace]:
        """Apply ROI extraction to a sample with LC data."""
        ms_data = MSData(sample)
        params = MakeRoiParameters.model_validate(self)
        params.ms_level = sample.ms_level
        params.start_time = sample.start_time
        params.end_time = sample.end_time
        return make_roi(ms_data, params)

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        """Set the processor default parameters.

        :param instrument : the instrument type used in the experimental setup
        :param separation : the LC platform used in the experimental setup
        :param polarity : the MS polarity used in the experiment

        """
        match instrument:
            case MSInstrument.QTOF:
                tolerance = 0.01
            case MSInstrument.ORBITRAP:
                tolerance = 0.005
            case _ as never:
                assert_never(never)

        match separation:
            case SeparationMode.UPLC:
                min_length = 10
            case SeparationMode.HPLC:
                min_length = 20
            case SeparationMode.DART:
                raise InvalidSeparationMode(f"{separation} is not a valid LC mode.")
            case _ as never:
                assert_never(never)
        return cls(max_missing=1, pad=2, min_intensity=250.0, tolerance=tolerance, min_length=min_length)


class LCTraceBaselineEstimator(RoiTransformer[MZTrace, Peak]):
    """Estimate the noise level and baseline in an m/z trace.

    The default values for this filter usually produce good results in most LC
    traces. Do not modify these values unless you know what you are doing.
    See :ref:`here <algorithms-peak-extraction>` for a description of the noise estimation
    and baseline estimation algorithms.

    """

    robust: bool = True
    """If ``True``, use the median absolute deviation as an estimator of the
    noise standard deviation. If ``False``. use the standard deviation."""

    min_slice_size: pydantic.PositiveInt = 200
    """The minimum size of a signal slice for local noise estimation. If the signal
    size is smaller than this value, the noise is estimated using the whole array.
    """

    n_slices: pydantic.PositiveInt = 5
    """Number of slices to create. The size of each slice must be greater than
    `min_slice_size`.
    """

    smoothing_strength: pydantic.PositiveFloat | None = 1.0
    """If specified, apply a temporary gaussian smoothing to the trace intensity.
    This step usually improves baseline estimation.
    """

    min_proba: float = pydantic.Field(default=0.05, le=0.5, gt=0.0)
    """The minimum probability of a signal chunk to be considered as baseline."""

    def transform_roi(self, roi: MZTrace):
        """Add noise and baseline to an LC trace."""
        roi.noise = estimate_noise(
            roi.spint, min_chunk_size=self.min_slice_size, n_chunks=self.n_slices, robust=self.robust
        )
        if self.smoothing_strength is not None:
            spint = smooth(roi.spint, self.smoothing_strength)
        else:
            spint = roi.spint

        roi.baseline = minimum(estimate_baseline(spint, roi.noise, min_proba=self.min_proba), roi.spint)
        return roi

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        """Set the processor default parameters.

        :param instrument : the instrument type used in the experimental setup
        :param separation : the LC platform used in the experimental setup
        :param polarity : the MS polarity used in the experiment

        """
        return cls()


class LCTraceSmoother(RoiTransformer[MZTrace, Peak]):
    """Smooth LC traces intensity using a gaussian kernel."""

    strength: pydantic.PositiveFloat = 1.0
    """The smoothing strength, defined as the standard deviation of the gaussian kernel"""

    def transform_roi(self, roi: MZTrace):
        """Add noise and baseline to an LC trace."""
        roi.spint = smooth(roi.spint, self.strength)
        return roi

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity) -> Self:
        """Set the processor default parameters.

        :param instrument : the instrument type used in the experimental setup
        :param separation : the LC platform used in the experimental setup
        :param polarity : the MS polarity used in the experiment

        """
        return cls()


class LCPeakExtractor(FeatureExtractor[MZTrace, Peak]):
    """Extract peaks from LC m/z traces.

    A complete description of the algorithm used for peak extraction can be found
    :ref:`here <algorithms-peak-extraction>`.

    """

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity):
        """Set the processor default parameters.

        :param instrument : the instrument type used in the experimental setup
        :param separation : the LC platform used in the experimental setup
        :param polarity : the MS polarity used in the experiment

        """
        match separation:
            case SeparationMode.HPLC:
                bounds = {"width": (10, 90), "snr": (5, None)}
            case SeparationMode.UPLC:
                bounds = {"width": (4, 60), "snr": (5, None)}
            case SeparationMode.DART:
                raise NotImplementedError
            case _ as never:
                assert_never(never)
        return cls(bounds=bounds)

    def extract_features(self, roi: MZTrace):
        """Detect peaks in an LC trace."""
        if roi.noise is None:
            roi.noise = estimate_noise(roi.spint)

        if roi.baseline is None:
            roi.baseline = zeros_like(roi.spint)

        start, apex, end = detect_peaks(roi.spint, roi.noise, roi.baseline)
        return [Peak(start=s, apex=a, end=e, roi=roi) for s, a, e in zip(start, apex, end)]
