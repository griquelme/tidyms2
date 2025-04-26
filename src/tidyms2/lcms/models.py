"""Data models for working with LC-MS data."""

from __future__ import annotations

import bisect
from functools import cached_property
from math import nan

import numpy as np
import pydantic
from scipy.integrate import cumulative_trapezoid, trapezoid
from typing_extensions import Self

from ..core.models import AnnotableFeature, IsotopicEnvelope, MZTrace


class Peak(AnnotableFeature[MZTrace]):
    """Representation of a chromatographic peak."""

    start: pydantic.NonNegativeInt
    """index in the m/z trace where the peak begins. Must be smaller than `apex`"""

    apex: pydantic.PositiveInt
    """index in the m/z trace where the apex of the peak is located. Must be smaller than `end`"""

    end: pydantic.PositiveInt
    """index in the m/z trace where the peak ends. Start and end used as slices defines the peak region."""

    @pydantic.model_validator(mode="after")
    def check_peak_definition(self) -> Self:
        """Perform peak sanity check.

        Check that start, apex and end indices are strictly sorted. Check that
        peak end index is lower than trace length.

        """
        msg = "start must be lower than loc and loc must be lower than end"
        assert self.start < self.apex < self.end, msg

        msg = "peak end must be lower than m/z trace length."
        assert self.end <= self.roi.scan.size, msg

        return self

    @pydantic.computed_field(repr=False)
    @cached_property
    def rt_start(self) -> float:
        """The peak start time."""
        return self.roi.time[self.start].item()

    @pydantic.computed_field(repr=False)
    @cached_property
    def rt_end(self) -> float:
        """The peak end time."""
        return self.roi.time[self.end - 1].item()

    @pydantic.computed_field
    @cached_property
    def rt(self) -> float:
        """Peak retention time, defined as the weighted average of the trace time in the peak region.

        The trace height is used as weights.
        """
        try:
            weights = self.roi.get_slice_height(self.start, self.end)
            return np.average(self.roi.time[self.start : self.end], weights=weights).item()
        except ZeroDivisionError:
            return nan

    @pydantic.computed_field(repr=False)
    @cached_property
    def height(self) -> float:
        """Peak height, defined as the difference between the peak intensity and the peak baseline at the apex."""
        height = self.roi.spint[self.apex].item()
        if self.roi.baseline is not None:
            height = height - self.roi.baseline[self.apex].item()
        return height

    @pydantic.computed_field(repr=False)
    @cached_property
    def area(self) -> float:
        """The peak area."""
        peak_extension = self.roi.get_slice_height(self.start, self.end)
        return trapezoid(peak_extension, self.roi.time[self.start : self.end]).item()

    @pydantic.computed_field(repr=False)
    @cached_property
    def width(self) -> float:
        """Compute the peak width.

        The peak width is defined as the region where the 95 % of the total peak
        area is distributed.

        Returns
        -------
        width : positive number.

        """
        height = self.roi.get_slice_height(self.start, self.end)

        area = cumulative_trapezoid(height, self.roi.time[self.start : self.end])
        if area[-1] > 0:
            relative_area = area / area[-1]
            percentile = [0.025, 0.975]
            start, end = self.start + np.searchsorted(relative_area, percentile)
            return float(self.roi.time[end] - self.roi.time[start])
        else:
            return nan

    @pydantic.computed_field(repr=False)
    @cached_property
    def extension(self) -> float:
        """The peak extension, defined as the length of the peak region."""
        return (self.roi.time[self.end - 1] - self.roi.time[self.start]).item()

    @pydantic.computed_field(repr=False)
    @cached_property
    def snr(self) -> float:
        """The peak signal-to-noise ratio.

        The SNR is defined as the quotient between the peak height and the noise level at the apex.
        If the noise level is not available, the SNR is set to ``nan``.
        """
        if self.roi.noise is None or np.allclose(self.roi.noise[self.apex], 0.0):
            return nan
        else:
            return (self.get("height") / self.roi.noise[self.apex]).item()

    @pydantic.computed_field
    @cached_property
    def mz(self) -> float:
        """The peak m/z, defined as the weighted average of the trace m/z in the peak region.

        The trace height is used as weights.
        """
        weights = self.roi.get_slice_height(self.start, self.end)
        try:
            mz_mean = np.average(self.roi.mz[self.start : self.end], weights=weights)
            return max(0.0, mz_mean.item())
        except ZeroDivisionError:
            return nan

    @pydantic.computed_field(repr=False)
    @cached_property
    def mz_std(self) -> float:
        """The peak m/z standard deviation."""
        return self.roi.mz[self.start : self.end].std().item()

    def compare(self, other: Self) -> float:
        """Compute the similarity between a pair of peaks.

        The similarity is defined as the cosine distance between the overlapping
        region of two peaks.

        """
        return _compare_features_lc(self, other)

    @staticmethod
    def compute_isotopic_envelope(*features) -> IsotopicEnvelope:
        """Compute the isotopic envelope (m/z and abundance) of a list of peaks.

        :param features: the peaks that conform the envelope

        """
        scan_start = 0
        scan_end = 10000000000  # dummy value
        for ft in features:
            scan_start = max(scan_start, ft.roi.scan[ft.start])
            scan_end = min(scan_end, ft.roi.scan[ft.end - 1])

        p = list()
        mz = list()
        if scan_start < scan_end:
            for ft in features:
                start = bisect.bisect(ft.roi.scan, scan_start)
                end = bisect.bisect(ft.roi.scan, scan_end)
                apex = (start + end) // 2  # dummy value
                tmp_peak = Peak(start=start, apex=apex, end=end, roi=ft.roi)
                p_area = trapezoid(tmp_peak.roi.spint[start:end], tmp_peak.roi.time[start:end])
                p.append(p_area)
                mz.append(ft.mz)
        total_area = sum(p)
        p = [x / total_area for x in p]
        return IsotopicEnvelope(mz=mz, p=p)


def _compare_features_lc(ft1: Peak, ft2: Peak) -> float:
    """Feature similarity function used in LC-MS data."""
    start1 = ft1.roi.scan[ft1.start]
    start2 = ft2.roi.scan[ft2.start]
    if start1 > start2:
        ft1, ft2 = ft2, ft1
    overlap_ratio = _overlap_ratio(ft1, ft2)
    min_overlap = 0.5
    if overlap_ratio > min_overlap:
        os1, oe1, os2, oe2 = _get_overlap_index(ft1, ft2)
        norm1 = np.linalg.norm(ft1.roi.spint[ft1.start : ft1.end])
        norm2 = np.linalg.norm(ft2.roi.spint[ft2.start : ft2.end])
        x1 = ft1.roi.spint[os1:oe1] / norm1
        x2 = ft2.roi.spint[os2:oe2] / norm2
        similarity = np.dot(x1, x2)
    else:
        similarity = 0.0
    return similarity


def _overlap_ratio(ft1: Peak, ft2: Peak) -> float:
    """Compute the overlap ratio, between a pair of peaks.

    The overlap ratio is the quotient between the overlap region and the
    maximum value of the extension.

    """
    start2 = ft2.roi.scan[ft2.start]
    end1 = ft1.roi.scan[ft1.end - 1]
    end2 = ft2.roi.scan[ft2.end - 1]
    # start1 <= start2. end1 > start2 is a sufficient condition for overlap
    if end1 > start2:
        # the overlap ratio is the quotient between the length overlapped region
        # and the extension of the shortest feature.
        if end1 <= end2:
            start2_index_in1 = bisect.bisect_left(ft1.roi.scan, start2)
            overlap_length = ft1.end - start2_index_in1
        else:
            overlap_length = ft2.end - ft2.start
        min_length = min(ft1.end - ft1.start, ft2.end - ft2.start)
        res = overlap_length / min_length
    else:
        res = 0.0
    return res


def _get_overlap_index(ft1: Peak, ft2: Peak) -> tuple[int, int, int, int]:
    """Compute the overlap indices for ft1 and ft2.

    `ft1` must start before `ft2`

    """
    end1 = ft1.roi.scan[ft1.end - 1]
    end2 = ft2.roi.scan[ft2.end - 1]
    start2 = ft2.roi.scan[ft2.start]
    if end1 >= end2:
        overlap_start1 = bisect.bisect_left(ft1.roi.scan, start2)
        overlap_end1 = bisect.bisect(ft1.roi.scan, end2)
        overlap_start2 = ft2.start
        overlap_end2 = ft2.end
    else:
        overlap_start1 = bisect.bisect_left(ft1.roi.scan, start2)
        overlap_end1 = ft1.end
        overlap_start2 = ft2.start
        overlap_end2 = bisect.bisect(ft2.roi.scan, end1)
    return overlap_start1, overlap_end1, overlap_start2, overlap_end2
