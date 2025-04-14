"""Low level utilities to work with raw data."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Sequence

import numpy as np
import pydantic
from scipy.interpolate import interp1d

from ..core.enums import MSDataMode
from ..core.models import Chromatogram, MSSpectrum, MZTrace, Sample
from ..core.utils.numpy import FloatArray1D, IntArray1D, find_closest
from ..io.msdata import MSData


class BaseRawMsFunctionParameter(pydantic.BaseModel):
    """Base class that other parameter models inherit from."""

    ms_level: pydantic.PositiveInt | None = None
    """the data MS level"""

    start_time: pydantic.NonNegativeFloat | None = None
    """Use spectra with acquisition times greater than this value"""

    end_time: pydantic.NonNegativeFloat | None = None
    """Use spectra with acquisition times greater than this value. If ``None``, it doesn't
    filter scans by time.
    """


class MakeTICParameters(BaseRawMsFunctionParameter):
    """Store make_tic parameters."""

    kind: str = "tic"
    """`tic` computes the total ion chromatogram. `bpi` computes the base peak chromatogram"""

    @pydantic.field_validator("kind", mode="before")
    @classmethod
    def _check_kind(cls, value: str) -> str:
        msg = f"`kind` must be set either to `tic` or `bpi`. Got {value}."
        assert value in {"tic", "bpi"}, msg
        return value


class MakeChromatogramParameters(BaseRawMsFunctionParameter):
    """Store make_chromatogram parameters."""

    mz: Sequence[float]
    """a sorted sequence of m/z values"""

    window: pydantic.PositiveFloat = 0.05
    """m/z tolerance to build EICs."""

    accumulator: str = "sum"
    """How multiple values in a m/z windows are accumulated. ``"sum"`` computes
    the total intensity inside the window. ``"mean"`` divides the total intensity
    using the number of points inside the window.
    """

    fill_missing: bool = True
    """If ``True``, sets the intensity to zero if no signal was found in a m/z window.
    If ``False``, missing values are set to ``nan``.
    """


class MakeRoiParameters(BaseRawMsFunctionParameter):
    """Store make_roi parameters."""

    tolerance: float = 0.01
    """connect m/z values across scans if they are closer than this value"""

    max_missing: pydantic.NonNegativeInt = 1
    """maximum number of consecutive missing values in a ROI."""

    min_length: pydantic.PositiveInt = 5
    """The minimum length of a ROI, defined as the number of non-NaN values in the ROI."""

    min_intensity: pydantic.NonNegativeFloat = 0.0
    """Discard ROIs if all of its elements have an intensity lower than this value."""

    multiple_match: str = "reduce"
    """How points are matched when there are multiple matches. If ``"closest"``
    is used, the closest peak is assigned as a match and the others are used to
    create new ROIs. If ``"reduce"`` is used, an m/z and intensity pair is
    computed for all matching points using the mean for m/z and the sum for the
    intensity.
    """

    targeted_mz: Sequence[float] | None = None
    """If provided, only ROI with these m/z values will be created."""

    pad: pydantic.NonNegativeInt = 0
    """The number of dummy values to pad the ROI with"""

    @pydantic.field_validator("multiple_match", mode="before")
    @classmethod
    def _check_multiple_match(cls, value: str) -> str:
        msg = f"`multiple_match` must be set either to `closest` or `reduce`. Got {value}."
        assert value in {"closest", "reduce"}, msg
        return value


class AccumulateSpectraParameters(pydantic.BaseModel):
    """Store accumulate_spectra parameters."""

    ms_level: pydantic.PositiveInt = 1
    """the data MS level"""

    start_time: pydantic.NonNegativeFloat
    """Accumulate spectra starting at this acquisition time"""

    end_time: pydantic.NonNegativeFloat
    """Accumulate spectra starting until this acquisition time"""

    subtract_left_time: float | None = None
    """Scans with acquisition times lower than this value are subtracted from the
    accumulated spectrum. If ``None``, no subtraction is done.

    """
    subtract_right_time: float | None = None
    """Scans with acquisition times greater than this value are subtracted from the
    accumulated spectrum. If ``None``, no subtraction is done.
    """


def make_tic(data: MSData, params: MakeTICParameters) -> Chromatogram:
    """Create a total ion chromatogram.

    :param data: The data file used to create the TIC
    :param params: make TIC parameters.

    .. seealso:: MakeTICParameters

    """
    rt = list()
    tic = list()

    reduce = np.sum if params.kind == "tic" else np.max

    with data.using_tmp_config(ms_level=params.ms_level, start_time=params.start_time, end_time=params.end_time):
        for sp in data:
            rt.append(sp.time)
            tic.append(reduce(sp.int) if sp.int.size else 0.0)
    return Chromatogram(time=np.array(rt), int=np.array(tic))


def make_chromatograms(data: MSData, params: MakeChromatogramParameters) -> list[Chromatogram]:
    """Compute multiple :term:`EIC` from raw data using a list of m/z values.

    :param ms_data: raw data
    :param params: function parameters

    .. seealso:: MakeChromatogramParameters

    """
    n_sp = data.get_n_spectra()

    mz_arr = np.array(sorted(params.mz))

    # create an array of m/z windows: [mz0 - w, mz0 + w, mz1 - w, mz1 + w, ..., mzn + w]
    # where w is the window parameter
    mz_intervals = np.vstack((mz_arr - params.window, mz_arr + params.window)).T.reshape(mz_arr.size * 2)

    eic = np.zeros((mz_arr.size, n_sp))
    if not params.fill_missing:
        eic[:] = np.nan

    rt = np.zeros(n_sp)
    valid_index = list()
    with data.using_tmp_config(ms_level=params.ms_level, start_time=params.start_time, end_time=params.end_time):
        for sp in data:
            valid_index.append(sp.index)
            rt[sp.index] = sp.time

            # prevents error when working with empty spectra
            if sp.mz.size == 0:
                continue

            # consecutive values in window index defines the location of an m/z window
            window_index = np.searchsorted(sp.mz, mz_intervals)
            window_width = window_index[1::2] - window_index[::2]
            window_has_mz = (window_width) > 0
            # elements added at the end of mz_sp raise IndexError
            window_index[window_index >= sp.mz.size] = sp.mz.size - 1
            # Fast sum of values in a window
            fill = 0.0 if params.fill_missing else np.nan
            tmp_eic = np.where(window_has_mz, np.add.reduceat(sp.int, window_index)[::2], fill)

            if params.accumulator == "mean":
                window_width[~window_has_mz] = 1
                tmp_eic = tmp_eic / window_width
            eic[:, sp.index] = tmp_eic

    valid_index = np.array(valid_index)
    rt = rt[valid_index]
    eic = eic[:, valid_index]

    chromatograms = list()
    for row in eic:
        chromatogram = Chromatogram(time=rt.copy(), int=row)
        chromatograms.append(chromatogram)
    return chromatograms


def make_roi(data: MSData, params: MakeRoiParameters) -> list[MZTrace]:
    """Build regions of interest (ROI) from raw data.

    ROI are created by connecting values across scans based on the closeness in
    m/z. See the :ref:`algorithms-roi-extraction` for a description of the
    algorithm used.

    :param data : raw data
    :param params: algorithm parameters

    .. seealso:: lcms.MZTrace
    .. seealso:: MakeRoiParameters

    """
    if data.sample.ms_data_mode is MSDataMode.PROFILE and data.centroider is None:
        msg = "`ms_data` needs to be in centroid mode or needs to have a centroider."
        raise ValueError(msg)

    targeted = params.targeted_mz is not None

    if params.targeted_mz is not None:
        roi_seeds = np.sort(list(params.targeted_mz))
    else:
        if params.min_intensity > 0.0:
            roi_seeds = _compute_roi_seeds(data, params)
        else:
            roi_seeds = None

    rt = np.zeros(data.get_n_spectra())
    roi_maker = _RoiMaker(data.get_sample(), params, roi_seeds=roi_seeds, targeted=targeted)

    scans = list()
    with data.using_tmp_config(ms_level=params.ms_level, start_time=params.start_time, end_time=params.end_time):
        for spectrum in data:
            rt[spectrum.index] = spectrum.time
            scans.append(spectrum.index)
            roi_maker.feed_spectrum(spectrum)
            roi_maker.clear_completed_roi()

    # add roi not completed during the last scan
    roi_maker.flag_as_completed()
    roi_maker.clear_completed_roi()
    scans = np.array(scans)
    roi_list = roi_maker.tmp_roi_to_roi(scans, rt, params.pad)

    return roi_list


def accumulate_spectra(ms_data: MSData, params: AccumulateSpectraParameters) -> MSSpectrum:
    """Accumulate consecutive spectra into a single spectrum.

    :param ms_data: raw data file
    :param params: algorithm parameters

    .. seealso:: AccumulateSpectraParameters

    """
    if params.subtract_left_time is None:
        params.subtract_left_time = params.start_time

    if params.subtract_right_time is None:
        params.subtract_right_time = params.end_time

    if ms_data.sample.ms_data_mode is MSDataMode.CENTROID:
        return _accumulate_spectra_centroid(ms_data, params)
    return _accumulate_spectra_profile(ms_data, params)


def _accumulate_spectra_centroid(ms_data: MSData, params: AccumulateSpectraParameters) -> MSSpectrum:
    # don't remove any m/z value when detecting rois
    max_missing = ms_data.get_n_spectra()

    roi_params = MakeRoiParameters(
        max_missing=max_missing,
        min_length=1,
        start_time=params.subtract_left_time,
        end_time=params.subtract_right_time,
        ms_level=params.ms_level,
    )

    roi = make_roi(ms_data, roi_params)

    mz = np.zeros(len(roi))
    spint = mz.copy()

    # set subtract values to negative
    for k, r in enumerate(roi):
        sign = -np.ones(r.time.size)
        start_index, end_index = np.searchsorted(r.time, [params.start_time, params.end_time])
        sign[start_index:end_index] = 1
        mz[k] = np.nanmean(r.mz)
        spint[k] = np.nansum(r.spint * sign)

    # remove negative values
    pos_values = spint > 0
    mz = mz[pos_values]
    spint = spint[pos_values]

    # sort values
    sorted_index = np.argsort(mz)
    mz = mz[sorted_index]
    spint = spint[sorted_index]

    sp = MSSpectrum(mz=mz, int=spint, ms_level=params.ms_level)
    return sp


def _accumulate_spectra_profile(ms_data: MSData, params: AccumulateSpectraParameters) -> MSSpectrum:
    # The spectra are accumulated in two steps:
    #
    #  1.  iterate through scans to build a grid of m/z values for the
    #      accumulated spectra.
    #  2.  A second iteration is done to interpolate the intensity in each
    #      scan to the m/z grid and generate the accumulated spectrum.
    #
    #  This process is done in two steps to avoid storing the intensity
    #  values from each scan until the grid is built.

    # first iteration. Builds a grid of m/z values for the accumulated
    # spectrum. The grid is extended using new m/z values that appear
    # in each new scan
    accumulated_mz = _make_mz_filter(ms_data, MakeRoiParameters(**params.model_dump()))

    assert params.subtract_left_time is not None

    accumulated_sp = np.zeros_like(accumulated_mz)

    with ms_data.using_tmp_config(
        ms_level=params.ms_level, start_time=params.subtract_left_time, end_time=params.subtract_right_time
    ):
        for sp in ms_data:
            interpolator = interp1d(sp.mz, sp.int, fill_value=0.0)
            if (sp.time < params.start_time) or (sp.time > params.end_time):
                sign = -1
            else:
                sign = 1
            accumulated_sp += interpolator(accumulated_mz) * sign

    # set negative values that may result from subtraction to zero
    is_positive_sp = accumulated_sp > 0
    accumulated_mz = accumulated_mz[is_positive_sp]
    accumulated_sp = accumulated_sp[is_positive_sp]

    return MSSpectrum(mz=accumulated_mz, int=accumulated_sp, centroid=False)


class _RoiMaker:
    """Create and extends ROIs using spectrum data.

    Auxiliary class to make_roi

    Attributes
    ----------
    params : MakeRoiParameters
    mz_filter : sorted array
        m/z values used as a first filter for the values of spectra provided.
        m/z values in spectra with distances larger than `tolerance` are
        ignored.
    targeted : bool
        If ``True``, the mean of each ROI is updated after a new element is
        appended. Else, the mean when the ROI was created is used.

    """

    def __init__(
        self, sample: Sample, params: MakeRoiParameters, roi_seeds: np.ndarray | None, targeted: bool = False
    ):
        self.tmp_roi_list = _TempRoiList(update_mean=not targeted)
        self.valid_roi: deque["_TempRoi"] = deque()
        self.mz_filter = roi_seeds
        self.params = params
        self.sample = sample
        self.targeted = targeted

        if roi_seeds is not None:
            self.tmp_roi_list.initialize(roi_seeds)

    def feed_spectrum(self, sp: MSSpectrum):
        """Use a spectrum to extend and create ROIs."""
        if self.mz_filter is not None:
            sp = _filter_invalid_mz(self.mz_filter, sp, self.params.tolerance)

        if self.tmp_roi_list.roi:
            match = _match_mz(self.tmp_roi_list.mz_mean, sp, self.params.tolerance, self.params.multiple_match)
            self.tmp_roi_list.extend(match)

            if not self.targeted:
                self.tmp_roi_list.insert(match.no_match)
        else:
            if not self.targeted:
                self.tmp_roi_list.insert(sp)

    def clear_completed_roi(self):
        """Flag ROI as completed.

        Completed valid ROIs are stored  in the `roi` attribute. Invalid ROIs are cleared.

        """
        finished_mask = self.tmp_roi_list.missing_count > self.params.max_missing
        finished_roi_index = np.where(finished_mask)[0]
        valid_mask = finished_mask & (self.tmp_roi_list.max_int >= self.params.min_intensity)
        valid_mask &= self.tmp_roi_list.length >= self.params.min_length

        valid_index = np.where(valid_mask)[0]

        for i in valid_index:
            r = self.tmp_roi_list.get_roi_copy(i)
            self.valid_roi.append(r)

        self.tmp_roi_list.clear(finished_roi_index)

    def flag_as_completed(self):
        """Mark all ROis as completed."""
        self.tmp_roi_list.missing_count[:] = self.params.max_missing + 1

    def tmp_roi_to_roi(self, valid_scan: np.ndarray, time: np.ndarray, pad: int) -> list[MZTrace]:
        """Convert completed valid _TempRoi objects into LCTraces.

        :param valid_scan: scan values used to build the ROIs.
        :param time: acquisition time of each scan.


        """
        scan_to_index = {x.item(): k for k, x in enumerate(valid_scan)}
        valid_roi = list()
        while self.valid_roi:
            tmp = self.valid_roi.popleft()

            roi = tmp.to_mz_trace(time, valid_scan, self.params.pad, scan_to_index, self.sample)

            valid_roi.append(roi)
        return valid_roi


class _TempRoiList:
    """Container object of Temporary ROI.

    Auxiliary class used in make_roi.

    :param roi : list of ROI, sorted by mean m/z value.
    :param mz_mean: tracks the mean m/z value of each ROI.
    :param mz_sum: sum of m/z values in each ROI, used to update the mean.
    :param max_int: maximum intensity stored in each ROI.
    :param missing_count: number of times that a ROI has not been extended after calling `extend`.
        Each time that a ROI is extended the count is reset to 0.
    :param length: number of elements in each ROI.
    :param update_mean: If set to ``True``, the mean of each ROI is updated after a new element is
        appended. Else, the mean when the ROI was created is used.

    """

    def __init__(self, update_mean: bool = True):
        self.roi: list["_TempRoi"] = list()
        self.mz_mean = np.array([])
        self.mz_sum = np.array([])
        self.max_int = np.array([])
        self.missing_count = np.array([], dtype=int)
        self.length = np.array([], dtype=int)
        self.update_mean = update_mean

    def insert(self, sp: MSSpectrum):
        """Create new ROI and insert them in the list while keeping the order."""
        index = np.searchsorted(self.mz_mean, sp.mz)
        # update roi tracking values
        self.mz_mean = np.insert(self.mz_mean, index, sp.mz)
        self.mz_sum = np.insert(self.mz_sum, index, sp.mz)
        self.max_int = np.insert(self.max_int, index, sp.int)
        self.missing_count = np.insert(self.missing_count, index, np.zeros_like(index))
        self.length = np.insert(self.length, index, np.ones_like(index))

        # insert new roi
        new_roi = _create_roi_list(sp)
        offset = 0
        for i, roi in zip(index, new_roi):
            self.roi.insert(i + offset, roi)
            offset += 1

    def get_roi_copy(self, index: int) -> _TempRoi:
        """Create a deep copy of a temp ROI."""
        roi_copy = deepcopy(self.roi[index])
        roi_copy.mz_mean = self.mz_mean[index].item()
        return roi_copy

    def extend(self, match: SpectrumMatch):
        """Extend existing ROI with matching values."""
        index = match.index
        mz = match.match.mz
        sp = match.match.int
        for i, m, s in zip(index, mz, sp):
            self.roi[i].append(m, s, match.match.index)
        self.length[index] += 1
        if self.update_mean:
            self.mz_sum[index] += mz
            self.mz_mean[index] = self.mz_sum[index] / self.length[index]
        self.max_int[index] = np.maximum(self.max_int[index], sp)
        self.missing_count += 1
        self.missing_count[index] = 0

    def clear(self, index: np.ndarray):
        """Empty the m/z, intensity and scan values stored in each ROI.

        The mean value of each ROI is kept.

        :param index: indices of ROI to clear.

        """
        for i in index:
            self.roi[i].clear()

        self.mz_sum[index] = 0
        self.max_int[index] = 0
        self.missing_count[index] = 0
        self.length[index] = 0

    def initialize(self, mz: np.ndarray):
        init_sp = MSSpectrum(mz=mz, int=mz, index=0)
        self.insert(init_sp)
        self.clear(np.arange(mz.size))


class _TempRoi:
    """Stores data from a ROI.

    Auxiliary class used in make_roi.
    """

    def __init__(self):
        """Create a new empty Temporary ROI."""
        self.mz = deque()
        self.spint = deque()
        self.scan = deque()
        self.mz_mean = 0.0

    def append(self, mz: float, spint: float, scan: int):
        """Append new m/z, intensity and scan values."""
        self.mz.append(mz)
        self.spint.append(spint)
        self.scan.append(scan)

    def clear(self):
        """Empty the m/z, intensity and scan values stored."""
        self.mz = deque()
        self.spint = deque()
        self.scan = deque()

    def pad(self, n: int, valid_scan: np.ndarray):
        """Pad the ROI m/z and intensity with NaN.

        Values are padded only if the scan number allows it, that means if there
        are `n` points to the left or the right of the minimum and maximum scan
        number of the ROI in `valid_scans`.
        """
        first_scan = self.scan[0]
        last_scan = self.scan[-1]
        start, end = np.searchsorted(valid_scan, [first_scan, last_scan + 1])
        left_pad_index = max(0, start - n)
        n_left = start - left_pad_index
        right_pad_index = min(valid_scan.size, end + n)
        n_right = right_pad_index - end

        # left pad
        self.mz.extendleft([np.nan] * n_left)
        self.spint.extendleft([np.nan] * n_left)
        self.scan.extendleft(valid_scan[left_pad_index:start][::-1])

        # right pad
        self.mz.extend([np.nan] * n_right)
        self.spint.extend([np.nan] * n_right)
        self.scan.extend(valid_scan[end:right_pad_index])

    def to_mz_trace(
        self,
        rt: np.ndarray,
        scans: np.ndarray,
        pad: int,
        scan_to_index: dict[int, int],
        sample: Sample,
    ) -> MZTrace:
        """Convert to a m/z trace."""
        start_idx, end_idx = np.searchsorted(scans, [self.scan[0], self.scan[-1] + 1])
        start_pad_idx = max(0, start_idx - pad)
        end_pad_idx = min(scans.size, end_idx + pad)

        trace_scans = scans[start_pad_idx:end_pad_idx].copy()
        trace_rt = rt[start_pad_idx:end_pad_idx].copy()

        start_idx -= start_pad_idx
        end_idx -= start_pad_idx
        end_pad_idx -= start_pad_idx
        start_pad_idx = 0

        found_index = list()
        missing_index = list()
        found_scans = set(self.scan)
        for idx, scan in enumerate(trace_scans[start_idx:end_idx], start=start_idx):
            if scan in found_scans:
                found_index.append(idx)
            else:
                missing_index.append(idx)

        trace_mz = np.ones(shape=trace_scans.size, dtype=float) * self.mz_mean
        trace_mz[found_index] = self.mz

        trace_sp = np.zeros(shape=trace_scans.size, dtype=float)
        trace_sp[found_index] = self.spint
        if trace_scans.size > len(self.mz):
            trace_sp[missing_index] = np.interp(missing_index, found_index, self.spint)

        if start_pad_idx < start_idx:
            slope = (trace_sp[start_idx + 1] - trace_sp[start_idx]) / (trace_rt[start_idx + 1] - trace_rt[start_idx])
            intercept = trace_sp[start_idx]
            for idx in range(start_pad_idx, start_idx):
                trace_sp[idx] = max(0.0, (idx - start_idx) * slope + intercept)

        if end_idx < end_pad_idx:
            slope = (trace_sp[end_idx - 1] - trace_sp[end_idx - 2]) / (trace_rt[end_idx - 2] - trace_rt[end_idx - 1])
            intercept = trace_sp[end_idx - 2]
            for idx in range(end_idx, end_pad_idx):
                trace_sp[idx] = max(0.0, (idx - end_idx - 1) * slope + intercept)
        return MZTrace(sample=sample, time=trace_rt, mz=trace_mz, scan=trace_scans, spint=trace_sp)

    def convert_to_roi(self, rt: np.ndarray, scans: np.ndarray, sample: Sample) -> MZTrace:
        """Convert a TemporaryRoi into a ROI object.

        :param rt: acquisition times of each scan
        :param scans: sorted scan numbers used to build the ROIs.

        """
        # new arrays that include missing values. These new arrays may include
        # scans from other ms levels that must be removed
        first_scan = self.scan[0]
        last_scan = self.scan[-1]
        size = last_scan + 1 - first_scan
        mz_tmp = np.ones(size) * np.nan
        spint_tmp = mz_tmp.copy()

        # copy values from the ROI to the new arrays
        scan_index = np.array(self.scan) - self.scan[0]
        mz_tmp[scan_index] = self.mz
        spint_tmp[scan_index] = self.spint

        # remove scan numbers from other ms levels (i.e. scan numbers that are
        # not in the scans array)
        start_ind, end_ind = np.searchsorted(scans, [first_scan, last_scan + 1])
        scan_tmp = scans[start_ind:end_ind].copy()
        valid_index = scan_tmp - first_scan
        mz_tmp = mz_tmp[valid_index]
        spint_tmp = spint_tmp[valid_index]
        rt_tmp = rt[scan_tmp].copy()

        return MZTrace(time=rt_tmp, spint=spint_tmp, mz=mz_tmp, scan=scan_tmp, sample=sample)


def _make_mz_filter(data: MSData, params: MakeRoiParameters) -> FloatArray1D:
    """Create a list of m/z values to initialize ROI.

    Auxiliary function to make_roi.
    """
    with data.using_tmp_config(ms_level=params.ms_level, start_time=params.start_time, end_time=params.end_time):
        mz_seed = [sp.mz[sp.int > params.min_intensity] for sp in data]
    return np.unique(np.hstack(mz_seed))


def _filter_invalid_mz(valid_mz: np.ndarray, sp: MSSpectrum, tolerance: float) -> MSSpectrum:
    """Find values in the spectrum that are within tolerance with the m/z values in the seed.

    Auxiliary function to _RoiProcessor.extend

    :param valid_mz: sorted array of m/z values.
    :param mz: m/z values to filter.
    :param sp: intensity values associated to each m/z.
    :param tolerance: m/z tolerance

    """
    closest_index = find_closest(valid_mz, sp.mz)
    dmz = np.abs(valid_mz[closest_index] - sp.mz)
    match_mask = dmz <= tolerance  # type: np.ndarray
    return MSSpectrum(mz=sp.mz[match_mask], int=sp.int[match_mask], index=sp.index)


def _create_roi_list(sp: MSSpectrum) -> list[_TempRoi]:
    roi_list = list()
    for m, s in zip(sp.mz, sp.int):
        roi = _TempRoi()
        roi.append(m, s, sp.index)
        roi_list.append(roi)
    return roi_list


class SpectrumMatch(pydantic.BaseModel):
    """Return type for match_mz."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    index: IntArray1D
    match: MSSpectrum
    no_match: MSSpectrum


def _match_mz(mz1: FloatArray1D, sp2: MSSpectrum, tolerance: float, mode: str) -> SpectrumMatch:
    """Find values in a spectrum that fall within the specified tolerance in a list of m/z values.

    :param mz1: _RoiProcessor mz_mean
    :param sp2: intensity values associated to mz2
    :param tolerance: tolerance used to match values
    :param mode: {"closest", "reduce"}
        Behavior when more than one peak in mz2 matches with a given peak in
        mz1. If mode is `closest`, then the closest peak is assigned as a
        match and the others are assigned to no match. If mode is `merge`, then
        a unique mz and int value is generated using the average of the mz and
        the sum of the intensities.
    :return: a tuple consisting of: matching indices, matching mz values in the spectrum, sp values
        in the spectrum, non matching m/z values and non matching sp values.

    """
    closest_index = find_closest(mz1, sp2.mz)
    dmz = np.abs(mz1[closest_index] - sp2.mz)
    match_mask = dmz <= tolerance  # type: np.ndarray
    no_match_mask = ~match_mask
    match_index = closest_index[match_mask]

    # check multiple_matches
    match_unique, match_first, match_count = np.unique(match_index, return_counts=True, return_index=True)

    # set match values
    match_index = match_unique
    sp_match = sp2.int[match_mask]
    mz_match = sp2.mz[match_mask]

    # solve multiple matches
    multiple_match_mask = match_count > 1
    multiple_match_first = match_first[multiple_match_mask]

    if match_first.size > 0:
        multiple_match_count = match_count[multiple_match_mask]
        if mode == "reduce":
            for first, count in zip(multiple_match_first, multiple_match_count):
                # mz1 and mz2 are both sorted, the multiple matches are
                # consecutive
                mz_multiple_match = mz_match[first : (first + count)]
                sp_multiple_match = sp_match[first : (first + count)]
                mz_match[first] = np.mean(mz_multiple_match)
                sp_match[first] = np.sum(sp_multiple_match)
        elif mode == "closest":
            match_index_mz = np.where(match_mask)[0][match_first]
            multiple_match_index_mz = match_index_mz[multiple_match_mask]
            iterator = zip(multiple_match_index_mz, multiple_match_first, multiple_match_count)
            for mz2_index, first, count in iterator:
                closest = np.argmin(dmz[mz2_index : mz2_index + count])
                # flag all multiple matches as no match except the closest one
                no_match_mask[mz2_index : mz2_index + count] = True
                no_match_mask[mz2_index + closest] = False
                mz_match[first] = mz_match[first + closest]
                sp_match[first] = sp_match[first + closest]
        else:
            msg = "mode must be `closest` or `merge`"
            raise ValueError(msg)

    match_sp = MSSpectrum(mz=mz_match[match_first], int=sp_match[match_first], index=sp2.index)
    no_match_sp = MSSpectrum(mz=sp2.mz[no_match_mask], int=sp2.int[no_match_mask], index=sp2.index)

    return SpectrumMatch(index=match_index, match=match_sp, no_match=no_match_sp)


def _compute_roi_seeds(ms_data: MSData, params: MakeRoiParameters) -> FloatArray1D:
    """Create ROI seed values used for untargeted ROI extraction in make_roi.

    Aux function for make_roi.

    The algorithm is as follows:

    -   Create an array with all seed candidates by concatenating all m/z values
        with intensities greater than `min_intensity` into an array.
    -   Sort the candidates array.
    -   Combine close candidates into clusters. The standard deviation of each
        cluster is lower than the `tolerance` parameter.
    -   Compute seed values as the mean from each cluster.

    """
    seed_candidates = _create_seed_candidates(ms_data, params)
    seeds = _combine_seed_candidates(seed_candidates, params)
    return seeds


def _combine_seed_candidates(mz: np.ndarray, params: MakeRoiParameters) -> FloatArray1D:
    N = mz.size

    if N == 0:
        return np.array([], dtype=mz.dtype)

    mz_cumsum = np.cumsum(mz)
    mz2_cumsum = np.cumsum(mz**2)
    start, end = 0, 0
    seed_list = list()

    var_thresh = params.tolerance**2
    previous_mean = 0.0
    previous_var = 0.0
    while end <= N:
        slice_mean, slice_var = _compute_slice_stats(mz_cumsum, mz2_cumsum, start, end)

        if slice_var > var_thresh:
            # create a cluster for the data using the previous valid value
            # arrays are copied to avoid references to the concatenated mz, sp
            # and scans arrays also use the original data dtype
            seed_list.append(previous_mean)
            start = end - 1
        else:
            end += 1
        previous_mean, previous_var = slice_mean, slice_var

    if 0.0 <= previous_var < var_thresh:
        seed_list.append(previous_mean)

    return np.array(seed_list)


def _create_seed_candidates(ms_data: MSData, params: MakeRoiParameters) -> FloatArray1D:
    """Concatenate and sort spectra using m/z values."""
    mz_list = list()
    with ms_data.using_tmp_config(start_time=params.start_time, end_time=params.end_time):
        for sp in ms_data:
            mz_list.append(sp.mz[sp.int >= params.min_intensity])

    # Round m/z values to the 5th decimal place to reduce the number of unique values
    mz = np.round(np.hstack(mz_list), 5)
    return np.unique(mz)


def _compute_slice_stats(mz_cumsum: np.ndarray, mz2_cumsum: np.ndarray, start: int, end: int) -> tuple[float, float]:
    """Compute the population mean and variance of the slice mz[start:end]."""
    # This method uses the formula used for estimation of the variance in the
    # welford algorithm.
    # The variance of the slice x[start:end] can be computed as follows
    # S: std; xm: mean; i= start; j = end;
    # N = j - i
    # xm = (\sum_{i}^{j - 1} x_{i}) / N
    # S^{2} = \sum_{i}^{j} (x_{i} - xm)^{2} / N
    # S^{2} = (\sum_{i}^{j} x_{i}^{2} - N * xm^{2}) / N
    # S^{2} is computed using cumulative sums
    size = end - start
    # computes the sum in the slice mz[start:end]

    if size == 0:
        return 0.0, 0.0

    if start == 0:
        mz_start = 0.0
        mz2_start = 0.0
    else:
        mz_start = mz_cumsum[start - 1]
        mz2_start = mz2_cumsum[start - 1]

    mz_sum_slice = mz_cumsum[end - 1] - mz_start
    mz2_sum_slice = mz2_cumsum[end - 1] - mz2_start

    mz_mean = mz_sum_slice / size
    mz_var = (mz2_sum_slice - mz_mean**2 * size) / size

    return mz_mean, mz_var
