import pathlib

import numpy as np
import pytest

from tidyms2.algorithms import raw
from tidyms2.core.models import Chromatogram, MSSpectrum, Sample
from tidyms2.io import MSData


@pytest.fixture
def sample(lcms_sample_factory) -> Sample:
    return lcms_sample_factory("sample")


@pytest.fixture
def ms_data(sample: Sample):
    return MSData(sample)


@pytest.fixture
def mz_list(lcms_sample_factory) -> list[float]:
    sample_spec = lcms_sample_factory.create_simulated_sample_spec("default", 0)
    return list(sample_spec.make_grid())


class TestMakeRoi:
    @pytest.fixture
    def params(self) -> raw.MakeRoiParameters:
        return raw.MakeRoiParameters(tolerance=0.01, max_missing=0, min_length=5)

    def test_ok(self, ms_data: MSData, mz_list: list[float], params):
        roi_list = raw.make_roi(ms_data, params)
        assert len(roi_list) == len(mz_list)

    def test_targeted_mz_ok(self, ms_data: MSData, mz_list: list[float], params):
        # the first three m/z values generated by simulated experiment are used
        params.targeted_mz = mz_list
        params.targeted_mz.pop()
        roi_list = raw.make_roi(ms_data, params)
        assert len(roi_list) == len(params.targeted_mz)

    def test_make_roi_min_intensity(self, ms_data, params):
        # only two features have intensity larger than 2000
        expected_n_roi = 1
        params.min_intensity = 2000.0
        roi_list = raw.make_roi(ms_data, params)
        assert len(roi_list) == expected_n_roi

    def test_make_roi_multiple_match_closest(self, ms_data: MSData, params, mz_list):
        params.multiple_match = "closest"
        params.tolerance = 2.0  # Make match conflict between isotopologues
        roi_list = raw.make_roi(ms_data, params)
        assert len(roi_list) == len(mz_list)

    def test_make_roi_multiple_match_reduce(self, ms_data: MSData, params, mz_list):
        params.multiple_match = "reduce"
        params.targeted_mz = mz_list[::2]
        params.tolerance = 2.0  # Make match conflict between isotopologues
        roi_list = raw.make_roi(ms_data, params)
        assert len(roi_list) == len(mz_list) // 2


class TestAccumulateSpectra:
    def test_centroid(self, ms_data, mz_list):
        params = raw.AccumulateSpectraParameters(start_time=0.0, end_time=50.0)
        cum_sp = raw.accumulate_spectra(ms_data, params)
        assert cum_sp.mz.size == len(mz_list)


class TestMakeChromatograms:
    def test_make_chromatograms_using_sum_accumulator(self, ms_data: MSData, mz_list):
        # test that the chromatograms generated are valid

        # create chromatograms
        n_sp = ms_data.get_n_spectra()

        params = raw.MakeChromatogramParameters(mz=mz_list)

        chromatograms = raw.make_chromatograms(ms_data, params)
        assert len(chromatograms) == len(params.mz)
        for chrom in chromatograms:
            assert np.all(np.diff(chrom.time) > 0.0)
            assert chrom.time.size == n_sp

    def test_empty_mz_list_returns_empty_list(self, ms_data: MSData):
        params = raw.MakeChromatogramParameters(mz=list())
        actual = raw.make_chromatograms(ms_data, params)
        assert isinstance(actual, list)
        assert not actual

    def test_mz_list_without_signals_returns_all_zeroes(self, ms_data: MSData):
        params = raw.MakeChromatogramParameters(mz=[2500.0])

        chromatogram = raw.make_chromatograms(ms_data, params)[0]
        assert np.allclose(chromatogram.int, 0.0)

    def test_mz_list_without_signals_and_no_fill_missing_returns_with_all_nan(self, ms_data: MSData):
        params = raw.MakeChromatogramParameters(mz=[2500.0], fill_missing=False)
        chromatogram = raw.make_chromatograms(ms_data, params)[0]
        assert np.all(np.isnan(chromatogram.int))


class TestMakeTIC:
    def test_tic(self, ms_data: MSData):
        params = raw.MakeTICParameters(kind="tic")
        tic = raw.make_tic(ms_data, params)
        assert isinstance(tic, Chromatogram)
        assert np.all(tic.int >= 0.0)

    def test_bpi(self, ms_data: MSData):
        params = raw.MakeTICParameters(kind="bpi")
        bpi = raw.make_tic(ms_data, params)
        assert isinstance(bpi, Chromatogram)
        assert np.all(bpi.int >= 0.0)

    def test_bpi_intensity_is_lower_or_equal_than_tic_intensity(self, ms_data: MSData):
        bpi = raw.make_tic(ms_data, raw.MakeTICParameters(kind="bpi"))
        tic = raw.make_tic(ms_data, raw.MakeTICParameters(kind="tic"))
        assert np.all(tic.int >= bpi.int)


class TestTempRoi:
    def test_creation(self):
        temp_roi = raw._TempRoi()
        assert len(temp_roi.mz) == 0
        assert len(temp_roi.spint) == 0
        assert len(temp_roi.scan) == 0

    def test_append_ok(self):
        mz, sp, scan = 150.0, 200.0, 10
        temp_roi = raw._TempRoi()
        temp_roi.append(mz, sp, scan)

        assert temp_roi.mz[-1] == mz
        assert temp_roi.spint[-1] == sp
        assert temp_roi.scan[-1] == scan

    def test_clear_empties_temp_roi(self):
        mz, sp, scan = 150.0, 200.0, 10
        temp_roi = raw._TempRoi()
        temp_roi.append(mz, sp, scan)
        temp_roi.clear()

        assert len(temp_roi.mz) == 0
        assert len(temp_roi.spint) == 0
        assert len(temp_roi.scan) == 0

    def test_pad_appends_nan_on_both_ends(self):
        temp_roi = raw._TempRoi()
        temp_roi.append(1, 1, 2)
        temp_roi.append(1, 1, 3)
        scans = np.array([0, 1, 2, 3, 4, 5])
        temp_roi.pad(2, scans)
        assert list(temp_roi.mz) == [np.nan, np.nan, 1, 1, np.nan, np.nan]
        assert list(temp_roi.spint) == [np.nan, np.nan, 1, 1, np.nan, np.nan]
        assert list(temp_roi.scan) == list(scans)

    def test_pad_stops_on_left_scan_boundary(self):
        temp_roi = raw._TempRoi()
        temp_roi.append(1, 1, 1)
        temp_roi.append(1, 1, 2)
        scans = np.array([0, 1, 2, 3, 4, 5])
        temp_roi.pad(2, scans)
        # 0 is the minimum scan number, only one element should be padded to the left
        assert list(temp_roi.mz) == [np.nan, 1, 1, np.nan, np.nan]
        assert list(temp_roi.spint) == [np.nan, 1, 1, np.nan, np.nan]
        assert list(temp_roi.scan) == list([0, 1, 2, 3, 4])

    def test_pad_no_left_pad_if_roi_starts_on_boundary(self):
        temp_roi = raw._TempRoi()
        temp_roi.append(1, 1, 0)
        temp_roi.append(1, 1, 1)
        scans = np.arange(6)
        temp_roi.pad(2, scans)
        # 0 is the minimum scan number, no elements should be padded to the left
        assert list(temp_roi.mz) == [1, 1, np.nan, np.nan]
        assert list(temp_roi.spint) == [1, 1, np.nan, np.nan]
        assert list(temp_roi.scan) == list([0, 1, 2, 3])

    def test_pad_stops_on_right_scan_boundary(self):
        temp_roi = raw._TempRoi()
        temp_roi.append(1, 1, 3)
        temp_roi.append(1, 1, 4)
        scans = np.arange(6)
        temp_roi.pad(2, scans)
        # 5 is the maximum scan number, right pad should add only one element
        assert list(temp_roi.mz) == [np.nan, np.nan, 1, 1, np.nan]
        assert list(temp_roi.spint) == [np.nan, np.nan, 1, 1, np.nan]
        assert list(temp_roi.scan) == list([1, 2, 3, 4, 5])

    def test_pad_no_right_pad_if_roi_ends_on_boundary(self):
        temp_roi = raw._TempRoi()
        temp_roi.append(1, 1, 4)
        temp_roi.append(1, 1, 5)
        scans = np.arange(5)
        temp_roi.pad(2, scans)
        # 5 is the maximum scan number, no elements should be padded to the right
        assert list(temp_roi.mz) == [np.nan, np.nan, 1, 1]
        assert list(temp_roi.spint) == [np.nan, np.nan, 1, 1]
        assert list(temp_roi.scan) == list([2, 3, 4, 5])

    def test_convert_to_mz_trace(self):
        temp_roi = raw._TempRoi()
        temp_roi.append(1, 1, 2)
        temp_roi.append(1, 1, 4)
        temp_roi.append(1, 1, 5)
        scans = np.arange(6)
        time = np.arange(6)
        sample = Sample(id="test", path=pathlib.Path("."))
        mz_trace = temp_roi.convert_to_roi(time, scans, sample)

        assert np.array_equal(mz_trace.scan, [2, 3, 4, 5])
        assert np.allclose(mz_trace.time, [2, 3, 4, 5])


class TestRoiList:
    def test_creation(self):
        raw._TempRoiList()
        assert True

    def test_insert_empty_RoiList(self):
        roi_list = raw._TempRoiList()
        n = 10
        sp = MSSpectrum(mz=np.arange(n), int=np.arange(n), index=1)
        roi_list.insert(sp)

        # check tracking values
        assert np.array_equal(sp.mz, roi_list.mz_mean)
        assert np.array_equal(sp.mz, roi_list.mz_sum)
        assert np.array_equal(sp.int, roi_list.max_int)
        assert (roi_list.length == 1).all()
        assert (roi_list.missing_count == 0).all()

        for r, r_mz, r_sp in zip(roi_list.roi, sp.mz, sp.int):
            assert len(r.mz) == 1
            assert len(r.spint) == 1
            assert len(r.scan) == 1
            assert r.mz[-1] == r_mz
            assert r.spint[-1] == r_sp
            assert r.scan[-1] == sp.index

    def test_RoiList_two_consecutive_insert(self):
        roi_list = raw._TempRoiList()
        # first insert
        mz1 = np.array([0, 1, 2, 3, 4, 6, 7, 9])  # 5 and 8 are missing
        sp1 = MSSpectrum(mz=mz1, int=mz1.copy(), index=1)
        roi_list.insert(sp1)

        # second insert
        mz2 = np.array([5, 8])
        sp2 = MSSpectrum(mz=mz2, int=mz2.copy(), index=2)
        roi_list.insert(sp2)

        # check tracking values
        expected_mz = np.arange(10)
        expected_missing_count = np.zeros_like(roi_list.missing_count)
        expected_missing_count[mz1] = 1
        assert np.allclose(roi_list.mz_mean, expected_mz)
        assert np.allclose(roi_list.mz_sum, expected_mz)
        assert np.allclose(roi_list.max_int, expected_mz)
        assert (roi_list.length == 1).all()
        assert (roi_list.missing_count == 0).all()
        assert (np.diff(roi_list.mz_mean) >= 0).all()  # check is sorted

        # check roi values
        # values for mz and spint should be 0, 1, 2, 3, ...
        for expected, r in enumerate(roi_list.roi):
            assert r.mz[-1] == expected
            assert r.spint[-1] == expected
            if expected in [5, 8]:
                assert r.scan[-1] == sp2.index
            else:
                assert r.scan[-1] == sp1.index
            assert len(r.mz) == 1
            assert len(r.spint) == 1
            assert len(r.scan) == 1


def assert_match(actual: raw.SpectrumMatch, expected: raw.SpectrumMatch):
    """Test equality of match objects."""
    assert np.array_equal(expected.index, actual.index)
    assert np.array_equal(expected.match.mz, actual.match.mz)
    assert np.array_equal(expected.match.int, actual.match.int)
    assert np.array_equal(expected.no_match.mz, actual.no_match.mz)
    assert np.array_equal(expected.no_match.int, actual.no_match.int)


class TestMatchMz:
    tolerance = 2

    @pytest.fixture(scope="class")
    def mz1(self):
        return np.array([50, 75, 100, 125, 150], dtype=float)

    def test_no_multiple_matches(self, mz1):
        mz2 = np.array([40, 51, 78, 91, 99, 130, 150])
        sp2 = MSSpectrum(mz=mz2, int=np.ones_like(mz2))
        mode = "closest"
        actual = raw._match_mz(mz1, sp2, self.tolerance, mode)

        mz1_match_index = np.array([0, 2, 4], dtype=int)
        mz2_match_index = np.array([1, 4, 6], dtype=int)
        mz2_no_match_index = np.array([0, 2, 3, 5], dtype=int)
        expected = raw.SpectrumMatch(
            index=mz1_match_index,
            match=MSSpectrum(mz=sp2.mz[mz2_match_index], int=sp2.int[mz2_match_index]),
            no_match=MSSpectrum(mz=sp2.mz[mz2_no_match_index], int=sp2.int[mz2_no_match_index]),
        )
        assert_match(actual, expected)

    def test_no_matches(self, mz1):
        mz2 = np.array([40, 53, 78, 91, 97, 130, 154])
        sp2 = MSSpectrum(mz=mz2, int=np.ones_like(mz2))

        # expected values for match/no match indices
        mz1_match_index = np.array([], dtype=int)
        mz2_match_index = np.array([], dtype=int)
        mz2_no_match_index = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
        expected = raw.SpectrumMatch(
            index=mz1_match_index,
            match=MSSpectrum(mz=sp2.mz[mz2_match_index], int=sp2.int[mz2_match_index]),
            no_match=MSSpectrum(mz=sp2.mz[mz2_no_match_index], int=sp2.int[mz2_no_match_index]),
        )

        mode = "closest"
        actual = raw._match_mz(mz1, sp2, self.tolerance, mode)

        assert_match(actual, expected)

    def test_all_match(self, mz1):
        mz2 = np.array([51, 77, 99, 126, 150])
        sp2 = MSSpectrum(mz=mz2, int=np.ones_like(mz2))

        mz1_match_index = np.array([0, 1, 2, 3, 4], dtype=int)
        mz2_match_index = np.array([0, 1, 2, 3, 4], dtype=int)
        mz2_no_match_index = np.array([], dtype=int)
        expected = raw.SpectrumMatch(
            index=mz1_match_index,
            match=MSSpectrum(mz=sp2.mz[mz2_match_index], int=sp2.int[mz2_match_index]),
            no_match=MSSpectrum(mz=sp2.mz[mz2_no_match_index], int=sp2.int[mz2_no_match_index]),
        )

        mode = "closest"
        actual = raw._match_mz(mz1, sp2, self.tolerance, mode)

        assert_match(actual, expected)

    def test_match_mz_multiple_matches_mode_closest(self, mz1):
        mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151])
        sp2 = MSSpectrum(mz=mz2, int=np.ones_like(mz2))
        # expected values for match/no match indices
        # in `closest` mode, argmin is used to select the closest value. If more
        # than one value has the same difference, the first one in the array is
        # going to be selected.
        mz1_match_index = np.array([0, 2, 3, 4], dtype=int)
        mz2_match_index = np.array([0, 4, 6, 7], dtype=int)
        mz2_no_match_index = np.array([1, 2, 3, 5, 8], dtype=int)
        expected = raw.SpectrumMatch(
            index=mz1_match_index,
            match=MSSpectrum(mz=sp2.mz[mz2_match_index], int=sp2.int[mz2_match_index]),
            no_match=MSSpectrum(mz=sp2.mz[mz2_no_match_index], int=sp2.int[mz2_no_match_index]),
        )

        mode = "closest"
        actual = raw._match_mz(mz1, sp2, self.tolerance, mode)

        assert_match(actual, expected)

    def test_match_mz_multiple_matches_mode_reduce(self, mz1):
        mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151], dtype=float)
        sp2 = MSSpectrum(mz=mz2, int=np.ones_like(mz2))
        # expected values for match/no match indices
        # in `closest` mode, argmin is used to select the closest value. If more
        # than one value has the same difference, the first one in the array is
        # going to be selected.
        mz1_match_index = np.array([0, 2, 3, 4], dtype=int)
        mz2_no_match_index = np.array([2], dtype=int)
        expected_mz2_match = np.array([50.0, 100.0, 126.0, 150.5])
        expected_sp2_match = np.array([2, 3, 1, 2])
        expected = raw.SpectrumMatch(
            index=mz1_match_index,
            match=MSSpectrum(mz=expected_mz2_match, int=expected_sp2_match),
            no_match=MSSpectrum(mz=sp2.mz[mz2_no_match_index], int=sp2.int[mz2_no_match_index]),
        )

        mode = "reduce"
        actual = raw._match_mz(mz1, sp2, self.tolerance, mode)

        assert_match(actual, expected)

    def test_match_mz_invalid_mode(self, mz1):
        mz2 = np.array([49, 51, 78, 99, 100, 101, 126, 150, 151])
        sp2 = MSSpectrum(mz=mz2, int=np.ones_like(mz2))
        # expected values for match/no match indices
        # in closest mode, argmin is used to select the closest value. If more
        # than one value has the same difference, the first one in the array is
        # going to be selected.
        mode = "invalid-mode"
        with pytest.raises(ValueError):
            raw._match_mz(mz1, sp2, self.tolerance, mode)


class TestComputeSliceVariance:
    size = 100

    @pytest.fixture
    def x(self):
        return np.random.normal(size=self.size)

    @pytest.mark.parametrize("index", [0, 50, size])
    def test_compute_slice_variance_on_empty_slice_returns_zero(self, x, index):
        start = index
        end = index

        expected_mean = 0.0
        expected_var = 0.0

        x_cumsum = np.cumsum(x)
        x2_cumsum = np.cumsum(x**2)
        actual_mean, actual_var = raw._compute_slice_stats(x_cumsum, x2_cumsum, start, end)

        assert np.isclose(expected_mean, actual_mean)
        assert np.isclose(expected_var, actual_var)

    def test_compute_slice_variance_inside_array(self, x):
        start = 10
        end = 30

        expected_mean = x[start:end].mean()
        expected_var = x[start:end].var()

        x_cumsum = np.cumsum(x)
        x2_cumsum = np.cumsum(x**2)
        actual_mean, actual_var = raw._compute_slice_stats(x_cumsum, x2_cumsum, start, end)

        assert np.isclose(expected_mean, actual_mean)
        assert np.isclose(expected_var, actual_var)

    def test_compute_slice_variance_start_array(self, x):
        start = 0
        end = 30

        expected_mean = x[start:end].mean()
        expected_var = x[start:end].var()

        x_cumsum = np.cumsum(x)
        x2_cumsum = np.cumsum(x**2)
        actual_mean, actual_var = raw._compute_slice_stats(x_cumsum, x2_cumsum, start, end)

        assert np.isclose(expected_mean, actual_mean)
        assert np.isclose(expected_var, actual_var)

    def test_compute_slice_variance_end_array(self, x):
        start = 0
        end = self.size

        expected_mean = x[start:end].mean()
        expected_var = x[start:end].var()

        x_cumsum = np.cumsum(x)
        x2_cumsum = np.cumsum(x**2)
        actual_mean, actual_var = raw._compute_slice_stats(x_cumsum, x2_cumsum, start, end)

        assert np.isclose(expected_mean, actual_mean)
        assert np.isclose(expected_var, actual_var)


class TestCombineSeedCandidates:
    def test_empty_data_ok(self):
        mz = np.array([])
        params = raw.MakeRoiParameters(tolerance=0.005)
        candidates = raw._combine_seed_candidates(mz, params)
        assert len(candidates) == 0

    def test_length_one_ok(self):
        mz = np.array([0.001])
        params = raw.MakeRoiParameters(tolerance=0.005)
        candidates = raw._combine_seed_candidates(mz, params)
        assert len(candidates) == 1

    def test_combine_ok_case1(self):
        mz = np.array([0.001, 0.002, 0.003, 0.5, 0.8, 0.801, 0.802])
        params = raw.MakeRoiParameters(tolerance=0.005)
        expected = np.array([0.002, 0.5, 0.801])
        actual = raw._combine_seed_candidates(mz, params)

        assert np.allclose(actual, expected)

    def test_combine_ok_case2(self):
        mz = np.array([0.001, 0.002, 0.003, 0.5, 0.8, 0.801, 0.802, 0.9])
        params = raw.MakeRoiParameters(tolerance=0.005)
        expected = np.array([0.002, 0.5, 0.801, 0.9])
        actual = raw._combine_seed_candidates(mz, params)

        assert np.allclose(actual, expected)

    def test_combine_large_array_ok(self):
        params = raw.MakeRoiParameters(tolerance=0.005)
        arr_list = list()
        size = 1000
        expected = np.zeros(size)
        for k in range(size):
            cluster = np.random.normal(size=10, loc=k, scale=params.tolerance / 2)
            expected[k] = np.mean(cluster)
            arr_list.append(cluster)
        mz = np.sort(np.concatenate(arr_list))
        actual = raw._combine_seed_candidates(mz, params)
        assert np.allclose(actual, expected)
