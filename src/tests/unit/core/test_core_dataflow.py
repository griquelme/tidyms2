import pytest

from tidyms2.core import dataflow
from tidyms2.core.exceptions import ProcessStatusError


class TestCheckProcessStatus:
    def test_base_status_is_compatible_with_itself(self):
        actual = dataflow.SampleProcessStatus()
        expected = dataflow.SampleProcessStatus()
        dataflow.check_process_status(actual, expected)

    def test_expected_roi_extracted_raises_error_if_false(self):
        actual = dataflow.SampleProcessStatus()
        expected = dataflow.SampleProcessStatus(roi_extracted=True)
        with pytest.raises(ProcessStatusError):
            dataflow.check_process_status(actual, expected)

    def test_expected_feature_extracted_raises_error_if_false(self):
        actual = dataflow.SampleProcessStatus()
        expected = dataflow.SampleProcessStatus(feature_extracted=True)
        with pytest.raises(ProcessStatusError):
            dataflow.check_process_status(actual, expected)

    def test_extra_field_not_in_reference_is_ignored(self):
        actual = dataflow.SampleProcessStatus(extra={"extra_check": True})
        expected = dataflow.SampleProcessStatus()
        dataflow.check_process_status(actual, expected)

    def test_extra_field_in_reference_ok(self):
        actual = dataflow.SampleProcessStatus(extra={"extra_expectation": True})
        expected = dataflow.SampleProcessStatus(extra={"extra_expectation": True})
        dataflow.check_process_status(actual, expected)

    def test_extra_field_in_reference_true_raise_error_if_missing(self):
        actual = dataflow.SampleProcessStatus()
        expected = dataflow.SampleProcessStatus(extra={"extra_expectation": True})
        with pytest.raises(ProcessStatusError):
            dataflow.check_process_status(actual, expected)

    def test_extra_field_in_reference_true_raise_error_if_false(self):
        actual = dataflow.SampleProcessStatus(extra={"extra_expectation": False})
        expected = dataflow.SampleProcessStatus(extra={"extra_expectation": True})
        with pytest.raises(ProcessStatusError):
            dataflow.check_process_status(actual, expected)


class TestUpdateProcessStatus:
    def test_update_two_equal_status_does_not_make_changes(self):
        actual = dataflow.SampleProcessStatus()
        expected = actual.model_copy(deep=True)
        reference = dataflow.SampleProcessStatus()
        dataflow.update_process_status(actual, reference)
        assert actual == expected

    def test_update_single_field(self):
        actual = dataflow.SampleProcessStatus()
        assert not actual.roi_extracted
        assert not actual.feature_extracted
        assert not actual.extra
        reference = dataflow.SampleProcessStatus(roi_extracted=True)
        dataflow.update_process_status(actual, reference)
        assert actual.roi_extracted
        assert not actual.feature_extracted
        assert not actual.extra

    def test_update_multiple_fields(self):
        actual = dataflow.SampleProcessStatus()
        assert not actual.roi_extracted
        assert not actual.feature_extracted
        assert not actual.extra
        reference = dataflow.SampleProcessStatus(roi_extracted=True, feature_extracted=True)
        dataflow.update_process_status(actual, reference)
        assert actual.roi_extracted
        assert actual.feature_extracted
        assert not actual.extra

    def test_update_extra(self):
        actual = dataflow.SampleProcessStatus()
        assert not actual.roi_extracted
        assert not actual.feature_extracted
        assert not actual.extra
        extra_field = "extra_check"
        reference = dataflow.SampleProcessStatus(extra={extra_field: True})
        dataflow.update_process_status(actual, reference)
        assert not actual.roi_extracted
        assert not actual.feature_extracted
        assert actual.get_extra(extra_field)
