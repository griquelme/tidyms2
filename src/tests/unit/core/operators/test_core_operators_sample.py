import pydantic
import pytest

from tidyms2.core.dataflow import SampleProcessStatus
from tidyms2.core.exceptions import PipelineConfigurationError, ProcessStatusError, RepeatedIdError
from tidyms2.core.operators.pipeline import Pipeline

from ... import helpers


class TestRoiExtractor:
    @pytest.fixture
    def op(self):
        return helpers.DummyRoiExtractor()

    def test_set_invalid_parameter_raise_ValidationError(self, op):
        with pytest.raises(pydantic.ValidationError):
            op.param2 = 10  # type: ignore

    def test_update_process_status(self, op: helpers.DummyRoiExtractor):
        status_in = SampleProcessStatus()
        assert not status_in.roi_extracted
        op.update_status(status_in)
        assert status_in.roi_extracted

    def test_apply(self, sample_storage, op):
        assert not sample_storage.list_rois()
        op.apply(sample_storage)

        assert len(sample_storage.list_rois()) == op.n_roi
        assert sample_storage.get_status().roi_extracted


class TestRoiTransformer:
    @pytest.fixture
    def op(self):
        return helpers.DummyRoiTransformer()

    def test_check_status_invalid_status_raises_error(self, op: helpers.DummyRoiTransformer):
        status = SampleProcessStatus()
        with pytest.raises(ProcessStatusError):
            op.check_status(status)

    def test_check_status_ok(self, op: helpers.DummyRoiTransformer):
        status = SampleProcessStatus(roi_extracted=True)
        op.check_status(status)

    def test_apply(self, sample_storage_with_rois, op):
        op.apply(sample_storage_with_rois)

        assert all(x.data == op.max_length for x in sample_storage_with_rois.list_rois())
        assert sample_storage_with_rois.get_status().roi_extracted


class TestFeatureExtractor:
    @pytest.fixture
    def op(self):
        return helpers.DummyFeatureExtractor()

    def test_check_status_invalid_status_raises_error(self, op: helpers.DummyFeatureExtractor):
        status = SampleProcessStatus()
        with pytest.raises(ProcessStatusError):
            op.check_status(status)

    def test_check_status_ok(self, op: helpers.DummyFeatureExtractor):
        status = SampleProcessStatus(roi_extracted=True)
        op.check_status(status)

    def test_apply(self, sample_storage_with_rois, op):
        op.apply(sample_storage_with_rois)

        rois = sample_storage_with_rois.list_rois()
        assert rois
        for roi in rois:
            features = sample_storage_with_rois.list_features(roi_id=roi.id)
            assert len(features) == op.n_features

        assert sample_storage_with_rois.get_status().roi_extracted
        assert sample_storage_with_rois.get_status().feature_extracted

    def test_apply_with_filter_remove_all_features(self, sample_storage_with_rois, op):
        op.bounds = {"height": (1000000.0, 2000000.0)}
        op.apply(sample_storage_with_rois)

        assert sample_storage_with_rois.list_rois()
        assert not sample_storage_with_rois.list_features()

        assert sample_storage_with_rois.get_status().roi_extracted
        assert sample_storage_with_rois.get_status().feature_extracted

    def test_apply_with_filter_fill_lower_bound_with_inf(self, sample_storage_with_rois, op):
        op.bounds = {"height": (None, 2000000.0)}
        op.apply(sample_storage_with_rois)

        assert sample_storage_with_rois.list_rois()
        assert sample_storage_with_rois.list_features()

        assert sample_storage_with_rois.get_status().roi_extracted
        assert sample_storage_with_rois.get_status().feature_extracted

    def test_apply_with_filter_fill_upper_bound_with_inf(self, sample_storage_with_rois, op):
        op.bounds = {"height": (0.0, None)}
        op.apply(sample_storage_with_rois)

        assert sample_storage_with_rois.list_rois()
        assert sample_storage_with_rois.list_features()

        assert sample_storage_with_rois.get_status().roi_extracted
        assert sample_storage_with_rois.get_status().feature_extracted


class TestFeatureTransformer:
    @pytest.fixture
    def op(self):
        return helpers.DummyFeatureTransformer()

    def test_check_status_invalid_status_raises_error(self, op: helpers.DummyFeatureTransformer):
        status = SampleProcessStatus(roi_extracted=True)
        with pytest.raises(ProcessStatusError):
            op.check_status(status)

    def test_check_status_ok(self, op: helpers.DummyRoiTransformer):
        status = SampleProcessStatus(roi_extracted=True, feature_extracted=True)
        op.check_status(status)

    def test_apply(self, sample_storage_with_features, op):
        op.apply(sample_storage_with_features)

        assert all(x.data_mz == op.feature_value for x in sample_storage_with_features.list_features())
        assert sample_storage_with_features.get_status().roi_extracted


class TestPipeline:
    @pytest.fixture
    def pipe(self):
        return Pipeline(id="test-pipeline")

    def test_add_sample_operator_to_empty_pipeline(self, pipe):
        pipe.add_operator(helpers.DummyRoiExtractor())

    def test_add_assay_operator_to_empty_pipeline(self, pipe):
        pipe.add_operator(helpers.DummyAnnotationPatcher())

    def test_add_sample_operator_with_repeated_raises_error(self, pipe):
        op = helpers.DummyRoiExtractor(id="dummy-id")
        pipe.add_operator(op)
        with pytest.raises(RepeatedIdError):
            pipe.add_operator(op)

    def test_add_sample_operator_to_pipeline_with_assay_operator_raises_error(self, pipe):
        assay_op = helpers.DummyDescriptorPatcher(id="dummy-descriptor-patcher-id")
        pipe.add_operator(assay_op)
        with pytest.raises(PipelineConfigurationError):
            sample_op = helpers.DummyRoiExtractor(id="dummy-roi-extractor-id")
            pipe.add_operator(sample_op)

    def test_add_assay_operator_to_pipeline_with_sample_operator_raises_error(self, pipe):
        sample_op = helpers.DummyRoiExtractor(id="dummy-roi-extractor-id")
        pipe.add_operator(sample_op)
        with pytest.raises(PipelineConfigurationError):
            assay_op = helpers.DummyDescriptorPatcher(id="dummy-descriptor-patcher-id")
            pipe.add_operator(assay_op)

    def test_add_empty_pipeline_raises_error(self, pipe):
        sub_pipe = Pipeline(id="empty-pipe")
        with pytest.raises(PipelineConfigurationError):
            pipe.add_operator(sub_pipe)

    def test_add_pipeline_with_assay_operators_to_pipeline_with_sample_operator_raises_error(self, pipe):
        sample_op = helpers.DummyRoiExtractor(id="dummy-roi-extractor-id")
        pipe.add_operator(sample_op)

        assay_op = helpers.DummyDescriptorPatcher(id="dummy-descriptor-patcher-id")
        assay_pipe = Pipeline(id="assay-pipe")
        assay_pipe.add_operator(assay_op)
        with pytest.raises(PipelineConfigurationError):
            pipe.add_operator(assay_pipe)

    def test_validate_dataflow_on_sample_ok_if_first_operator_is_roi_extractor(self, pipe):
        pipe.add_operator(helpers.DummyRoiExtractor())
        pipe.validate_dataflow()

    def test_validate_dataflow_on_sample_pipeline_raises_error_if_first_operator_is_not_roi_extractor(self, pipe):
        with pytest.raises(PipelineConfigurationError):
            pipe.add_operator(helpers.DummyRoiTransformer())
            pipe.validate_dataflow()

    def test_validate_dataflow_ok_if_multiple_operators_are_in_valid_order(self, pipe):
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        pipe.add_operator(helpers.DummyRoiTransformer(id="op2"))
        pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))
        pipe.validate_dataflow()

    def test_validate_dataflow_raise_error_if_multiple_operators_are_not_in_valid_order(self, pipe):
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        pipe.add_operator(helpers.DummyRoiTransformer(id="op2"))
        pipe.add_operator(helpers.DummyFeatureTransformer(id="op3"))
        pipe.add_operator(helpers.DummyFeatureExtractor(id="op4"))
        with pytest.raises(PipelineConfigurationError):
            pipe.validate_dataflow()

    def test_validate_dataflow_with_nested_pipeline_as_first_element(self, pipe):
        sub_pipe = Pipeline("sub-pipeline")
        sub_pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        sub_pipe.add_operator(helpers.DummyRoiTransformer(id="op2"))
        sub_pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        pipe.add_operator(sub_pipe)
        pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))
        pipe.validate_dataflow()

    def test_validate_dataflow_nested_pipeline(self, pipe):
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        sub_pipe = Pipeline("sub-pipeline")
        sub_pipe.add_operator(helpers.DummyRoiTransformer(id="op2"))
        sub_pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        sub_pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))
        pipe.add_operator(sub_pipe)
        pipe.validate_dataflow()

    def test_pipe_serialize_deserialize_flat_pipeline(self, pipe):
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        pipe.add_operator(helpers.DummyRoiTransformer(id="op2"))
        pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))

        pipe_serialized_deserialized = Pipeline.deserialize(pipe.serialize())

        assert pipe == pipe_serialized_deserialized

    def test_pipe_serialize_deserialize_nested_pipeline(self, pipe):
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        sub_pipe = Pipeline(id="sub-pipe")
        sub_pipe.add_operator(helpers.DummyRoiTransformer(id="op2"))
        sub_pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        sub_pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))
        pipe.add_operator(sub_pipe)

        pipe_serialized_deserialized = Pipeline.deserialize(pipe.serialize())

        assert pipe == pipe_serialized_deserialized

    def test_deserialize_dict_without_id_raises_error(self):
        d = dict()
        with pytest.raises(ValueError):
            Pipeline.deserialize(d)

    def test_deserialize_dict_without_operators_raises_error(self):
        d = {"id": "pipeline-id"}
        with pytest.raises(ValueError):
            Pipeline.deserialize(d)

    def test_deserialize_dict_with_invalid_operator_list_raises_error(self):
        d = {"id": "pipeline-id", "operators": [1]}
        with pytest.raises(ValueError):
            Pipeline.deserialize(d)

    def test_apply_flat_pipeline(self, pipe, sample_storage):
        assert not sample_storage.get_n_rois()
        assert not sample_storage.get_n_features()
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        pipe.add_operator(helpers.DummyRoiTransformer(id="op2", max_length=10))
        pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))
        pipe.apply(sample_storage)
        assert sample_storage.get_n_rois()
        assert sample_storage.get_n_features()

    def test_list_operators(self, pipe):
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        pipe.add_operator(helpers.DummyRoiTransformer(id="op2"))
        pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))

        assert pipe.list_operator_ids() == ["op1", "op2", "op3", "op4"]

    def test_fetch_operator(self, pipe):
        query_op = helpers.DummyRoiTransformer(id="op2")
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        pipe.add_operator(query_op)
        pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))

        fetched_op = pipe.get_operator(query_op.id)
        assert fetched_op == query_op

    def test_fetch_operator_not_found_raises_error(self, pipe):
        query_op = helpers.DummyRoiTransformer(id="op2")
        pipe.add_operator(helpers.DummyRoiExtractor(id="op1"))
        pipe.add_operator(query_op)
        pipe.add_operator(helpers.DummyFeatureExtractor(id="op3"))
        pipe.add_operator(helpers.DummyFeatureTransformer(id="op4"))

        with pytest.raises(ValueError):
            pipe.get_operator("invalid_id")
