import pydantic
import pytest

from tidyms2.core.dataflow import SampleProcessStatus
from tidyms2.core.exceptions import PipelineConfigurationError, ProcessStatusError
from tidyms2.core.operators import Pipeline
from tidyms2.storage.memory import OnMemorySampleStorage

from .. import helpers
from ..helpers import ConcreteFeature, ConcreteRoi


@pytest.fixture
def storage(tmp_path) -> OnMemorySampleStorage[ConcreteRoi, ConcreteFeature]:
    sample = helpers.create_sample(tmp_path, 1)
    return OnMemorySampleStorage(sample, ConcreteRoi, ConcreteFeature)


@pytest.fixture
def storage_with_rois(storage):
    roi_extractor = helpers.DummyRoiExtractor()
    roi_extractor.apply(storage)
    return storage


@pytest.fixture
def storage_with_features(storage_with_rois):
    feature_extractor = helpers.DummyFeatureExtractor()
    feature_extractor.apply(storage_with_rois)
    return storage_with_rois


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

    def test_apply(self, storage, op):
        assert not storage.list_rois()
        op.apply(storage)

        assert len(storage.list_rois()) == op.n_roi
        assert storage.get_status().roi_extracted


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

    def test_apply(self, storage_with_rois, op):
        op.apply(storage_with_rois)

        assert all(x.data == op.max_length for x in storage_with_rois.list_rois())
        assert storage_with_rois.get_status().roi_extracted


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

    def test_apply(self, storage_with_rois, op):
        op.apply(storage_with_rois)

        rois = storage_with_rois.list_rois()
        assert rois
        for roi in rois:
            features = storage_with_rois.list_features(roi_id=roi.id)
            assert len(features) == op.n_features

        assert storage_with_rois.get_status().roi_extracted
        assert storage_with_rois.get_status().feature_extracted

    def test_apply_with_filter_remove_all_features(self, storage_with_rois, op):
        op.bounds = {"height": (1000000.0, 2000000.0)}
        op.apply(storage_with_rois)

        assert storage_with_rois.list_rois()
        assert not storage_with_rois.list_features()

        assert storage_with_rois.get_status().roi_extracted
        assert storage_with_rois.get_status().feature_extracted

    def test_apply_with_filter_fill_lower_bound_with_inf(self, storage_with_rois, op):
        op.bounds = {"height": (None, 2000000.0)}
        op.apply(storage_with_rois)

        assert storage_with_rois.list_rois()
        assert storage_with_rois.list_features()

        assert storage_with_rois.get_status().roi_extracted
        assert storage_with_rois.get_status().feature_extracted

    def test_apply_with_filter_fill_upper_bound_with_inf(self, storage_with_rois, op):
        op.bounds = {"height": (0.0, None)}
        op.apply(storage_with_rois)

        assert storage_with_rois.list_rois()
        assert storage_with_rois.list_features()

        assert storage_with_rois.get_status().roi_extracted
        assert storage_with_rois.get_status().feature_extracted


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

    def test_apply(self, storage_with_features, op):
        op.apply(storage_with_features)

        assert all(x.data == op.feature_value for x in storage_with_features.list_features())
        assert storage_with_features.get_status().roi_extracted


class TestPipeline:
    def test_empty_pipeline_raises_error(self):
        with pytest.raises(pydantic.ValidationError):
            Pipeline(id="pipeline", operators=[])

    def test_validate_with_single_valid_operator(self):
        pipe = Pipeline(id="pipeline", operators=[helpers.DummyRoiExtractor()])
        pipe.validate_pipeline()

    def test_validate_with_invalid_operator_raises_error(self):
        with pytest.raises(PipelineConfigurationError):
            pipe = Pipeline(id="pipeline", operators=[helpers.DummyRoiTransformer()])
            pipe.validate_pipeline()

    def test_validate_with_multiple_operators_in_valid_order(self):
        pipe = Pipeline(
            id="test-pipeline",
            operators=[
                helpers.DummyRoiExtractor(id="op1"),
                helpers.DummyRoiTransformer(id="op2"),
                helpers.DummyFeatureExtractor(id="op3"),
                helpers.DummyFeatureTransformer(id="op4"),
            ],
        )
        pipe.validate_pipeline()

    def test_validate_with_multiple_operators_in_invalid_order_raises_error(self):
        with pytest.raises(PipelineConfigurationError):
            pipe = Pipeline(
                id="pipeline",
                operators=[
                    helpers.DummyRoiExtractor(id="op1"),
                    helpers.DummyFeatureTransformer(id="op2"),
                    helpers.DummyFeatureExtractor(id="op3"),
                ],
            )
            pipe.validate_pipeline()


# class TestProcessingPipeline:
#     def test_validate_processors_with_repeated_names_raises_error(
#         self,
#         roi_extractor: utils.DummyRoiExtractor,
#         feature_extractor: utils.DummyFeatureExtractor,
#     ):
#         feature_extractor.id = roi_extractor.id
#         id_ = "my-pipeline"
#         processing_steps = (roi_extractor, feature_extractor)
#         with pytest.raises(pydantic.ValidationError):
#             processors.ProcessingPipeline(id=id_, processors=processing_steps)

#     def test_validate_without_roi_extractor_in_first_step_raises_error(
#         self,
#         roi_transformer: utils.DummyRoiTransformer,
#         feature_extractor: utils.DummyFeatureExtractor,
#         feature_transformer: utils.DummyFeatureTransformer,
#     ):
#         id_ = "my-pipeline"
#         steps = (roi_transformer, feature_extractor, feature_transformer)
#         with pytest.raises(pydantic.ValidationError):
#             processors.ProcessingPipeline(id=id_, processors=steps)

#     def test_validate_feature_transformer_without_previous_feature_extractor_raises_error(
#         self,
#         roi_extractor: utils.DummyRoiExtractor,
#         roi_transformer: utils.DummyRoiTransformer,
#         feature_transformer: utils.DummyFeatureTransformer,
#     ):
#         id_ = "my-pipeline"
#         steps = (roi_extractor, roi_transformer, feature_transformer)

#         with pytest.raises(pydantic.ValidationError):
#             processors.ProcessingPipeline(id=id_, processors=steps)

#     def test_validate_feature_transformer_with_previous_feature_extractor_ok(
#         self,
#         roi_extractor: utils.DummyRoiExtractor,
#         roi_transformer: utils.DummyRoiTransformer,
#         feature_extractor: utils.DummyFeatureExtractor,
#         feature_transformer: utils.DummyFeatureTransformer,
#     ):
#         id_ = "my-pipeline"
#         steps = (roi_extractor, roi_transformer, feature_extractor, feature_transformer)
#         processors.ProcessingPipeline(id=id_, processors=steps)
#         assert True

#     def test_get_processor(self, pipeline: processors.ProcessingPipeline):
#         processor_id = "feature extractor"
#         processor = pipeline.get_processor(processor_id)
#         assert isinstance(processor, utils.DummyFeatureExtractor)

#     def test_get_processor_invalid_id_raises_error(self, pipeline: processors.ProcessingPipeline):
#         processor_id = "invalid processor id"
#         with pytest.raises(exceptions.ProcessorNotFound):
#             pipeline.get_processor(processor_id)

#     def test_to_dict(self, pipeline: processors.ProcessingPipeline):
#         d = pipeline.to_dict()
#         pipeline_from_dict = processors.ProcessingPipeline.from_dict(d)
#         assert pipeline_from_dict == pipeline


# def test_ProcessingPipeline_get_parameters(pipeline: assay.ProcessingPipeline):
#     for name, parameters in pipeline.get_parameters():
#         processor = pipeline.get_processor(name)
#         assert parameters == processor.get_parameters()


# def test_ProcessingPipeline_set_parameters(pipeline: assay.ProcessingPipeline):
#     new_parameters = {
#         "roi extractor": {"param1": 25, "param2": "new-value"},
#         "feature extractor": {
#             "param1": 15,
#             "param2": "new-value",
#             "filters": {"height": (10.0, None)},
#         },
#     }
#     pipeline.set_parameters(new_parameters)

#     for processor in pipeline.processors:
#         assert new_parameters[processor.name] == processor.get_parameters()


# def test_ProcessingPipeline_set_default_parameters():
#     processing_steps = [
#         ("ROI extractor", DummyRoiExtractor()),
#         ("Feature extractor", DummyFeatureExtractor()),
#     ]
#     pipeline = assay.ProcessingPipeline(processing_steps)
#     instrument = "qtof"
#     separation = "uplc"
#     pipeline.set_default_parameters(instrument, separation)
#     test_defaults = pipeline.get_parameters()

#     expected_defaults = list()
#     for name, processor in pipeline.processors:
#         processor.set_default_parameters(instrument, separation)
#         params = processor.get_parameters()
#         expected_defaults.append((name, params))
#     assert expected_defaults == test_defaults


# def test_ProcessingPipeline_process():
#     processing_steps = [
#         ("ROI extractor", DummyRoiExtractor()),
#         ("Feature extractor", DummyFeatureExtractor()),
#     ]
#     pipeline = assay.ProcessingPipeline(processing_steps)
#     sample = create_dummy_sample_data()
#     pipeline.process(sample)
