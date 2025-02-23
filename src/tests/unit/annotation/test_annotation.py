import pytest

from tidyms2.annotation import annotation
from tidyms2.core.models import MZTrace
from tidyms2.core.operators.pipeline import Pipeline
from tidyms2.lcms import LCPeakExtractor, LCTraceBaselineEstimator, LCTraceExtractor, Peak
from tidyms2.simulation.lcms import SimulatedLCMSSampleFactory
from tidyms2.storage.memory import OnMemorySampleStorage


@pytest.fixture
def annotation_tools_params():
    return annotation.AnnotatorParameters(
        bounds={
            "C": (0, 50),
            "H": (0, 100),
            "O": (0, 20),
            "N": (0, 20),
            "Cl": (0, 2),
            "B": (0, 1),
        },
        max_M=2500,
        max_length=10,
        max_charge=3,
        min_M_tol=0.005,
        max_M_tol=0.01,
        p_tol=0.05,
        min_similarity=0.9,
        min_p=0.01,
    )


def test__annotate_empty_feature_list(annotation_tools_params):
    tools = annotation.create_annotation_tools(annotation_tools_params)
    feature_list = list()
    annotation.annotate(feature_list, *tools)


@pytest.fixture
def peak_list() -> list[Peak]:
    factory_config = {
        "config": {"n_scans": 300, "amp_noise": 0.0},
        "adducts": [
            {
                "formula": "[C10H20O2]-",
                "abundance": {"mu": 10000},
                "rt_mean": 50,
                "rt_width": 3.0,
                "n_isotopologues": 4,
            },
            {
                "formula": "[C10H20BO3]-",
                "abundance": {"mu": 20000},
                "rt_mean": 75,
                "rt_width": 3.0,
                "n_isotopologues": 4,
            },
            {
                "formula": "[C20H40BO5]2-",
                "abundance": {"mu": 30000},
                "rt_mean": 150,
                "rt_width": 3.0,
                "n_isotopologues": 4,
            },
            {
                "formula": "[C18H19N2O3]-",
                "abundance": {"mu": 25000},
                "rt_mean": 200,
                "rt_width": 3.0,
                "n_isotopologues": 4,
            },
            {
                "formula": "[C18H20N2O3Cl]-",
                "abundance": {"mu": 25000},
                "rt_mean": 200,
                "rt_width": 3.0,
                "n_isotopologues": 4,
            },
            {
                "formula": "[C10H20Cl]-",
                "abundance": {"mu": 20000},
                "rt_mean": 175,
                "rt_width": 3.0,
                "n_isotopologues": 4,
            },
        ],
    }
    factory = SimulatedLCMSSampleFactory(**factory_config)
    sample = factory("simulated_sample")

    data = OnMemorySampleStorage(sample, MZTrace, Peak)

    pipe = Pipeline(id="annotation_pipeline")
    pipe.add_operator(LCTraceExtractor(id="roi_extractor"))
    pipe.add_operator(LCTraceBaselineEstimator(id="baseline_estimator"))
    pipe.add_operator(LCPeakExtractor(id="peak_extractor"))
    pipe.apply(data)

    return data.list_features()


def test_annotate(peak_list: list[Peak], annotation_tools_params):
    tools = annotation.create_annotation_tools(annotation_tools_params)
    annotation.annotate(peak_list, *tools)

    # group features by isotopologue label.
    annotation_check = dict()
    for peak in peak_list:
        assert peak.annotation is not None
        group_list = annotation_check.setdefault(peak.annotation.isotopologue_label, list())
        group_list.append(peak)
    if -1 in annotation_check:
        annotation_check.pop(-1)
    assert len(annotation_check) == 6
    for v in annotation_check.values():
        assert len(v) == 4  # features where generated with 4 isotopologues.
