import pytest

from tidyms2.core.enums import SampleType
from tidyms2.simulation.utils import create_sample_list


def test_simulate_empty_sample_list():
    simulated = create_sample_list(list())
    assert not simulated


def test_simulate_no_groups_groups_are_equal_to_sample_type():
    sample_types = [SampleType.TECHNICAL_QC, SampleType.SAMPLE]
    simulated = create_sample_list(sample_types)
    assert len(simulated) == len(sample_types)
    for sample in simulated:
        assert sample.meta.type == sample.meta.group


def test_simulate_with_start_order():
    sample_types = [SampleType.TECHNICAL_QC, SampleType.SAMPLE]
    start_at = 5
    simulated = create_sample_list(sample_types, start_order=start_at)
    assert len(simulated) == len(sample_types)
    for order, sample in enumerate(simulated, start=start_at):
        assert sample.meta.type == sample.meta.group
        assert order == sample.meta.order


def test_simulate_with_type_to_groups_missing_group_raises_error():
    sample_types = [SampleType.TECHNICAL_QC, SampleType.SAMPLE]
    type_to_groups = {SampleType.SAMPLE: ["group1"]}
    with pytest.raises(ValueError):
        create_sample_list(sample_types, type_to_groups=type_to_groups)


def test_simulate_with_type_to_groups():
    sample_types = [SampleType.TECHNICAL_QC, SampleType.SAMPLE]
    type_to_groups = {SampleType.SAMPLE: ["group1"], SampleType.TECHNICAL_QC: ["QC"]}
    simulated = create_sample_list(sample_types, type_to_groups=type_to_groups)
    for sample in simulated:
        # FIXME: fix when SampleMeta.type is a SampleType
        assert sample.meta.group in type_to_groups[sample.meta.type]  # type: ignore


def test_simulate_multiple_batches():
    sample_types = [SampleType.TECHNICAL_QC, SampleType.SAMPLE]
    n_batches = 5
    simulated = create_sample_list(sample_types, n_batches=n_batches)
    for order, sample in enumerate(simulated):
        assert sample.meta.order == order
        assert sample.meta.batch == order // len(sample_types)
