from tidyms2.core.enums import MSInstrument, Polarity, SeparationMode
from tidyms2.lcms import create_lcms_assay


def test_e2e_lcms_assay_on_memory_storage_sequential_sample_executor(lcms_sample_factory):
    assay = create_lcms_assay(
        "test-lcms-assay-e2e",
        instrument=MSInstrument.QTOF,
        separation=SeparationMode.UPLC,
        polarity=Polarity.POSITIVE,
        annotate_isotopologues=True,
    )
    n_samples = 10
    samples = (lcms_sample_factory(f"sample-{k}", order=k) for k in range(n_samples))
    assay.add_samples(*samples)

    assay.process_samples()
    assay.process_assay()
    matrix = assay.create_data_matrix()

    assert matrix.get_n_samples() == n_samples
    assert matrix.get_n_features() == 6  # see lcms_sample_factory on conftest


def test_e2e_lcms_assay_on_disk_storage_sequential_sample_executor(lcms_sample_factory, tmp_path):
    assay = create_lcms_assay(
        "test-lcms-assay-e2e",
        instrument=MSInstrument.QTOF,
        separation=SeparationMode.UPLC,
        polarity=Polarity.POSITIVE,
        annotate_isotopologues=True,
        on_disk=True,
        storage_path=str(tmp_path / "data.db"),
    )
    n_samples = 10
    samples = (lcms_sample_factory(f"sample-{k}", order=k) for k in range(n_samples))
    assay.add_samples(*samples)

    assay.process_samples()
    assay.process_assay()

    matrix = assay.create_data_matrix()

    assert matrix.get_n_samples() == n_samples
    assert matrix.get_n_features() == 6  # see lcms_sample_factory on conftest


def test_e2e_lcms_assay_on_memory_storage_parallel_sample_executor(lcms_sample_factory):
    assay = create_lcms_assay(
        "test-lcms-assay-e2e",
        instrument=MSInstrument.QTOF,
        separation=SeparationMode.UPLC,
        polarity=Polarity.POSITIVE,
        annotate_isotopologues=True,
        max_workers=2,
    )
    n_samples = 10
    samples = (lcms_sample_factory(f"sample-{k}", order=k) for k in range(n_samples))
    assay.add_samples(*samples)

    assay.process_samples()
    assay.process_assay()

    matrix = assay.create_data_matrix()

    assert matrix.get_n_samples() == n_samples
    assert matrix.get_n_features() == 6  # see lcms_sample_factory on conftest


def test_e2e_lcms_assay_on_disk_storage_parallel_sample_executor(lcms_sample_factory, tmp_path):
    assay = create_lcms_assay(
        "test-lcms-assay-e2e",
        instrument=MSInstrument.QTOF,
        separation=SeparationMode.UPLC,
        polarity=Polarity.POSITIVE,
        annotate_isotopologues=True,
        on_disk=True,
        max_workers=2,
        storage_path=str(tmp_path / "data.db"),
    )
    n_samples = 10
    samples = (lcms_sample_factory(f"sample-{k}", order=k) for k in range(n_samples))
    assay.add_samples(*samples)

    assay.process_samples()
    assay.process_assay()

    matrix = assay.create_data_matrix()

    assert matrix.get_n_samples() == n_samples
    assert matrix.get_n_features() == 6  # see lcms_sample_factory on conftest
