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
    samples = (lcms_sample_factory(f"sample-{k}", order=k) for k in range(10))
    assay.add_samples(*samples)

    assay.process_samples()
    assay.process_assay()


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
    samples = (lcms_sample_factory(f"sample-{k}", order=k) for k in range(10))
    assay.add_samples(*samples)

    assay.process_samples()
    assay.process_assay()


def test_e2e_lcms_assay_on_memory_storage_parallel_sample_executor(lcms_sample_factory):
    assay = create_lcms_assay(
        "test-lcms-assay-e2e",
        instrument=MSInstrument.QTOF,
        separation=SeparationMode.UPLC,
        polarity=Polarity.POSITIVE,
        annotate_isotopologues=True,
        max_workers=2,
    )
    samples = (lcms_sample_factory(f"sample-{k}", order=k) for k in range(10))
    assay.add_samples(*samples)

    assay.process_samples()
    assay.process_assay()


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
    samples = (lcms_sample_factory(f"sample-{k}", order=k) for k in range(10))
    assay.add_samples(*samples)

    assay.process_samples()
    assay.process_assay()
