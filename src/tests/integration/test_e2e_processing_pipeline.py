from tidyms2.core.enums import MSInstrument, Polarity, SeparationMode
from tidyms2.lcms import create_lcms_assay


def test_e2e_lcms_assay(lcms_sample_factory):
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
