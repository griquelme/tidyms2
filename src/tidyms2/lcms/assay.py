"""Utilities to create LC-MS assays."""

from ..annotation.operators import IsotopologueAnnotator
from ..assay import Assay
from ..assay.executors import ParallelSampleProcessor, SequentialSampleProcessor
from ..core.enums import MSInstrument, Polarity, SeparationMode
from ..core.models import MZTrace
from ..storage.memory import OnMemoryAssayStorage
from ..storage.sqlite import SQLiteAssayStorage
from .models import Peak
from .operators import LCFeatureMatcher, LCPeakExtractor, LCTraceBaselineEstimator, LCTraceExtractor


def create_lcms_assay(
    id: str,
    *,
    instrument: MSInstrument,
    separation: SeparationMode,
    polarity: Polarity,
    annotate_isotopologues: bool = True,
    on_disk: bool = False,
    max_workers: int = 1,
) -> Assay[MZTrace, Peak]:
    """Create a new Assay instance for LC-MS data.

    :param id: the assay name
    :param instrument: the instrument used in the experimental measurements. Used to define operator defaults.
    :param separation: the separation mode used in the experimental measurements. Used to define operator defaults.
    :param polarity: the instrument polarity. Used to define operator defaults.
    :param instrument: the instrument used for the assay. Used to define operator defaults.
    :param annotate_isotopologues: If set to ``True`` and isotopologue annotation step is included in
        the sample pipeline.
    :param on_disk: store assay results on disk to reduce memory consumption. Recommended for large datasets.

    """
    if on_disk:
        host = f"{id}.sqlite"
        storage = SQLiteAssayStorage(id, host, MZTrace, Peak)
    else:
        storage = OnMemoryAssayStorage(id, MZTrace, Peak)

    if max_workers == 1:
        executor = SequentialSampleProcessor()
    else:
        executor = ParallelSampleProcessor(max_workers=max_workers)

    assay = Assay(id, storage, executor)

    sample_ops = list()
    sample_ops.append(LCTraceExtractor.from_defaults(instrument, separation, polarity))
    sample_ops.append(LCTraceBaselineEstimator.from_defaults(instrument, separation, polarity))
    sample_ops.append(LCPeakExtractor.from_defaults(instrument, separation, polarity))
    if annotate_isotopologues:
        sample_ops.append(IsotopologueAnnotator.from_defaults(instrument, separation, polarity))

    for op in sample_ops:
        op.id = f"{assay.id}-{op.__class__.__name__}"
        assay.pipes.sample.add_operator(op)

    assay.pipes.assay.add_operator(LCFeatureMatcher.from_defaults(instrument, separation, polarity))

    return assay
