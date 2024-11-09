"""Annotate function implementation."""

from typing import Sequence

from ..chem import DEFAULT_CONTEXT, EM, ChemicalContext, EnvelopeValidator
from ..core.models import AnnotableFeature, AnnotableFeatureType
from .annotation_data import AnnotationData
from .config import AnnotatorParameters
from .envelope_finder import EnvelopeFinder
from .mmi_finder import MMIFinder


def create_annotation_tools(
    config: AnnotatorParameters, context: ChemicalContext | None = None
) -> tuple[MMIFinder, EnvelopeFinder, EnvelopeValidator]:
    """Create an annotator object. Auxiliary function to _annotate.

    :param config: the annotation parameters
    :param context: the chemical context used in during annotation

    """
    context = DEFAULT_CONTEXT if context is None else context
    bin_size = 100
    mmi_finder: MMIFinder = MMIFinder(config, bin_size, context)
    envelope_finder: EnvelopeFinder = EnvelopeFinder(config, context)
    envelope_validator = EnvelopeValidator(config)
    return mmi_finder, envelope_finder, envelope_validator


def annotate(
    feature_list: Sequence[AnnotableFeature],
    mmi_finder: MMIFinder,
    envelope_finder: EnvelopeFinder,
    envelope_validator: EnvelopeValidator,
) -> None:
    """Annotate isotopologues in a sample.

    Annotations are added to the `annotation` attribute of each feature.

    :param feature_list: list of features obtained after feature extraction.
    :param mmi_finder: MMIFinder
    :param envelope_finder: EnvelopeFinder
    :param envelope_validator: EnvelopeValidator

    """
    data = AnnotationData(feature_list)
    monoisotopologue = data.get_monoisotopologue()
    polarity = mmi_finder.polarity
    while monoisotopologue is not None:
        mmi_candidates = mmi_finder.find(data)
        envelope, charge = find_best_envelope(
            data,
            monoisotopologue,
            polarity,
            mmi_candidates,
            envelope_finder,
            envelope_validator,
        )
        data.annotate(envelope, charge)
        monoisotopologue = data.get_monoisotopologue()


def find_best_envelope(
    data: AnnotationData[AnnotableFeatureType],
    monoisotopologue: AnnotableFeatureType,
    polarity: int,
    mmi_candidates: Sequence[tuple[AnnotableFeatureType, int]],
    envelope_finder: EnvelopeFinder,
    envelope_validator: EnvelopeValidator,
) -> tuple[Sequence[AnnotableFeatureType], int]:
    """Find the most fitting envelope candidate."""
    best_length = 1
    best_candidate = [monoisotopologue]
    best_charge = -1
    for mmi, charge in mmi_candidates:
        envelope_candidates = envelope_finder.find(data, mmi, charge)
        for candidate in envelope_candidates:
            validated_length = _validate_candidate(
                candidate,
                monoisotopologue,
                charge,
                polarity,
                best_length,
                envelope_validator,
            )
            if validated_length > best_length:
                best_length = validated_length
                best_candidate = candidate[:validated_length]
                best_charge = charge
    return best_candidate, best_charge


def _validate_candidate(
    candidate: Sequence[AnnotableFeatureType],
    monoisotopologue: AnnotableFeatureType,
    charge: int,
    polarity: int,
    min_length: int,
    validator: EnvelopeValidator,
) -> int:
    if len(candidate) <= min_length:
        return 0

    if monoisotopologue not in candidate:
        return 0

    envelope = candidate[0].compute_isotopic_envelope(*candidate)
    em_correction = EM * charge * polarity
    M = [x * charge - em_correction for x in envelope.mz]
    return validator.validate(M, envelope.p)
