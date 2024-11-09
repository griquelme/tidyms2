"""Annotation tools configuration models."""

import pydantic

from ..chem import EnvelopeValidatorConfiguration


class AnnotatorParameters(EnvelopeValidatorConfiguration):
    """Store isotopologue annotator parameters."""

    max_charge: int = 3
    """Maximum charge of the features. Use negative values for negative polarity."""

    min_similarity: float = pydantic.Field(default=0.9, ge=0.5, le=1.0)
    """Minimum cosine similarity between a pair of features"""

    min_p: float = pydantic.Field(default=0.005, ge=0.0, le=0.1)
    """Minimum abundance of isotopes to include in candidate search."""
