"""Utilities used by multiple storage classes."""

from typing import Callable

from numpy import array, float64, nanmean, unique, zeros

from ..annotation.consensus import create_consensus_annotation
from ..core.models import Annotation, FeatureGroup, GroupAnnotation, Sample
from ..core.utils.numpy import FloatArray


def create_matrix_data(
    descriptor: list[float],
    annotations: list[Annotation],
    samples: list[Sample],
    fill_values: dict[str, dict[int, float]],
) -> FloatArray:
    """Create the data matrix array."""
    sample_to_index = {s.id: k for k, s in enumerate(samples)}
    n_samples = len(sample_to_index)

    groups = set([x.group for x in annotations])
    groups.discard(-1)

    group_to_index = {g: k for k, g in enumerate(sorted(groups))}
    n_feature_groups = len(group_to_index)

    X = zeros((n_samples, n_feature_groups), dtype=float64)

    for value, ann in zip(descriptor, annotations):
        if ann.group > -1:
            row = sample_to_index[ann.sample_id]
            col = group_to_index[ann.group]
            X[row, col] = value

    for sample_id, sample_fills in fill_values.items():
        for group, value in sample_fills.items():
            row = sample_to_index[sample_id]
            col = group_to_index[group]
            X[row, col] = value

    return X


def create_feature_groups(
    descriptors: dict[str, list[float]],
    annotations: list[Annotation],
    agg: dict[str, Callable] | Callable | None = None,
) -> list[FeatureGroup]:
    """Compute feature groups from a feature list."""
    n_ft_descriptors = len(next(iter(descriptors.values())))
    assert n_ft_descriptors == len(annotations), "incompatible descriptor and annotation length."

    if isinstance(agg, dict):
        agg = agg.update({k: nanmean for k in descriptors if k not in agg})
    else:
        default_agg = agg or nanmean
        agg = {k: default_agg for k in descriptors}

    group_annotations = create_consensus_annotation(annotations)

    group_labels, group_index = unique(array([x.group for x in annotations], dtype="int64"), return_index=True)
    descriptor_array = {k: array(v) for k, v in descriptors.items()}
    features_groups = list()
    for label, index in zip(group_labels, group_index):
        if label == -1:
            continue

        assert agg is not None  # helping pyright...
        group_descriptors = {k: agg[k](v[index]).item() for k, v in descriptor_array.items()}
        ann = group_annotations.get(label) or GroupAnnotation(label=label)
        group = FeatureGroup(group=label, annotation=ann, descriptors=group_descriptors)
        features_groups.append(group)

    return features_groups
