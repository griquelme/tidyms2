"""Common utilities for data simulation."""

from pathlib import Path
from random import choice

from ..core.enums import SampleType
from ..core.models import Sample


def create_sample_list(
    sample_types: list[SampleType],
    type_to_groups: dict[SampleType, list[str]] | None = None,
    n_batches: int = 1,
    start_order: int = 0,
) -> list[Sample]:
    """Create a list of sample models using a list of sample types as template.

    :param sample_types: the type of each simulated sample
    :param type_to_groups: sample groups to assign randomly to each sample. If not provided,
        the sample type is used.
    :param n_batches: the number of batches to simulate. If ``n_batches > 1`` then the sample
        types template is concatenated to build each batch of samples increasing the
        sample order and batch accordingly. Note that sample groups may vary across
        batches for due to the fact that the sample group is randomly sampled from
        `type_to_groups`.
    :param start_order: the start value for order.

    """
    if type_to_groups is None:
        type_to_groups = {x: [x] for x in sample_types}

    samples = list()
    for b in range(n_batches):
        batch = [_create_sample(t, i, b, type_to_groups) for i, t in enumerate(sample_types, start=start_order)]
        samples.extend(batch)
        start_order += len(batch)
    return samples


def _create_sample(type_: SampleType, order: int, batch: int, type_to_groups: dict[SampleType, list[str]]) -> Sample:
    if type_ not in type_to_groups:
        raise ValueError(f"sample type {type_} not included `type_to_groups`.")
    return Sample.model_validate(
        {
            "id": f"sample-{order}",
            "path": Path("."),
            "meta": {
                "group": choice(type_to_groups[type_]),
                "order": order,
                "batch": batch,
                "type": type_,
            },
        }
    )
