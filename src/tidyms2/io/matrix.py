"""Utilities to read matrix data in a variety of formats."""

import csv
from pathlib import Path

from numpy import array

from ..core.matrix import DataMatrix
from ..core.models import FeatureGroup, GroupAnnotation, Sample, SampleMetadata

PROGENESIS_WIDTH = "Chromatographic peak width (min)"
PROGENESIS_RT = "Retention time (min)"
PROGENESIS_CHARGE = "Charge"
PROGENESIS_MZ = "m/z"
PROGENESIS_NAME = "Compound"
NORMALIZED_HEADER_FIELD = "Normalised abundance"
RAW_HEADER_FIELD = "Raw abundance"


class ProgenesisReaderError(ValueError):
    """Exception raised when parsing a progenesis file fails."""


def read_progenesis(path: Path) -> DataMatrix:
    """Read progenesis CSV data into a data matrix.

    :param path: Path to the Progenesis data file.

    """
    with path.open() as fp:
        reader = csv.reader(fp)

        # first row contains the location of normalized and raw data
        first_row = next(reader)
        ft_meta_start, ft_meta_end, data_start, data_end = _get_data_location(first_row)

        # second row contain sample groups
        second_row = next(reader)
        sample_groups = _get_sample_groups(second_row, data_start, data_end)

        # third row contain sample ids and feature metadata fields
        third_row = next(reader)

        sample_ids = third_row[data_start:data_end]
        ft_meta_fields = third_row[ft_meta_start:ft_meta_end]

        samples = _create_samples(sample_ids, sample_groups)

        meta_field_to_index = _map_ft_meta_field_to_index(ft_meta_fields, ft_meta_start)

        columns = list()
        feature_groups = list()
        for k, row in enumerate(reader):
            feature_groups.append(_create_feature_group(row, meta_field_to_index, k))
            columns.append(_create_matrix_column(row, data_start, data_end))

        X = array(columns).T

    return DataMatrix(samples, feature_groups, X)


def _get_data_location(first_row: list[str]) -> tuple[int, int, int, int]:
    """Find indices where feature metadata ends and matrix data starts."""
    try:
        ft_meta_end = first_row.index(NORMALIZED_HEADER_FIELD)
    except ValueError:
        raise ProgenesisReaderError(f"Header row does not contain header field `{NORMALIZED_HEADER_FIELD}`.")

    try:
        data_start = first_row.index(RAW_HEADER_FIELD)
    except ValueError:
        raise ProgenesisReaderError(f"Header row does not contain header field `{RAW_HEADER_FIELD}`.")

    ft_meta_start = 0
    data_start = data_start
    data_end = 2 * data_start - ft_meta_end
    return ft_meta_start, ft_meta_end, data_start, data_end


def _get_sample_groups(second_row: list[str], data_start: int, data_end: int) -> list[str]:
    groups = list()

    if not second_row:
        raise ProgenesisReaderError("Expected second data row to contain sample group data but found None.")

    prev = ""
    for g in second_row[data_start:data_end]:
        current = g or prev
        groups.append(current)
        prev = g
    return groups


def _create_samples(sample_ids: list[str], sample_groups: list[str]) -> list[Sample]:
    if len(sample_ids) != len(sample_groups):
        raise ProgenesisReaderError("Number of samples ids and sample groups in data do not match.")
    samples = list()
    for order, (id_, group) in enumerate(zip(sample_ids, sample_groups)):
        s = Sample(id=id_, path=Path("."), meta=SampleMetadata(order=order, group=group))
        samples.append(s)
    return samples


def _map_ft_meta_field_to_index(fields: list[str], ft_meta_start: int) -> dict[str, int]:
    """Create a mapping from field names to indices for efficient access to feature metadata."""
    field_to_index = dict()
    for field in [PROGENESIS_NAME, PROGENESIS_MZ, PROGENESIS_CHARGE, PROGENESIS_RT, PROGENESIS_WIDTH]:
        try:
            field_to_index[field] = fields.index(field) + ft_meta_start
        except ValueError:
            raise ProgenesisReaderError(f"`{field}` not found in data file.")
    return field_to_index


def _create_feature_group(row: list[str], field_to_index: dict[str, int], group: int) -> FeatureGroup:
    charge = int(row[field_to_index[PROGENESIS_CHARGE]])
    name = row[field_to_index[PROGENESIS_NAME]]  # progenesis internal name
    ann = GroupAnnotation(label=group, charge=charge, name=name)

    descriptors = {
        "mz": float(row[field_to_index[PROGENESIS_MZ]]),
        "rt": float(row[field_to_index[PROGENESIS_RT]]) * 60,  # progenesis express time in min
        "width": float(row[field_to_index[PROGENESIS_WIDTH]]) * 60,
    }
    return FeatureGroup(group=group, annotation=ann, descriptors=descriptors)


def _create_matrix_column(row: list[str], start: int, end: int) -> list[float]:
    return [float(x or 0.0) for x in row[start:end]]
