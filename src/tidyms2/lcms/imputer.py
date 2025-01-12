"""LC-MS missing imputer."""

from __future__ import annotations

import bisect
import concurrent.futures
from logging import getLogger
from typing import Generator

from scipy.integrate import trapezoid

from ..algorithms.raw import MakeChromatogramParameters, make_chromatograms
from ..core.models import FeatureGroup, FillValue, Sample
from ..core.storage import AssayStorage
from ..io import MSData

logger = getLogger(__name__)


def compute_fill_values_lc(data: AssayStorage, mz_tolerance: float, max_workers: int | None = None) -> list[FillValue]:
    r"""Compute fill missing values using the chromatogram area.

    :param data:
        Assay data with features matched
    :param mz_tolerance:
        m/z tolerance used to build chromatograms
    :param max_workers:
        maximum number of workers for parallel execution
    :return:
        a list of fill values.

    Notes
    -----
    Missing Chromatographic peaks are searched in the raw data using the
    peak descriptors obtained after features extraction and correspondence.
    Initially, chromatograms for missing features in each sample are built by
    using the mean m/z from the features descriptors. A time window for
    searching the peaks is defined based on the mean Rt and the standard
    deviation values features:

    .. math::

        t_{mean} \pm n_{dev} t_{std}

    If a peak is found in the region, its area is used as fill value. If more
    than one peak is found in the region, the closes to the mean Rt is chosen.
    If no peak was found in the region, an estimation of the peak area is
    computed by integrating the region comprised by the mean start and end
    times of the peak in the detected features.

    """
    all_samples = data.list_samples()
    all_groups = data.fetch_feature_groups()

    def iterator(
        samples: list[Sample], groups: list[FeatureGroup]
    ) -> Generator[tuple[Sample, list[FeatureGroup]], None, None]:
        sample_id_to_sample = {x.id: x for x in samples}
        group_id_to_group = {x.group: x for x in groups}
        sample_to_groups = {x: set(group_id_to_group) for x in sample_id_to_sample}

        for ann in data.fetch_annotations():
            if ann.group == -1:
                continue

            sample_groups = sample_to_groups.get(ann.sample_id)
            if sample_groups is None:
                continue

            sample_groups.remove(ann.group)

        for sample_id, missing_groups in sample_to_groups.items():
            yield sample_id_to_sample[sample_id], [group_id_to_group[x] for x in missing_groups]

    result = list()

    params = MakeChromatogramParameters(window=mz_tolerance, fill_missing=True, mz=list())
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        n_samples = len(all_samples)
        futures = [
            executor.submit(_worker, sample, group, params) for sample, group in iterator(all_samples, all_groups)
        ]
        for k, future in enumerate(concurrent.futures.as_completed(futures)):
            sample_fill_values = future.result()
            logger.info(f"Processing {k}/{n_samples}).")
            result.append(sample_fill_values)
    return result


def _worker(
    sample: Sample,
    features: list[FeatureGroup],
    params: MakeChromatogramParameters,
) -> list[FillValue]:
    """Find fill values for features not detected during feature extraction in a sample."""
    ms_data = MSData(sample)
    params = params.model_copy()
    params.mz = [x.mz for x in features]
    chromatograms = make_chromatograms(ms_data, params)

    fill_values = list()
    for chrom, group in zip(chromatograms, features):
        start = bisect.bisect_left(chrom.time, group.descriptors["rt_start"])
        end = bisect.bisect_left(chrom.time, group.descriptors["rt_end"])
        value = trapezoid(chrom.int[start:end], chrom.time[start:end])
        fill = FillValue(sample_id=sample.id, feature_group=group.group, value=value)
        fill_values.append(fill)
    return fill_values
