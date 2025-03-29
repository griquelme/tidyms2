"""Cluster-based feature correspondence utilities."""

from collections import Counter
from functools import partial
from typing import Any, Generator, cast

import numpy as np
import pydantic
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from ..core.models import Annotation, AnnotationPatch, Sample
from ..core.utils.numpy import FloatArray, IntArray1D


class FeatureCorrespondenceParameters(pydantic.BaseModel):
    """Store match_features parameters."""

    groups: list[str] | None = None
    """If provided, use only samples from these groups to create feature groups."""

    min_fraction: float = 0.25
    """The minimum fraction of samples of a given group in a cluster. If `groups`
    is ``None``, the total number of sample is used to compute the minimum fraction.
    """

    max_deviation: float = 3.0
    """The maximum deviation of a feature from a cluster, measured in numbers of
    standard deviations from the cluster.
    """

    n_jobs: int | None = None
    """Number of jobs to run in parallel. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
    context. ``-1`` means using all processors."""

    max_sample_overlap: float = pydantic.Field(default=0.25, ge=0.0, le=1.0)
    """If `merge_close_features` is set to ``True``, close features will be
    merged if the number of samples where both features where found is higher
    than the number of samples where at least one of them was found. As an example,
    if one feature group has features detected in samples 1, 2, 3, 4 and the
    other has been detected in samples 1 and 2, they were both detected in
    two out of four samples (sample overlap = 0.5). In this case the features
    will not be merged. On the other hand id, if one feature is detected in
    samples 1, 2, 3, 4, and 5, while the other is detected only in sample 1, in
    this case, both feature groups will be merged into a single group.
    second
    """

    silent: bool = True
    """If ``False``, shows a progress bar."""


def match_features(
    descriptors: dict[str, list[float]],
    annotations: list[Annotation],
    sample_list: list[Sample],
    descriptor_tolerance: dict[str, float],
    params: FeatureCorrespondenceParameters,
) -> list[AnnotationPatch]:
    """Match features across samples using DBSCAN and GMM.

    See the :ref:`user guide <ft-correspondence>` for a detailed description of the algorithm.

    :param descriptors: feature descriptors
    :param tolerance: a dictionary that maps descriptors to tolerance value. Descriptors not included
        in this dictionary are not included in the algorithm.
    :param params: the algorithm parameters
    :return: a list annotation patches with feature group data

    """
    # adapt input to algorithm input
    tolerance = np.array(list(descriptor_tolerance.values()))

    # create feature matrix and scale descriptors
    X = np.vstack([descriptors[x] for x in descriptor_tolerance]).T
    n_features, _ = X.shape

    # scale X using minimum tolerance value
    argmin = np.argmin(tolerance).item()
    X = X * tolerance[argmin] / tolerance

    # create sample array
    sample_ids = [x.sample_id for x in annotations]
    sample_encoded, _ = encode(sample_ids)

    # create group array
    sample_id_to_sample = {x.id: x for x in sample_list}
    groups, group_mapping = encode([sample_id_to_sample[x.sample_id].meta.group for x in annotations])

    # count samples per group
    group_sample_pairs = {(x, y) for x, y in zip(groups, sample_encoded)}
    samples_per_group = Counter([x[0] for x in group_sample_pairs])

    if params.groups:
        include_groups = [group_mapping[x] for x in params.groups if x in group_mapping]
    else:
        include_groups = None

    # DBSCAN clustering
    min_samples = _estimate_dbscan_min_sample(samples_per_group, include_groups, params.min_fraction)
    max_size = 100000
    eps = tolerance[argmin]
    cluster = cluster_dbscan(X, eps, min_samples, max_size)

    # estimate the number of species per DBSCAN cluster
    species_per_cluster = _estimate_gmm_n_components(
        sample_encoded, cluster, groups, samples_per_group, include_groups, params.min_fraction
    )

    # split DBSCAN clusters with multiple species
    cluster_iterator = _get_cluster_iterator(X, cluster, sample_encoded, species_per_cluster)

    func = partial(_split_cluster_worker, max_deviation=params.max_deviation)
    func = delayed(func)
    data = Parallel(n_jobs=params.n_jobs)(func(x) for x in cluster_iterator)
    # TODO: Remove score computation.
    refined_cluster, score = _build_label(data, n_features)

    # TODO: remove parameter
    max_sample_overlap = 0.25
    refined_cluster = merge_feature_groups(X, refined_cluster, sample_ids, eps, max_sample_overlap)
    return [AnnotationPatch(id=x.id, field="group", value=y) for x, y in zip(annotations, refined_cluster)]


def encode(lst: list[str]) -> tuple[IntArray1D, dict[str, int]]:
    """Encode a list as an array of integers."""
    unique = set(lst)
    mapping = {v: k for k, v in enumerate(unique)}
    return np.array([mapping[x] for x in lst]), mapping


def _estimate_dbscan_min_sample(
    samples_per_group: dict[int, int], include_groups: list[int] | None, min_fraction: float
) -> int:
    """Compute the DBSCAN model parameter `min_sample`."""
    if include_groups is None:
        min_samples = round(sum(samples_per_group.values()) * min_fraction)
    else:
        min_samples = sum(samples_per_group.values())
        for k, v in samples_per_group.items():
            if k in include_groups:
                tmp = round(v * min_fraction)
                min_samples = min(tmp, min_samples)
    return min_samples


def cluster_dbscan(X: np.ndarray, eps: float, min_samples: int, max_size: int) -> np.ndarray:
    """Cluster rows of X using the DBSCAN algorithm.

    `X` is split into chunks to reduce memory usage. The split is done in a way
    such that the solution obtained is the same as the solution using `X`.

    Auxiliary function to match_features.

    :param X: m/z and rt values for each feature
    :param eps: Used to build epsilon parameter of DBSCAN
    :param min_samples: parameter to pass to DBSCAN
    :param max_size: maximum number of rows in X. If the number of rows is greater than
        this value, the data is processed in chunks to reduce memory usage.

    :return: the assigned cluster by DBSCAN

    """
    n_rows = X.shape[0]

    if n_rows > max_size:
        # sort X based on the values of the columns and find positions to
        # split into smaller chunks of data.
        x1 = X.T[1]
        sorted_index = np.argsort(x1)
        revert_sorted_index = np.argsort(sorted_index)
        X = X[sorted_index]

        # indices to split X based on max_size
        split_index = np.arange(max_size, n_rows, max_size)

        # find split indices candidates
        # it can be shown that if X is split at one of these points, the
        # points in each one of the chunks are not connected with points in
        # another chunk
        dx = np.diff(x1)
        split_candidates = np.where(dx > eps)[0]
        close_index = np.searchsorted(split_candidates, split_index)

        close_index[-1] = min(split_candidates.size - 1, close_index[-1])
        split_index = split_candidates[close_index] + 1
        split_index = np.hstack((0, split_index, n_rows))
    else:
        split_index = np.array([0, n_rows])
        revert_sorted_index = np.arange(X.shape[0])

    # cluster using DBSCAN on each chunk
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="chebyshev")
    n_chunks = split_index.size - 1
    cluster = np.zeros(X.shape[0], dtype=int)
    cluster_counter = 0
    for k in range(n_chunks):
        start = split_index[k]
        end = split_index[k + 1]
        dbscan.fit(X[start:end, :])
        labels = dbscan.labels_
        n_cluster = (np.unique(labels) >= 0).sum()
        labels[labels >= 0] += cluster_counter
        cluster_counter += n_cluster
        cluster[start:end] = labels

    # revert sort on cluster and X
    if n_rows > max_size:
        cluster = cluster[revert_sorted_index]
    return cluster


def _estimate_gmm_n_components(
    samples: np.ndarray[Any, np.dtype[np.integer]],
    clusters: np.ndarray[Any, np.dtype[np.integer]],
    groups: np.ndarray[Any, np.dtype[np.integer]],
    samples_per_group: dict[int, int],
    include_groups: list[int] | None,
    min_fraction: float,
) -> dict[int, int]:
    """Estimate the number of species in a cluster.

    Auxiliary function to match_features.

    :return: a mapping from cluster label to the number of species estimated.

    """
    if include_groups is None:
        include_groups = list(samples_per_group)
    n_clusters: int = np.max(clusters) + 1
    n_groups = len(include_groups)
    species_array = np.zeros((n_groups, n_clusters), dtype=int)
    # estimate the number of species in a cluster according to each group
    for k, cl in enumerate(include_groups):
        n_samples = samples_per_group[cl]
        n_min = round(n_samples * min_fraction)
        c_mask = groups == cl
        c_samples = samples[c_mask]
        c_clusters = clusters[c_mask]
        c_species = _estimate_n_species_one_group(c_samples, c_clusters, n_min, n_clusters)
        species_array[k, :] = c_species
    # keep the estimation with the highest number of species
    species = species_array.max(axis=0)
    n_species_per_cluster = dict(zip(np.arange(n_clusters), species))
    return n_species_per_cluster  # type: ignore


def _estimate_n_species_one_group(
    samples: np.ndarray, clusters: np.ndarray, min_samples: int, n_clusters: int
) -> np.ndarray:
    """Estimates the number of species in a cluster. Assumes only one group.

    Auxiliary function to _estimate_n_species.

    """
    species = np.zeros(n_clusters, dtype=int)
    for cl in range(n_clusters):
        c_mask = clusters == cl
        c_samples = samples[c_mask]
        # count features per sample in a cluster
        s_unique, s_counts = np.unique(c_samples, return_counts=True)
        # count the number of times a sample has k features
        k_unique, k_counts = np.unique(s_counts, return_counts=True)
        k_mask = k_counts >= min_samples
        k_unique = k_unique[k_mask]
        if k_unique.size:
            species[cl] = k_unique.max()
    return species


def _get_cluster_iterator(
    X: np.ndarray,
    cluster: np.ndarray,
    samples: np.ndarray,
    species_per_cluster: dict[int, int],
) -> Generator[tuple[np.ndarray, np.ndarray, int, np.ndarray], None, None]:
    """Yield the rows of X associated with a cluster.

    Auxiliary function to match_features.

    :yield: a tuple consisting of rows of X associated to a cluster, the sample labels associated to X_c
        and the number of species estimated for the cluster and indices of the rows of `X_c` in `X`

    """
    n_cluster = cluster.max() + 1
    for cl in range(n_cluster):
        n_species = species_per_cluster[cl]
        if n_species > 0:
            index = np.where(cluster == cl)[0]
            X_c = X[index, :]
            samples_c = samples[index]
            yield X_c, samples_c, n_species, index


def split_cluster_gmm(
    X_c: np.ndarray, samples_c: np.ndarray, n_species: int, max_deviation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Process each cluster using GMM.

    Auxiliary function to `match_features`.

    :return: a tuple with the labels and indecisiveness
    """
    # fit GMM
    n_rows = X_c.shape[0]
    if n_rows == 1:
        return samples_c, np.ones(n_rows, dtype=float)

    gmm = GaussianMixture(n_components=n_species, covariance_type="diag")
    gmm.fit(X_c)

    # compute the deviation of the features respect to each cluster
    deviation = _get_deviation(
        X_c,
        cast(np.ndarray[Any, np.dtype[np.floating]], gmm.covariances_),
        cast(np.ndarray[Any, np.dtype[np.floating]], gmm.means_),
    )

    # assign each feature in a sample to component in the GMM minimizing the
    # total deviation in the sample.
    label = -np.ones_like(samples_c)  # by default features are set as noise
    unique_samples = np.unique(samples_c)
    # the indecisiveness is a metric that counts the number of samples
    # where more than one feature can be potentially assigned to a species
    # that is, for each species, the number of rows in deviation with values
    # lower than max_deviation are counted as 1 and zero otherwise.
    # This is done for all samples and the indecisiveness is divided by the
    # number of samples.
    indecisiveness = np.zeros(n_species)
    for s in unique_samples:
        sample_mask = samples_c == s
        sample_deviation = deviation[sample_mask, :]
        # indecisiveness
        count = (sample_deviation < max_deviation).sum(axis=0)
        indecisiveness += count > 1

        # Find the best option for each feature
        best_row, best_col = linear_sum_assignment(sample_deviation)

        # features with deviation greater than max_deviation are set to noise
        valid_ft_mask = sample_deviation[best_row, best_col] <= max_deviation
        best_col = best_col[valid_ft_mask]
        best_row = best_row[valid_ft_mask]
        sample_label = -np.ones(sample_deviation.shape[0], dtype=int)
        sample_label[best_row] = best_col
        label[sample_mask] = sample_label
    indecisiveness /= unique_samples.size
    return label, indecisiveness


def _get_deviation(X: np.ndarray, covariances_: np.ndarray, means_: np.ndarray) -> np.ndarray:
    """Compute the deviation of features.

    Auxiliary function to _process_cluster.

    """
    n_species = covariances_.shape[0]
    n_ft = X.shape[0]
    deviation = np.zeros((n_ft, n_species))
    for k, (m, s) in enumerate(zip(means_, covariances_)):
        # the deviation is the absolute value of X after standardization
        Xs = np.abs((X - m) / np.sqrt(s))
        deviation[:, k] = Xs.max(axis=1)
    return deviation


def _split_cluster_worker(args, max_deviation):
    """Worker used to parallelize feature clustering."""
    Xc, samples_c, n_ft, index = args
    label_c, score = split_cluster_gmm(Xc, samples_c, n_ft, max_deviation)
    return label_c, score, index, n_ft


def _build_label(data, size):
    """Merge the data obtained from each cluster.

    Auxiliary function to match_features.

    """
    label = -1 * np.ones(size, dtype=int)
    cluster_count = 0
    score_list = list()
    for c_label, c_score, c_index, c_n_ft in data:
        c_label[c_label > -1] += cluster_count
        label[c_index] = c_label
        cluster_count += c_n_ft
        score_list.append(c_score)
    score_list = np.hstack(score_list)
    return label, score_list


def _get_progress_bar_total(ft_per_cluster: dict[int, int]) -> int:
    total = 0
    for k, v in ft_per_cluster.items():
        if (k > -1) and (v > 0):
            total += 1
    return total


def merge_feature_groups(
    X: FloatArray, groups: IntArray1D, samples: list[str], tol: float, max_sample_overlap: float
) -> IntArray1D:
    """Merge close features.

    Feature groups are merged if the two following conditions are meet:

    1.  Each feature group centroid must be closer than the corresponding tolerance.
    2.  The overlap between feature groups must be lower than the `max_sample_overlap` parameter.

    Groups are checked an merged in a pairwise fashion. If a feature group appear
    in more than one pair, then only one is considered. Then after all features
    groups are checked, the algorithm is run until candidate pairs are no longer found.

    """
    unique_groups, centroids = _compute_feature_group_centroids(X, groups)
    group_to_samples = _create_group_to_samples_dict(groups, samples)

    candidates = _find_merge_candidates(centroids, unique_groups, tol)

    if not candidates:
        return groups

    merge_map: dict[int, int] = dict()

    for group1, group2 in candidates:
        if group1 in merge_map or group2 in merge_map:
            continue

        samples1 = group_to_samples[group1]
        samples2 = group_to_samples[group2]

        merger, merged = _merge_if_close(group1, samples1, group2, samples2, max_sample_overlap)

        if merger > -1 and merged > -1:
            merge_map[merged] = merger

    groups = np.array([merge_map[x] if x in merge_map else x for x in groups])
    return merge_feature_groups(X, groups, samples, tol, max_sample_overlap)


def _compute_feature_group_centroids(X: FloatArray, groups: IntArray1D) -> tuple[IntArray1D, FloatArray]:
    """Compute the mean value for each descriptor in a feature group."""
    sorted_index = np.argsort(groups)

    X = X[sorted_index]
    groups = groups[sorted_index]

    # split X into so rows belong to the same feature group
    unique_groups, group_sections = np.unique(groups, return_index=True)

    if unique_groups.size:
        group_sections = group_sections[1:]  # remove first index so it matches split expected format
        centroids = np.vstack([x.mean(axis=0) for x in np.split(X, group_sections)])
    else:
        # TODO: check if this is the best approach for the empty case
        centroids = np.array([], dtype=X.dtype)
    return unique_groups, centroids


def _find_merge_candidates(centroids: FloatArray, groups: IntArray1D, tol: float) -> list[tuple[int, int]]:
    """Group a pair of feature groups if all of their descriptor centroids are closer than the tolerance."""
    sorted_centroids_index = np.lexsort(centroids.T)
    groups = groups[sorted_centroids_index]
    rows_close = np.abs(np.diff(centroids[sorted_centroids_index], axis=0)) < tol
    diff_rows = np.all(rows_close, axis=1)

    candidates = list()
    for i in np.where(diff_rows)[0]:
        candidates.append((groups[i], groups[i + 1]))
    return candidates


def _create_group_to_samples_dict(groups: IntArray1D, samples: list[str]) -> dict[int, set[str]]:
    """Create a dictionary that maps feature groups to samples where a feature was detected."""
    samples_per_group = dict()

    for g, s in zip(groups, samples):
        g_samples = samples_per_group.setdefault(g, set())
        g_samples.add(s)

    return samples_per_group


def _merge_if_close(
    group1: int, samples1: set[str], group2: int, samples2: set[str], max_sample_overlap: float
) -> tuple[int, int]:
    """Flag a pair of feature groups for merging if the sample overlap is lower than maximum specified.

    Returns
    -------
    merger : int
        The feature group label that will be kept after the merge. If no merge is
        is to be made, this value is set to ``-1``.
    merged : int
        The feature group label that will be deleted after the merge. If no merge is
        is to be made, this value is set to ``-1``.

    """
    total = len(samples1.union(samples2))
    n_overlap = len(samples1.intersection(samples2))

    if (n_overlap / total) > max_sample_overlap:
        return (-1, -1)
    return (group1, group2) if len(samples1) >= len(samples2) else (group2, group1)
