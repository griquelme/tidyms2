import pathlib

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from tidyms2.annotation import correspondence
from tidyms2.core.models import Annotation, AnnotationPatch, Sample
from tidyms2.core.utils.common import create_id


@pytest.mark.parametrize(
    "n_sample,n_cluster,max_size", [[20, 2, 10], [100, 4, 125], [200, 25, 1500], [200, 10, 20000]]
)
def test_cluster_db_scan_solution_is_equal_to_sklearn_solution(n_sample, n_cluster, max_size):
    X1 = np.arange(n_sample)
    X = np.vstack((X1, X1)).T
    X = np.repeat(X, n_cluster, axis=0)
    X = np.random.permutation(X)

    eps = 0.1
    min_samples = round(n_sample * 0.2)
    test_cluster = correspondence.cluster_dbscan(X, eps, min_samples, max_size)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="chebyshev")
    dbscan.fit(X)
    expected_cluster = dbscan.labels_
    assert np.array_equal(test_cluster, expected_cluster)


class TestEstimateNSpecies:
    @pytest.mark.parametrize(
        "min_samples,expected",
        [[1, np.array([2, 2])], [2, np.array([2, 2])], [3, np.array([0, 0])]],
    )
    def test_one_class(self, min_samples, expected):
        samples = np.array([0] * 4 + [1] * 4)  # 8 features detected in total in two samples
        clusters = np.array(([0] * 2 + [1] * 2) * 2)  # two clusters
        n_clusters = 2
        # two species in two clusters are expected
        res = correspondence._estimate_n_species_one_group(samples, clusters, min_samples, n_clusters)
        assert np.array_equal(res, expected)

    def test_estimate_n_species_multiple_groups(self):
        samples = np.array([0] * 4 + [1] * 4 + [2] * 4)  # 12 features in three samples
        clusters = np.array(([0] * 2 + [1] * 2) * 3)  # two clusters
        classes = np.array([0] * 8 + [1] * 4)  # two groups
        min_dr = 0.5
        # two species in two clusters are expected
        expected = {0: 2, 1: 2}
        include_classes = [0, 1]
        samples_per_class = {0: 2, 1: 1}

        res = correspondence._estimate_gmm_n_components(
            samples, clusters, classes, samples_per_class, include_classes, min_dr
        )
        assert res == expected


class TestEstimateDBSCANMinSample:
    @pytest.fixture
    def samples_per_class(self):
        res = {0: 8, 1: 16, 2: 24}
        return res

    def test_no_include_groups_ok(self, samples_per_class):
        min_fraction = 0.25
        include_classes = None
        test_min_samples = correspondence._estimate_dbscan_min_sample(samples_per_class, include_classes, min_fraction)
        expected_min_samples = round(sum(samples_per_class.values()) * min_fraction)
        assert expected_min_samples == test_min_samples

    def test_include_groups_ok(self, samples_per_class):
        min_fraction = 0.25
        include_classes = [0, 1]
        test_min_samples = correspondence._estimate_dbscan_min_sample(samples_per_class, include_classes, min_fraction)
        n_include = [v for k, v in samples_per_class.items() if k in include_classes]
        expected_min_samples = round(min(n_include) * min_fraction)
        assert expected_min_samples == test_min_samples


class TestProcessCluster:
    def test_one_species(self):
        np.random.seed(1234)
        # features
        n = 200
        X = np.random.normal(size=(n, 2))
        samples = np.arange(n)

        # add noise
        n_noise = 10
        noise = np.random.normal(size=(n_noise, 2), loc=4)
        X = np.vstack((X, noise))
        s_noise = np.random.choice(samples, size=n_noise)
        samples = np.hstack((samples, s_noise))

        expected = np.array([0] * n + [-1] * n_noise)

        n_species = 1
        max_deviation = 4
        labels, score = correspondence.split_cluster_gmm(X, samples, n_species, max_deviation)
        assert np.array_equal(labels, expected)

    def test_two_species(self):
        np.random.seed(1234)
        # features
        n = 200
        x_list = list()
        s_list = list()
        for loc in [0, 4]:
            x_list.append(np.random.normal(size=(n, 2), loc=loc))
            s_list.append(np.arange(n))

        # add noise
        n_noise = 10
        x_list.append(np.random.normal(size=(n_noise, 2), loc=8))
        X = np.vstack(x_list)
        s_list.append(np.random.choice(s_list[0], size=n_noise))
        samples = np.hstack(s_list)

        n_species = 2
        max_deviation = 4
        labels, score = correspondence.split_cluster_gmm(X, samples, n_species, max_deviation)

        # create expected array: it is created after actual computation because
        # it is not possible to know which label is assigned to each cluster
        expected_first_label, expected_second_label = (0, 1) if labels[0] == 0 else (1, 0)
        expected = np.array([expected_first_label] * n + [expected_second_label] * n + [-1] * n_noise)

        assert np.array_equal(labels, expected)


def test_match_features():
    np.random.seed(1234)
    descriptors = dict()
    n_noise = 10
    n_samples = 200
    samples = [Sample(id=f"sample-{x}", path=pathlib.Path(".")) for x in range(n_samples)]
    annotations = list()
    for n, loc in [(200, 0), (200, 4), (n_noise, 8)]:
        mz_list = descriptors.setdefault("mz", list())
        mz_list.extend(np.random.normal(size=n, loc=loc))

        rt_list = descriptors.setdefault("rt", list())
        rt_list.extend(np.random.normal(size=n, loc=loc))

        annotations.extend(Annotation(sample_id=f"sample-{x}", roi_id=create_id()) for x in range(n))

    tolerance = {"mz": 2.0, "rt": 2.0}

    params = correspondence.FeatureCorrespondenceParameters(max_deviation=4.0)
    actual = correspondence.match_features(descriptors, annotations, samples, tolerance, params)

    # expected result is created after actual computation because it is not
    # possible to know which label is assigned to each cluster before the algorithm runs
    expected_first_label, expected_second_label = (0, 1) if actual[0].value == 0 else (1, 0)
    expected_labels = np.array([expected_first_label] * 200 + [expected_second_label] * 200 + [-1] * n_noise)
    expected = [AnnotationPatch(id=x.id, field="group", value=y) for x, y in zip(annotations, expected_labels)]

    assert actual == expected


class TestComputeFeatureGroupCentroids:
    n_descriptors = 2

    def test_no_data_ok(self):
        X = np.array([], dtype=float)
        groups = np.array([], dtype=int)
        unique_groups, centroids = correspondence._compute_feature_group_centroids(X, groups)

        assert unique_groups.size == 0
        assert unique_groups.dtype == int

        assert centroids.size == 0
        assert centroids.dtype == float

    @pytest.mark.parametrize("n_ft", [1, 5, 10])
    def test_single_group_ok(self, n_ft):
        X = np.ones(shape=(n_ft, self.n_descriptors), dtype=float)
        groups = np.zeros(n_ft, dtype=int)

        expected_centroids = np.array([1.0 for _ in range(self.n_descriptors)])
        expected_unique_groups = np.array([0])
        actual_unique_groups, actual_centroids = correspondence._compute_feature_group_centroids(X, groups)

        assert np.array_equal(expected_unique_groups, actual_unique_groups)
        assert np.allclose(expected_centroids, actual_centroids)

    @pytest.mark.parametrize("n_ft1", [1, 5, 10])
    def test_multiple_group_ok(self, n_ft1):
        n_ft2 = 5

        ft1_constant = 1.0
        ft2_constant = 2.0
        X = np.vstack(
            (
                ft1_constant * np.ones(shape=(n_ft1, self.n_descriptors), dtype=float),
                ft2_constant * np.ones(shape=(n_ft2, self.n_descriptors), dtype=float),
            )
        )
        groups = np.hstack([np.zeros(n_ft1, dtype=int), np.ones(n_ft2, dtype=int)])

        expected_unique_groups = np.array([0, 1])
        expected_centroids = np.array([[x] * self.n_descriptors for x in [ft1_constant, ft2_constant]])

        actual_unique_groups, actual_centroids = correspondence._compute_feature_group_centroids(X, groups)

        assert np.array_equal(expected_unique_groups, actual_unique_groups)
        assert np.allclose(expected_centroids, actual_centroids)


class TestFindMergeCandidates:
    n_descriptors = 2
    tol = 0.1

    def test_no_candidates(self):
        centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
        groups = np.array([0, 1])

        candidates = correspondence._find_merge_candidates(centroids, groups, self.tol)
        assert not candidates

    def test_one_candidate(self):
        centroids = np.array([[0.0, 0.0], [1.0, 1.0], [1.05, 1.05]])
        groups = np.array([0, 1, 2])

        actual = correspondence._find_merge_candidates(centroids, groups, self.tol)
        expected = [(1, 2)]
        assert actual == expected

    def test_two_candidates(self):
        centroids = np.array([[0.0, 0.0], [1.0, 1.0], [1.05, 1.05], [2.0, 2.0], [2.05, 2.05]])
        groups = np.array([0, 1, 2, 3, 4])

        actual = correspondence._find_merge_candidates(centroids, groups, self.tol)
        expected = [(1, 2), (3, 4)]
        assert actual == expected

    def test_two_consecutive_candidates(self):
        centroids = np.array([[0.0, 0.0], [1.0, 1.0], [1.05, 1.05], [1.075, 1.1]])
        groups = np.array([0, 1, 2, 3])

        actual = correspondence._find_merge_candidates(centroids, groups, self.tol)
        expected = [(1, 2), (2, 3)]
        assert actual == expected


class TestMergeIfClose:
    # samples with 0.5 overlap
    group1 = 1
    samples1 = {"s1", "s2", "s3", "s4"}
    group2 = 2
    samples2 = {"s1", "s2"}

    def test_no_merge_if_sample_overlap_is_higher_than_max(self):
        max_sample_overlap = 0.25

        expected_merger, expected_merged = -1, -1
        actual_merger, actual_merged = correspondence._merge_if_close(
            self.group1, self.samples1, self.group2, self.samples2, max_sample_overlap
        )

        assert expected_merger == actual_merger
        assert expected_merged == actual_merged

    def test_no_merge_symmetric_result(self):
        max_sample_overlap = 0.25
        actual_merger, actual_merged = correspondence._merge_if_close(
            self.group1, self.samples1, self.group2, self.samples2, max_sample_overlap
        )
        swapped_merger, swapped_merged = correspondence._merge_if_close(
            self.group2, self.samples2, self.group1, self.samples1, max_sample_overlap
        )
        assert actual_merger == swapped_merger
        assert actual_merged == swapped_merged

    def test_merge_if_sample_overlap_is_lower_than_max(self):
        max_sample_overlap = 0.6

        expected_merger, expected_merged = 1, 2

        actual_merger, actual_merged = correspondence._merge_if_close(
            self.group1, self.samples1, self.group2, self.samples2, max_sample_overlap
        )

        assert expected_merger == actual_merger
        assert expected_merged == actual_merged

    def test_merge_symmetric_result(self):
        max_sample_overlap = 0.6
        actual_merger, actual_merged = correspondence._merge_if_close(
            self.group1, self.samples1, self.group2, self.samples2, max_sample_overlap
        )
        swapped_merger, swapped_merged = correspondence._merge_if_close(
            self.group2, self.samples2, self.group1, self.samples1, max_sample_overlap
        )
        assert actual_merger == swapped_merger
        assert actual_merged == swapped_merged


def test_merge_candidates():
    n = 2
    ft_constants = [0.0, 1.0, 2.0, 2.05]
    n_descriptors = 2

    X = np.vstack([x * np.ones(shape=(n, n_descriptors)) for x in ft_constants])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    samples = ["s1", "s2", "s1", "s2", "s1", "s2", "s3", "s4"]

    max_sample_overlap = 0.5
    tol = 0.1

    expected = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    actual = correspondence.merge_feature_groups(X, groups, samples, tol, max_sample_overlap)

    assert np.array_equal(actual, expected)
