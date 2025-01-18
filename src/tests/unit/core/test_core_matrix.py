import pathlib
import random

import numpy
import pytest

from tidyms2.core import exceptions
from tidyms2.core.matrix import DataMatrix, FeatureVector, SampleVector, validate_data_matrix
from tidyms2.core.models import FeatureGroup, GroupAnnotation, Sample

from ..helpers import create_sample


def create_feature_group(group: int) -> FeatureGroup:
    ann = GroupAnnotation(label=group)
    descriptors = {"mz": random.uniform(100.0, 1000.0)}
    return FeatureGroup(group=group, annotation=ann, descriptors=descriptors)


def ft_transformer(data: FeatureVector, **kwargs):
    """Feature transformer function for tests."""
    data.data[:] = 1.0
    return data


def sample_transformer(data: SampleVector, **kwargs):
    """Feature transformer function for tests."""
    data.data[:] = 1.0
    return data


class TestValidateDataMatrix:
    def test_create_matrix_using_samples_with_repeated_ids_raise_error(self, tmp_path: pathlib.Path):
        samples = [create_sample(tmp_path, 1), create_sample(tmp_path, 1)]
        features = [create_feature_group(x) for x in range(5)]
        data = numpy.random.normal(loc=100.0, size=(2, 5))
        with pytest.raises(exceptions.RepeatedIdError):
            validate_data_matrix(samples, features, data)

    def test_create_matrix_using_samples_with_repeated_order_raise_error(self, tmp_path: pathlib.Path):
        s1 = create_sample(tmp_path, 1)
        s1.meta.order = 1
        s2 = create_sample(tmp_path, 2)
        s2.meta.order = 1
        samples = [s1, s2]
        features = [create_feature_group(x) for x in range(5)]
        data = numpy.random.normal(loc=100.0, size=(2, 5))
        with pytest.raises(exceptions.RepeatedSampleOrder):
            validate_data_matrix(samples, features, data)

    def test_create_matrix_using_features_with_repeated_group_raise_error(self, tmp_path: pathlib.Path):
        samples = [create_sample(tmp_path, 1), create_sample(tmp_path, 1)]
        features = [create_feature_group(x) for x in range(5)]
        features[2].group = 3
        features[3].group = 3
        data = numpy.random.normal(loc=100.0, size=(2, 5))
        with pytest.raises(exceptions.RepeatedIdError):
            validate_data_matrix(samples, features, data)

    def test_create_matrix_with_non_matching_sizes_in_sample_data_raises_error(self, tmp_path: pathlib.Path):
        samples = [create_sample(tmp_path, 1), create_sample(tmp_path, 1)]
        features = [create_feature_group(x) for x in range(5)]
        data = numpy.random.normal(loc=100.0, size=(3, 5))
        with pytest.raises(ValueError):
            validate_data_matrix(samples, features, data)

    def test_create_matrix_with_non_matching_sizes_in_feature_data_raises_error(self, tmp_path: pathlib.Path):
        samples = [create_sample(tmp_path, 1), create_sample(tmp_path, 1)]
        features = [create_feature_group(x) for x in range(5)]
        data = numpy.random.normal(loc=100.0, size=(2, 6))
        with pytest.raises(ValueError):
            validate_data_matrix(samples, features, data)

    def test_create_matrix_without_samples_raises_error(self, tmp_path: pathlib.Path):
        samples = list()
        features = [create_feature_group(x) for x in range(5)]
        data = numpy.random.normal(loc=100.0, size=(2, 5))
        with pytest.raises(exceptions.EmptyDataMatrix):
            validate_data_matrix(samples, features, data)

    def test_create_matrix_without_features_raises_error(self, tmp_path: pathlib.Path):
        samples = [create_sample(tmp_path, 1), create_sample(tmp_path, 1)]
        features = []
        data = numpy.random.normal(loc=100.0, size=(2, 5))
        with pytest.raises(exceptions.EmptyDataMatrix):
            validate_data_matrix(samples, features, data)

    def test_create_matrix_using_non_2d_array_shape_raises_error(self, tmp_path: pathlib.Path):
        samples = [create_sample(tmp_path, x) for x in range(3)]
        features = [create_feature_group(x) for x in range(5)]
        data = numpy.random.normal(loc=100.0, size=(3))
        with pytest.raises(ValueError):
            validate_data_matrix(samples, features, data)

    def test_create_matrix_using_non_float_array_shape_raises_error(self, tmp_path: pathlib.Path):
        samples = [create_sample(tmp_path, x) for x in range(3)]
        features = [create_feature_group(x) for x in range(5)]
        data = numpy.ones(shape=(3, 5), dtype=int)
        with pytest.raises(ValueError):
            validate_data_matrix(samples, features, data)


class TestDataMatrix:
    n_samples = 10
    n_features = 20

    def test_create_matrix_using_samples_unsorted_samples_sorts_by_order(self, tmp_path: pathlib.Path):
        samples = list(reversed([create_sample(tmp_path, x) for x in range(3)]))
        features = [create_feature_group(x) for x in range(5)]
        data = numpy.random.normal(loc=100.0, size=(3, 5))
        matrix = DataMatrix(samples, features, data)

        assert matrix.list_samples() == list(reversed(samples))
        assert data is not matrix.get_data()
        assert numpy.array_equal(data[::-1], matrix.get_data())

    @pytest.fixture
    def samples(self, tmp_path: pathlib.Path):
        return [create_sample(tmp_path, k) for k in range(self.n_samples)]

    @pytest.fixture
    def features(self):
        return [create_feature_group(k) for k in range(self.n_features)]

    @pytest.fixture
    def matrix(self, samples, features):
        data = numpy.random.normal(loc=100.0, size=(self.n_samples, self.n_features))
        return DataMatrix(samples, features, data)

    def test_get_n_features(self, matrix):
        assert matrix.get_n_features() == self.n_features

    def test_get_n_samples(self, matrix):
        assert matrix.get_n_samples() == self.n_samples

    def test_list_samples(self, matrix, samples):
        assert samples == matrix.list_samples()

    def test_list_features(self, matrix, features):
        assert features == matrix.list_features()

    def test_get_columns_missing_group_raises_error(self, matrix: DataMatrix):
        with pytest.raises(exceptions.FeatureGroupNotFound):
            invalid_group = 100000
            matrix.get_columns(invalid_group)

    def test_get_columns_single_column(self, matrix: DataMatrix):
        expected_feature = matrix.list_features()[1]
        column = matrix.get_columns(expected_feature.group)[0]
        assert column.feature == expected_feature
        assert numpy.array_equal(column.data, matrix.get_data()[:, 1])

    def test_get_columns_multiple_columns(self, matrix: DataMatrix):
        index = [1, 3, 4]
        expected_features = matrix.list_features()
        query_groups = [expected_features[x].group for x in index]
        columns = matrix.get_columns(*query_groups)
        for ind, col in zip(index, columns):
            assert col.feature == expected_features[ind]
            assert numpy.array_equal(col.data, matrix.get_data()[:, ind])

    def test_get_columns_no_groups_retrieve_all_columns(self, matrix: DataMatrix):
        expected_features = matrix.list_features()
        columns = matrix.get_columns()
        for col, ft, col_data in zip(columns, expected_features, matrix.get_data().T):
            assert col.feature == ft
            assert numpy.array_equal(col.data, col_data)

    def test_get_samples_missing_sample_id_raises_error(self, matrix: DataMatrix):
        with pytest.raises(exceptions.SampleNotFound):
            invalid_sample_id = "invalid-sample-id"
            matrix.get_rows(invalid_sample_id)

    def test_get_samples_single_sample(self, matrix: DataMatrix):
        expected_sample = matrix.list_samples()[1]
        row = matrix.get_rows(expected_sample.id)[0]
        assert row.sample == expected_sample
        assert numpy.array_equal(row.data, matrix.get_data()[1])

    def test_get_samples_multiple_samples(self, matrix: DataMatrix):
        index = [1, 3, 4]
        expected_samples = matrix.list_samples()
        query_ids = [expected_samples[x].id for x in index]
        rows = matrix.get_rows(*query_ids)
        for ind, row in zip(index, rows):
            assert row.sample == expected_samples[ind]
            assert numpy.array_equal(row.data, matrix.get_data()[ind])

    def test_get_samples_no_ids_retrieves_all_samples(self, matrix: DataMatrix):
        expected_samples = matrix.list_samples()
        rows = matrix.get_rows()
        for row, sample, data_row in zip(rows, expected_samples, matrix.get_data()):
            assert row.sample == sample
            assert numpy.array_equal(row.data, data_row)

    def test_set_data(self, matrix: DataMatrix):
        n_samples = matrix.get_n_samples()
        n_features = matrix.get_n_features()
        expected_new_data = numpy.random.normal(scale=100.0, size=(n_samples, n_features))

        matrix.set_data(expected_new_data)
        assert numpy.array_equal(matrix.get_data(), expected_new_data)

    def test_set_data_invalid_array_raises_error(self, matrix: DataMatrix):
        n_samples = matrix.get_n_samples()
        n_features = matrix.get_n_features() + 1
        expected_new_data = numpy.random.normal(scale=100.0, size=(n_samples, n_features))
        with pytest.raises(ValueError):
            matrix.set_data(expected_new_data)

    def test_set_columns_no_columns_ok(self, matrix: DataMatrix):
        expected = matrix.get_data().copy()
        matrix.set_columns()
        assert numpy.array_equal(matrix.get_data(), expected)

    def test_set_columns_single_column(self, matrix: DataMatrix):
        expected = matrix.get_data().copy()
        group = 1
        column = (group, numpy.random.normal(size=matrix.get_n_samples()))
        idx = matrix.get_feature_index(group)[0]
        expected[:, idx] = column[1]
        matrix.set_columns(column)
        assert numpy.array_equal(matrix.get_data(), expected)

    def test_set_columns_invalid_group_raise_error(self, matrix: DataMatrix):
        column = (1000, numpy.random.normal(size=matrix.get_n_samples()))
        with pytest.raises(exceptions.FeatureGroupNotFound):
            matrix.set_columns(column)

    def test_set_columns_non_matching_size_raise_error(self, matrix: DataMatrix):
        column = (1, numpy.random.normal(size=matrix.get_n_samples() + 1))
        with pytest.raises(ValueError):
            matrix.set_columns(column)

    def test_set_columns_non_float_dtype_raise_error(self, matrix: DataMatrix):
        column = (1, numpy.random.normal(size=matrix.get_n_samples()).astype(int))
        with pytest.raises(TypeError):
            matrix.set_columns(column)

    def test_set_columns_multiple_columns(self, matrix: DataMatrix):
        expected = matrix.get_data().copy()
        columns = list()
        for k in [1, 5, 10]:
            col = (k, numpy.random.normal(size=matrix.get_n_samples()))
            columns.append(col)
            idx = matrix.get_feature_index(k)[0]
            expected[:, idx] = col[1]
        matrix.set_columns(*columns)
        assert numpy.array_equal(matrix.get_data(), expected)

    def test_set_rows_no_rows_ok(self, matrix: DataMatrix):
        expected = matrix.get_data().copy()
        matrix.set_rows()
        assert numpy.array_equal(matrix.get_data(), expected)

    def test_set_row_single_row(self, matrix: DataMatrix):
        expected = matrix.get_data().copy()
        sample_id = matrix.list_samples()[2].id
        row = (sample_id, numpy.random.normal(size=matrix.get_n_features()))
        idx = matrix.get_sample_index(sample_id)[0]
        expected[idx] = row[1]
        matrix.set_rows(row)
        assert numpy.array_equal(matrix.get_data(), expected)

    def test_set_rows_invalid_sample_id_raise_error(self, matrix: DataMatrix):
        row = ("invalid_id", numpy.random.normal(size=matrix.get_n_features()))
        with pytest.raises(exceptions.SampleNotFound):
            matrix.set_rows(row)

    def test_set_rows_non_matching_size_raise_error(self, matrix: DataMatrix):
        row = (matrix.list_samples()[0].id, numpy.random.normal(size=matrix.get_n_features() + 1))
        with pytest.raises(ValueError):
            matrix.set_rows(row)

    def test_set_rows_non_float_dtype_raise_error(self, matrix: DataMatrix):
        row = (matrix.list_samples()[0].id, numpy.random.normal(size=matrix.get_n_features()).astype(int))
        with pytest.raises(TypeError):
            matrix.set_rows(row)

    def test_set_rows_multiple_rows(self, matrix: DataMatrix):
        expected = matrix.get_data().copy()
        rows = list()
        for sample_id in [x.id for x in matrix.list_samples()[::2]]:
            row = (sample_id, numpy.random.normal(size=matrix.get_n_features()))
            rows.append(row)
            idx = matrix.get_sample_index(sample_id)[0]
            expected[idx] = row[1]
        matrix.set_rows(*rows)
        assert numpy.array_equal(matrix.get_data(), expected)

    def test_remove_features_remove_invalid_feature_raises_error(self, matrix: DataMatrix):
        with pytest.raises(exceptions.FeatureGroupNotFound):
            matrix.remove_features(100000)

    def test_remove_features_remove_all_features_raises_error(self, matrix: DataMatrix, features: list[FeatureGroup]):
        all_groups = [x.group for x in features]
        with pytest.raises(exceptions.EmptyDataMatrix):
            matrix.remove_features(*all_groups)

    def test_remove_features_no_features_does_nothing(self, matrix: DataMatrix, features: list[FeatureGroup]):
        matrix.remove_features()
        for ft in features:
            assert matrix.has_feature(ft.group)

    def test_remove_features_remove_single_feature(self, matrix: DataMatrix, features: list[FeatureGroup]):
        rm_group = features[3].group

        data_before_delete = matrix.get_data()
        keep_features_index = matrix.get_feature_index(*(x.group for x in features if x.group != rm_group))

        matrix.remove_features(rm_group)

        # check size
        assert (self.n_features - 1) == matrix.get_n_features()

        # check data order
        assert numpy.array_equal(data_before_delete[:, keep_features_index], matrix.get_data())

        # check that removed feature is no longer stored
        assert not matrix.has_feature(rm_group)

        # check feature order
        remaining_features = [x for x in features if x.group != rm_group]
        assert remaining_features == matrix.list_features()

    def test_remove_features_remove_multiple_features(self, matrix: DataMatrix, features: list[FeatureGroup]):
        rm_groups = [features[3].group, features[2].group]  # using unsorted order to check that it still works
        data_before_delete = matrix.get_data()
        keep_features_index = matrix.get_feature_index(*(x.group for x in features if x.group not in rm_groups))

        matrix.remove_features(*rm_groups)

        # check size
        assert (self.n_features - len(rm_groups)) == matrix.get_n_features()

        # check data order
        assert numpy.array_equal(data_before_delete[:, keep_features_index], matrix.get_data())

        # check that removed features are no longer stored
        for g in rm_groups:
            assert not matrix.has_feature(g)

        # check feature order
        remaining_features = [x for x in features if x.group not in rm_groups]
        assert remaining_features == matrix.list_features()

    def test_remove_samples_remove_invalid_sample_raises_error(self, matrix: DataMatrix):
        with pytest.raises(exceptions.SampleNotFound):
            matrix.remove_samples("invalid_sample_id")

    def test_remove_samples_remove_all_samples_raises_error(self, matrix: DataMatrix, samples: list[Sample]):
        all_sample_ids = [x.id for x in samples]
        with pytest.raises(exceptions.EmptyDataMatrix):
            matrix.remove_samples(*all_sample_ids)

    def test_remove_samples_no_samples_does_nothing(self, matrix: DataMatrix, samples: list[Sample]):
        matrix.remove_samples()
        for sample in samples:
            assert matrix.has_sample(sample.id)

    def test_remove_sample_remove_single_sample(self, matrix: DataMatrix, samples: list[Sample]):
        rm_id = samples[2].id

        data_before_delete = matrix.get_data()
        keep_samples_index = matrix.get_sample_index(*(x.id for x in samples if x.id != rm_id))

        matrix.remove_samples(rm_id)

        # check size
        assert (self.n_samples - 1) == matrix.get_n_samples()

        # check data order
        assert numpy.array_equal(data_before_delete[keep_samples_index], matrix.get_data())

        # check that removed sample is no longer stored
        assert not matrix.has_sample(rm_id)

        # check sample order
        remaining_samples = [x for x in samples if x.id != rm_id]
        assert remaining_samples == matrix.list_samples()

    def test_remove_samples_remove_multiple_samples(self, matrix: DataMatrix, samples: list[Sample]):
        rm_ids = [samples[3].id, samples[2].id]  # using unsorted order to check that it still works
        data_before_delete = matrix.get_data()
        keep_samples_index = matrix.get_sample_index(*(x.id for x in samples if x.id not in rm_ids))

        matrix.remove_samples(*rm_ids)

        # check size
        assert (self.n_samples - len(rm_ids)) == matrix.get_n_samples()

        # check data order
        assert numpy.array_equal(data_before_delete[keep_samples_index], matrix.get_data())

        # check that removed samples are no longer stored
        for id_ in rm_ids:
            assert not matrix.has_sample(id_)

        # check sample order
        remaining_samples = [x for x in samples if x.id not in rm_ids]
        assert remaining_samples == matrix.list_samples()

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_transform_features(self, matrix: DataMatrix, max_workers: int):
        assert not numpy.allclose(matrix.get_data(), 1.0)
        matrix.transform_features(ft_transformer, max_workers)
        assert numpy.allclose(matrix.get_data(), 1.0)

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_transform_samples(self, matrix: DataMatrix, max_workers: int):
        assert not numpy.allclose(matrix.get_data(), 1.0)
        matrix.transform_samples(sample_transformer, max_workers)
        assert numpy.allclose(matrix.get_data(), 1.0)

    def test_split_matrix_no_groups_yield_same_matrix(self, matrix: DataMatrix):
        groups = [x for x in matrix.split()]
        assert len(groups) == 1
        group_value, sub_matrix = groups[0]

        assert not group_value
        assert numpy.array_equal(sub_matrix.get_data(), matrix.get_data())
        assert sub_matrix.list_samples() == matrix.list_samples()
        assert sub_matrix.list_features() == matrix.list_features()

    def test_split_combine_no_groups_yield_same_matrix(self, matrix: DataMatrix):
        combined = DataMatrix.combine(*(x[1] for x in matrix.split()))
        assert numpy.array_equal(combined.get_data(), matrix.get_data())
        assert combined.list_samples() == matrix.list_samples()
        assert combined.list_features() == matrix.list_features()

    def test_split_combine_groupby_sample_group(self, matrix: DataMatrix):
        for sample in matrix.list_samples():
            if sample.meta.order % 2:
                sample.meta.group = "even"
            else:
                sample.meta.group = "odd"

        submatrices = list()
        for group, mat in matrix.split("group"):
            assert len(group) == 1
            assert group[0] in ["odd", "even"]
            assert all(x.meta.group == group[0] for x in mat.list_samples())
            submatrices.append(mat)

        combined = DataMatrix.combine(*submatrices)
        assert numpy.array_equal(combined.get_data(), matrix.get_data())
        assert combined.list_samples() == matrix.list_samples()
        assert combined.list_features() == matrix.list_features()

    def test_split_combine_groupby_sample_group_and_meta_field(self, matrix: DataMatrix):
        meta_field = "batch"
        for sample in matrix.list_samples():
            if sample.meta.order % 2:
                sample.meta.group = "even"
            else:
                sample.meta.group = "odd"

            if sample.meta.order % 3 == 1:
                sample.meta.batch = 1
            else:
                sample.meta.batch = 2

        submatrices = list()
        for group_val, mat in matrix.split("group", meta_field):
            sample_group, sample_batch = group_val
            assert sample_group in ["odd", "even"]
            assert sample_batch in [1, 2]
            assert all(x.meta.group == sample_group for x in mat.list_samples())
            assert all(x.meta.batch == sample_batch for x in mat.list_samples())
            submatrices.append(mat)

        combined = DataMatrix.combine(*submatrices)
        assert numpy.array_equal(combined.get_data(), matrix.get_data())
        assert combined.list_samples() == matrix.list_samples()
        assert combined.list_features() == matrix.list_features()

    def test_split_missing_meta_field_raises_error(self, matrix: DataMatrix):
        meta_field = "missing_field"
        for sample in matrix.list_samples():
            if sample.meta.order % 2:
                sample.meta.group = "even"
            else:
                sample.meta.group = "odd"

            if sample.meta.order % 3 == 1:
                sample.meta.missing_field = 1  # type: ignore

        with pytest.raises(exceptions.SampleMetadataNotFound):
            [x for x in matrix.split("group", meta_field)]

    def test_combine_no_matrices_raises_error(self):
        with pytest.raises(ValueError):
            DataMatrix.combine()
