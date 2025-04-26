import pathlib

import numpy
import pytest

from tidyms2.core import exceptions
from tidyms2.core.matrix import DataMatrix, FeatureVector, SampleVector, validate_data_matrix
from tidyms2.core.models import FeatureGroup, Sample

from ..helpers import create_feature_group, create_sample


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

    def test_create_matrix_without_samples_raises_error(self):
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

        assert matrix.samples == tuple(reversed(samples))
        assert data is not matrix.get_data()
        assert numpy.array_equal(data[::-1], matrix.get_data())

    @pytest.fixture
    def samples(self, tmp_path: pathlib.Path):
        return tuple(create_sample(tmp_path, k) for k in range(self.n_samples))

    @pytest.fixture
    def features(self):
        return tuple(create_feature_group(k) for k in range(self.n_features))

    @pytest.fixture
    def matrix(self, samples, features):
        data = numpy.random.normal(loc=100.0, size=(self.n_samples, self.n_features))
        return DataMatrix(samples, features, data)

    def test_get_n_features(self, matrix):
        assert matrix.get_n_features() == self.n_features

    def test_get_n_samples(self, matrix):
        assert matrix.get_n_samples() == self.n_samples

    def test_list_samples(self, matrix: DataMatrix, samples):
        assert samples == matrix.samples

    def test_list_features(self, matrix, features):
        assert features == matrix.features

    def test_get_columns_missing_group_raises_error(self, matrix: DataMatrix):
        with pytest.raises(exceptions.FeatureGroupNotFound):
            invalid_group = 100000
            matrix.get_columns(invalid_group)

    def test_get_columns_single_column(self, matrix: DataMatrix):
        expected_feature = matrix.features[1]
        column = matrix.get_columns(expected_feature.group)[0]
        assert column.feature == expected_feature
        assert numpy.array_equal(column.data, matrix.get_data()[:, 1])

    def test_get_columns_multiple_columns(self, matrix: DataMatrix):
        index = [1, 3, 4]
        expected_features = matrix.features
        query_groups = [expected_features[x].group for x in index]
        columns = matrix.get_columns(*query_groups)
        for ind, col in zip(index, columns):
            assert col.feature == expected_features[ind]
            assert numpy.array_equal(col.data, matrix.get_data()[:, ind])

    def test_get_columns_no_groups_retrieve_all_columns(self, matrix: DataMatrix):
        expected_features = matrix.features
        columns = matrix.get_columns()
        for col, ft, col_data in zip(columns, expected_features, matrix.get_data().T):
            assert col.feature == ft
            assert numpy.array_equal(col.data, col_data)

    def test_get_samples_missing_sample_id_raises_error(self, matrix: DataMatrix):
        with pytest.raises(exceptions.SampleNotFound):
            invalid_sample_id = "invalid-sample-id"
            matrix.get_rows(invalid_sample_id)

    def test_get_samples_single_sample(self, matrix: DataMatrix):
        expected_sample = matrix.samples[1]
        row = matrix.get_rows(expected_sample.id)[0]
        assert row.sample == expected_sample
        assert numpy.array_equal(row.data, matrix.get_data()[1])

    def test_get_samples_multiple_samples(self, matrix: DataMatrix):
        index = [1, 3, 4]
        expected_samples = matrix.samples
        query_ids = [expected_samples[x].id for x in index]
        rows = matrix.get_rows(*query_ids)
        for ind, row in zip(index, rows):
            assert row.sample == expected_samples[ind]
            assert numpy.array_equal(row.data, matrix.get_data()[ind])

    def test_get_samples_no_ids_retrieves_all_samples(self, matrix: DataMatrix):
        expected_samples = matrix.samples
        rows = matrix.get_rows()
        for row, sample, data_row in zip(rows, expected_samples, matrix.get_data()):
            assert row.sample == sample
            assert numpy.array_equal(row.data, data_row)

    def test_get_data(self, matrix: DataMatrix):
        data = matrix.get_data()
        expected_shape = (matrix.get_n_samples(), matrix.get_n_features())
        assert data.shape == expected_shape

    def test_get_data_sample_subset(self, matrix: DataMatrix):
        all_samples = matrix.samples
        sample_ids = [all_samples[1].id, all_samples[5].id, all_samples[7].id]
        data = matrix.get_data(sample_ids=sample_ids)
        expected_shape = (len(sample_ids), matrix.get_n_features())
        assert data.shape == expected_shape

        for data_row, row_vector in zip(data, matrix.get_rows(*sample_ids)):
            assert numpy.array_equal(data_row, row_vector.data)

    def test_get_data_missing_sample_id_raises_error(self, matrix: DataMatrix):
        sample_ids = ["invalid_sample_id"]
        with pytest.raises(exceptions.SampleNotFound):
            matrix.get_data(sample_ids=sample_ids)

    def test_get_data_feature_subset(self, matrix: DataMatrix):
        all_features = matrix.features
        feature_groups = [all_features[1].group, all_features[5].group, all_features[11].group]
        data = matrix.get_data(feature_groups=feature_groups)
        expected_shape = (matrix.get_n_samples(), len(feature_groups))
        assert data.shape == expected_shape

        for data_column, column_vector in zip(data.T, matrix.get_columns(*feature_groups)):
            assert numpy.array_equal(data_column, column_vector.data)

    def test_get_data_missing_feature_groups_raises_error(self, matrix: DataMatrix):
        feature_groups = [12314]
        with pytest.raises(exceptions.FeatureGroupNotFound):
            matrix.get_data(feature_groups=feature_groups)

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
        sample_id = matrix.samples[2].id
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
        row = (matrix.samples[0].id, numpy.random.normal(size=matrix.get_n_features() + 1))
        with pytest.raises(ValueError):
            matrix.set_rows(row)

    def test_set_rows_non_float_dtype_raise_error(self, matrix: DataMatrix):
        row = (matrix.samples[0].id, numpy.random.normal(size=matrix.get_n_features()).astype(int))
        with pytest.raises(TypeError):
            matrix.set_rows(row)

    def test_set_rows_multiple_rows(self, matrix: DataMatrix):
        expected = matrix.get_data().copy()
        rows = list()
        for sample_id in [x.id for x in matrix.samples[::2]]:
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
        assert remaining_features == matrix.features

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
        assert remaining_features == matrix.features

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
        assert remaining_samples == matrix.samples

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
        assert remaining_samples == matrix.samples

    def test_combine_single_matrix_return_same_matrix(self, matrix: DataMatrix):
        combined = DataMatrix.combine(matrix)
        assert numpy.array_equal(combined.get_data(), matrix.get_data())
        assert combined.samples == matrix.samples
        assert combined.features == matrix.features

    def test_combine_no_matrices_raises_error(self):
        with pytest.raises(ValueError):
            DataMatrix.combine()

    def test_combine_multiple_matrices(self, matrix: DataMatrix):
        all_samples = matrix.samples
        sample_ids1 = [x.id for x in all_samples[:3]]
        sample_ids2 = [x.id for x in all_samples[3:8]]
        sample_ids3 = [x.id for x in all_samples[8:]]
        sub1 = matrix.create_submatrix(sample_ids=sample_ids1)
        sub2 = matrix.create_submatrix(sample_ids=sample_ids2)
        sub3 = matrix.create_submatrix(sample_ids=sample_ids3)

        combined = DataMatrix.combine(sub1, sub2, sub3)

        assert numpy.array_equal(combined.get_data(), matrix.get_data())
        assert combined.samples == matrix.samples
        assert combined.features == matrix.features

    def test_create_submatrix_no_sample_ids_no_feature_groups_return_equal_matrix(self, matrix: DataMatrix):
        actual = matrix.create_submatrix()

        assert actual.features == matrix.features
        assert actual.samples == matrix.samples
        assert numpy.array_equal(actual.get_data(), matrix.get_data())

    def test_create_submatrix_with_sample_ids(self, matrix: DataMatrix):
        all_samples = matrix.samples
        sample_ids = [all_samples[1].id, all_samples[5].id, all_samples[7].id]
        actual = matrix.create_submatrix(sample_ids=sample_ids)

        sub_matrix_samples = actual.samples
        assert len(sub_matrix_samples) == len(sample_ids)
        assert all(actual.has_sample(x) for x in sample_ids)

        assert actual.features == matrix.features
        assert numpy.array_equal(actual.get_data(), matrix.get_data(sample_ids=sample_ids))

    def test_create_submatrix_with_feature_groups(self, matrix: DataMatrix):
        all_features = matrix.features
        feature_groups = [all_features[1].group, all_features[5].group, all_features[11].group]
        actual = matrix.create_submatrix(feature_groups=feature_groups)

        sub_matrix_feature_groups = actual.features
        assert len(sub_matrix_feature_groups) == len(feature_groups)
        assert all(actual.has_feature(x) for x in feature_groups)

        assert actual.samples == matrix.samples
        assert numpy.array_equal(actual.get_data(), matrix.get_data(feature_groups=feature_groups))

    def test_create_submatrix_with_feature_groups_and_sample_ids(self, matrix: DataMatrix):
        all_features = matrix.features
        feature_groups = [all_features[1].group, all_features[5].group, all_features[11].group]
        all_samples = matrix.samples
        sample_ids = [all_samples[1].id, all_samples[5].id, all_samples[7].id]
        actual = matrix.create_submatrix(sample_ids=sample_ids, feature_groups=feature_groups)

        sub_matrix_samples = actual.samples
        assert len(sub_matrix_samples) == len(sample_ids)
        assert all(actual.has_sample(x) for x in sample_ids)

        sub_matrix_feature_groups = actual.features
        assert len(sub_matrix_feature_groups) == len(feature_groups)
        assert all(actual.has_feature(x) for x in feature_groups)

        expected_data = matrix.get_data(feature_groups=feature_groups, sample_ids=sample_ids)

        assert numpy.array_equal(actual.get_data(), expected_data)

    def test_create_submatrix_with_missing_sample_ids_raise_error(self, matrix: DataMatrix):
        all_samples = matrix.samples
        sample_ids = [all_samples[1].id, all_samples[5].id, "invalid_sample_id"]
        with pytest.raises(exceptions.SampleNotFound):
            matrix.create_submatrix(sample_ids=sample_ids)

    def test_create_submatrix_with_missing_feature_group_raise_error(self, matrix: DataMatrix):
        all_features = matrix.features
        invalid_group = 1203
        feature_groups = [all_features[1].group, all_features[5].group, invalid_group]
        with pytest.raises(exceptions.FeatureGroupNotFound):
            matrix.create_submatrix(feature_groups=feature_groups)


class TestDataMatrixQuery:
    n_samples = 10
    n_features = 20

    @pytest.fixture
    def samples(self, tmp_path: pathlib.Path):
        res = list()
        for k in range(self.n_samples):
            group = "odd" if k % 2 else "even"
            batch = k // 10
            s = create_sample(tmp_path, k, order=k, group=group, batch=batch)
            res.append(s)
        return res

    @pytest.fixture
    def features(self):
        return [create_feature_group(k) for k in range(self.n_features)]

    @pytest.fixture
    def matrix(self, samples, features):
        data = numpy.random.normal(loc=100.0, size=(self.n_samples, self.n_features))
        return DataMatrix(samples, features, data)

    def test_fetch_samples_id_no_filter_return_all_samples(self, matrix: DataMatrix):
        _, sample_ids = matrix.query.fetch_sample_ids()[0]
        assert all(matrix.has_sample(x) for x in sample_ids)

    def test_fetch_samples_invalid_filter_field_raises_error(self, matrix: DataMatrix):
        with pytest.raises(exceptions.SampleMetadataNotFound):
            matrix.query.filter(invalid_meta="invalid_value").fetch_sample_ids()

    def test_fetch_samples_with_equality_filter(self, matrix: DataMatrix):
        select_group = "odd"
        _, sample_ids = matrix.query.filter(group=select_group).fetch_sample_ids()[0]
        assert sample_ids
        for id_ in sample_ids:
            sample = matrix.get_sample(id_)
            assert sample.meta.group == "odd"
            assert sample.meta.batch == 0 or sample.meta.batch == 1
        assert all(matrix.has_sample(x) for x in sample_ids)

    def test_fetch_samples_with_in_filter(self, matrix: DataMatrix):
        select_order = [3, 4, 5]
        _, sample_ids = matrix.query.filter(order=select_order).fetch_sample_ids()[0]
        assert len(sample_ids) == len(select_order)
        for id_ in sample_ids:
            sample = matrix.get_sample(id_)
            assert sample.meta.order in select_order
        assert all(matrix.has_sample(x) for x in sample_ids)

    def test_fetch_samples_with_multiple_filters(self, matrix: DataMatrix):
        select_order = [3, 4, 5]
        select_group = "odd"
        _, sample_ids = matrix.query.filter(order=select_order, group=select_group).fetch_sample_ids()[0]
        assert len(sample_ids) == 2
        for id_ in sample_ids:
            sample = matrix.get_sample(id_)
            assert sample.meta.order == 3 or sample.meta.order == 5
            assert sample.meta.group == select_group

    def test_fetch_samples_invalid_group_by_field_raises_error(self, matrix: DataMatrix):
        with pytest.raises(exceptions.SampleMetadataNotFound):
            matrix.query.group_by("invalid_meta").fetch_sample_ids()

    def test_sample_with_group_by_no_groups_return_all_samples_in_an_empty_group(self, matrix: DataMatrix):
        group, sample_ids = matrix.query.group_by().fetch_sample_ids()[0]
        assert not group
        assert len(sample_ids) == matrix.get_n_samples()
        assert all(matrix.has_sample(x) for x in sample_ids)

    def test_sample_with_group_by_single_field(self, matrix: DataMatrix):
        actual = matrix.query.group_by("group").fetch_sample_ids()
        expected_groups = ["odd", "even"]
        for group, sample_ids in actual:
            assert len(group) == 1
            assert group[0] in expected_groups
            for sample_id in sample_ids:
                sample = matrix.get_sample(sample_id)
                assert sample.meta.group == group[0]

    def test_sample_with_group_by_multiple_fields(self, matrix: DataMatrix):
        actual = matrix.query.group_by("group", "batch").fetch_sample_ids()
        expected_groups = ["odd", "even"]
        expected_batches = [0, 1]
        for group, sample_ids in actual:
            group_group, group_batch = group
            assert len(group) == 2
            assert group_group in expected_groups
            assert group_batch in expected_batches
            for sample_id in sample_ids:
                sample = matrix.get_sample(sample_id)
                assert sample.meta.group == group_group
                assert sample.meta.batch == group_batch

    def test_sample_with_filter_and_group_by(self, matrix: DataMatrix):
        actual = matrix.query.filter(batch=1).group_by("group", "batch").fetch_sample_ids()
        expected_groups = ["odd", "even"]
        expected_batch = 1
        for group, sample_ids in actual:
            group_group, group_batch = group
            assert len(group) == 2
            assert group_group in expected_groups
            assert group_batch == expected_batch
            for sample_id in sample_ids:
                sample = matrix.get_sample(sample_id)
                assert sample.meta.group == group_group
                assert sample.meta.batch == expected_batch


class TestDataMatrixMetrics:
    n_samples = 10
    n_features = 20

    @pytest.fixture
    def samples(self, tmp_path: pathlib.Path):
        res = list()
        for k in range(self.n_samples):
            group = "odd" if k % 2 else "even"
            batch = k // 10
            s = create_sample(tmp_path, k, order=k, group=group, batch=batch)
            s.meta.type = "qc" if k < 5 else "sample"
            res.append(s)
        return res

    @pytest.fixture
    def features(self):
        return [create_feature_group(k) for k in range(self.n_features)]

    @pytest.fixture
    def matrix(self, samples, features):
        data = numpy.random.normal(loc=100.0, size=(self.n_samples, self.n_features))
        return DataMatrix(samples, features, data)

    def test_cv(self, matrix: DataMatrix):
        cv = matrix.metrics.cv()
        assert cv.size == matrix.get_n_features()

    def test_detection_rate(self, matrix: DataMatrix):
        dr = matrix.metrics.detection_rate()
        assert dr.size == matrix.get_n_features()

    def test_dratio(self, matrix: DataMatrix):
        dratio = matrix.metrics.dratio()
        assert dratio.size == matrix.get_n_features()

    def test_dratio_with_groups(self, matrix: DataMatrix):
        dratio = matrix.metrics.dratio(sample_groups=["odd"], qc_groups=["even"])
        assert dratio.size == matrix.get_n_features()

    def test_dratio_no_sample_group_raise_error(self, matrix: DataMatrix):
        with pytest.raises(ValueError):
            matrix.metrics.dratio(sample_groups=["invalid_group"])

    def test_dratio_no_qc_group_raise_error(self, matrix: DataMatrix):
        with pytest.raises(ValueError):
            matrix.metrics.dratio(qc_groups=["invalid_group"])

    def test_pca(self, matrix: DataMatrix):
        scores, loadings, variance = matrix.metrics.pca()
        assert scores.shape[0] == matrix.get_n_samples()
        assert loadings is None
        assert variance is None

    def test_pca_matrix_with_missing_values_raise_error(self, samples: list[Sample], features: list[FeatureGroup]):
        data = numpy.random.normal(size=(len(samples), len(features)))
        data[0, 2] = numpy.nan
        matrix = DataMatrix(samples, features, data)
        with pytest.raises(ValueError):
            matrix.metrics.pca()

    def test_correlation(self, matrix: DataMatrix):
        corr = matrix.metrics.correlation("order")
        assert corr.size == matrix.get_n_features()

    def test_correlation_non_numeric_field_raises_error(self, matrix: DataMatrix):
        with pytest.raises(ValueError):
            matrix.metrics.correlation("group")
