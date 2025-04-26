import numpy
import pytest

from tidyms2.core.dataflow import DataMatrixProcessStatus
from tidyms2.core.enums import MSInstrument, Polarity, SeparationMode
from tidyms2.core.exceptions import EmptyDataMatrix
from tidyms2.core.matrix import DataMatrix, FeatureVector, SampleVector
from tidyms2.core.operators.matrix import ColumnFilter, ColumnTransformer, MatrixTransformer, RowFilter, RowTransformer
from tidyms2.core.utils.numpy import FloatArray

from ...helpers import create_data_matrix

N_SAMPLES = 10
N_FEATURES = 20


@pytest.fixture
def matrix(tmp_path):
    return create_data_matrix(N_SAMPLES, N_FEATURES, tmp_path)


class DummyMatrixTransformer(MatrixTransformer):
    value: float = 10.0
    """Set all values in the data matrix to this value"""

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def _transform_matrix(self, data: DataMatrix) -> FloatArray:
        transformed = data.get_data().copy()
        transformed[:] = self.value
        return transformed

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity):
        return cls()


class DummyRowFilter(RowFilter):
    max_order: int = 5
    """Remove samples if their order is larger than this value"""

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def _create_remove_list(self, data: DataMatrix) -> list[str]:
        return [x.id for x in data.samples if x.meta.order > self.max_order]

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity):
        return cls()


class DummyColumnFilter(ColumnFilter):
    max_group: int = 10
    """Remove features if their group is larger than this value"""

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def _create_remove_list(self, data: DataMatrix) -> list[int]:
        return [x.group for x in data.features if x.group > self.max_group]

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity):
        return cls()


class DummyColumnTransformer(ColumnTransformer):
    odd_value: float = 1.0
    """Value to set on columns with odd group values"""

    even_value: float = 2.0
    """Value to set on columns with even group values"""

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def _transform_column(self, column: FeatureVector) -> FeatureVector:
        value = self.odd_value if column.feature.group % 2 else self.even_value
        size = column.data.size
        return FeatureVector(data=numpy.ones(size, dtype=float) * value, feature=column.feature, index=0)

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity):
        return cls()


class DummyRowTransformer(RowTransformer):
    odd_value: float = 1.0
    """Value to set on rows with odd order values"""

    even_value: float = 2.0
    """Value to set on rows with even order values"""

    def get_expected_status_in(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def get_expected_status_out(self) -> DataMatrixProcessStatus:
        return DataMatrixProcessStatus()

    def _transform_row(self, row: SampleVector) -> SampleVector:
        value = self.odd_value if row.sample.meta.order % 2 else self.even_value
        size = row.data.size
        return SampleVector(data=numpy.ones(size, dtype=float) * value, sample=row.sample, index=0)

    @classmethod
    def from_defaults(cls, instrument: MSInstrument, separation: SeparationMode, polarity: Polarity):
        return cls()


def test_matrix_transformer(matrix: DataMatrix):
    expected = 100.0
    op = DummyMatrixTransformer(value=expected)
    op.apply(matrix)

    assert numpy.allclose(matrix.get_data(), expected)


def test_row_filter(matrix: DataMatrix):
    max_order = 5
    removed = {x.id for x in matrix.samples if x.meta.order > max_order}
    op = DummyRowFilter(max_order=max_order)

    assert all(matrix.has_sample(x) for x in removed)

    op.apply(matrix)

    assert not any(matrix.has_sample(x) for x in removed)


def test_row_remove_none_ok(matrix: DataMatrix):
    max_order = 10000
    all_samples = {x.id for x in matrix.samples}
    op = DummyRowFilter(max_order=max_order)

    assert all(matrix.has_sample(x) for x in all_samples)

    op.apply(matrix)

    assert all(matrix.has_sample(x) for x in all_samples)


def test_row_filter_remove_all_raises_error(matrix: DataMatrix):
    max_order = -1
    op = DummyRowFilter(max_order=max_order)

    with pytest.raises(EmptyDataMatrix):
        op.apply(matrix)


def test_row_filter_with_exclude_list(matrix: DataMatrix):
    max_order = 5
    exclude = [x.id for x in matrix.samples if x.meta.order > 8]
    removed = {x.id for x in matrix.samples if x.meta.order > max_order}.difference(exclude)
    op = DummyRowFilter(max_order=max_order, exclude=exclude)

    assert all(matrix.has_sample(x) for x in removed)
    assert all(matrix.has_sample(x) for x in exclude)

    op.apply(matrix)

    assert not any(matrix.has_sample(x) for x in removed)
    assert all(matrix.has_sample(x) for x in exclude)


def test_column_filter(matrix: DataMatrix):
    max_group = 5
    removed = {x.group for x in matrix.features if x.group > max_group}
    op = DummyColumnFilter(max_group=max_group)

    assert all(matrix.has_feature(x) for x in removed)

    op.apply(matrix)

    assert not any(matrix.has_feature(x) for x in removed)


def test_column_filter_with_exclude_list(matrix: DataMatrix):
    max_group = 5
    exclude = [x.group for x in matrix.features if x.group > 8]
    removed = {x.group for x in matrix.features if x.group > max_group}.difference(exclude)
    op = DummyColumnFilter(max_group=max_group, exclude=exclude)

    assert all(matrix.has_feature(x) for x in removed)
    assert all(matrix.has_feature(x) for x in exclude)

    op.apply(matrix)

    assert not any(matrix.has_feature(x) for x in removed)
    assert all(matrix.has_feature(x) for x in exclude)


def test_column_filter_remove_none_ok(matrix: DataMatrix):
    max_group = 10000
    all_features = {x.group for x in matrix.features}
    op = DummyRowFilter(max_order=max_group)

    assert all(matrix.has_feature(x) for x in all_features)

    op.apply(matrix)

    assert all(matrix.has_feature(x) for x in all_features)


def test_column_filter_remove_all_raises_error(matrix: DataMatrix):
    max_group = -1
    op = DummyColumnFilter(max_group=max_group)

    with pytest.raises(EmptyDataMatrix):
        op.apply(matrix)


@pytest.mark.parametrize("max_workers", [1, 2])
def test_column_transformer(matrix: DataMatrix, max_workers):
    op = DummyColumnTransformer(max_workers=max_workers)
    op.apply(matrix)

    assert matrix.get_n_features()
    for col in matrix.get_columns():
        if col.feature.group % 2:
            assert numpy.allclose(col.data, op.odd_value)
        else:
            assert numpy.allclose(col.data, op.even_value)


@pytest.mark.parametrize("max_workers", [1, 2])
def test_column_transformer_with_exclude_list(matrix: DataMatrix, max_workers):
    exclude_groups = [x.group for x in matrix.features if x.group < 5]
    op = DummyColumnTransformer(max_workers=max_workers, exclude=exclude_groups)

    excluded_columns = {x.feature.group: x for x in matrix.get_columns(*exclude_groups)}
    op.apply(matrix)

    assert matrix.get_n_features()
    for col in matrix.get_columns():
        if col.feature.group in exclude_groups:
            col_before = excluded_columns[col.feature.group]
            assert numpy.array_equal(col.data, col_before.data)
        elif col.feature.group % 2:
            assert numpy.allclose(col.data, op.odd_value)
        else:
            assert numpy.allclose(col.data, op.even_value)


@pytest.mark.parametrize("max_workers", [1, 2])
def test_row_transformer(matrix: DataMatrix, max_workers):
    op = DummyRowTransformer(max_workers=max_workers)
    op.apply(matrix)

    for row in matrix.get_rows():
        if row.sample.meta.order % 2:
            assert numpy.allclose(row.data, op.odd_value)
        else:
            assert numpy.allclose(row.data, op.even_value)


@pytest.mark.parametrize("max_workers", [1, 2])
def test_row_transformer_with_exclude_list(matrix: DataMatrix, max_workers):
    exclude_ids = [x.id for x in matrix.samples if x.meta.order < 5]
    op = DummyRowTransformer(max_workers=max_workers, exclude=exclude_ids)

    excluded_rows = {x.sample.id: x for x in matrix.get_rows(*exclude_ids)}
    op.apply(matrix)

    for row in matrix.get_rows():
        if row.sample.id in exclude_ids:
            col_before = excluded_rows[row.sample.id]
            assert numpy.array_equal(row.data, col_before.data)
        elif row.sample.meta.order % 2:
            assert numpy.allclose(row.data, op.odd_value)
        else:
            assert numpy.allclose(row.data, op.even_value)
