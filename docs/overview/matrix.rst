.. _matrix-overview:

Matrix pipeline
===============

The data matrix pipeline address the data matrix curation process, which consists in removing
unwanted sources of variation in the data and filtering features or samples that do not
meet the minimal analytical robustness requirements to be included in the data analysis stage.

The :py:class:`~tidyms2.core.matrix.DataMatrix` model stores its data using a 2D Numpy array. Each
row in the matrix is associated with a :py:class:`~tidyms2.core.models.Sample` model. Each column in
the matrix is associated with a :py:class:`~tidyms2.core.models.FeatureGroup`. The matrix state is
defined by the :py:class:`~tidyms2.core.dataflow.DataMatrixProcessStatus`.

The data matrix data flow is the simplest to understand, as the only requirements are maintaining
consistent shape and values in the data matrix. Regarding the shape, empty matrices are not allowed.
A matrix must have at least one sample and one feature. Matrix transformations that result in an
empty matrix are considered an error. Regarding matrix values, it is required that all values
are non-negative. Missing values are accepted, but operators may require that the matrix do
not contain any NaN value. Three operations are included for applying transformations on
a data matrix: data transformation, feature filtering and sample filtering:

data transformation
    includes any transformation that does not alter the matrix shape. The
    :py:class:`~tidyms2.core.operators.matrix.MatrixTransformer` allows performing an
    arbitrary transformation on data. The
    :py:class:`~tidyms2.core.operators.matrix.RowTransformer` and 
    :py:class:`~tidyms2.core.operators.matrix.ColumnTransformer` operators provide an simpler
    interface for row or column-based transformations. These operators are used to
    implement common matrix operation as normalization, blank correction or analytical
    batch effects correction.
sample filtering
    Remove a set of selected samples. The :py:class:`~tidyms2.core.operators.matrix.RowFilter`
    allows to remove samples by defining a filtering function. e.g. removing samples that do
    not match the assay sample template or remove samples from a specific group.
feature filtering
    Remove a set of selected features. The :py:class:`~tidyms2.core.operators.matrix.ColumnFilter`
    allows to remove features by defining a filtering function. e.g. removing features with low
    detection rate or high coefficient of variation.
