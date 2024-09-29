.. _assay-overview:

Assay workflow
==============

The assay preprocessing pipeline is applied to the assay after all samples were processed by the sample
pipeline. The assay data flow enforces one main restriction in the assay workflow: with the exception of
:py:class:`~tidyms2.core.models.Annotation`, all other sample related entities (Roi and Feature) are
read only. That means, that only feature annotations may be modified during assay preprocessing. Three
operations are allowed at the assay level: patching feature descriptors, patching feature annotations
and imputing missing values in the to-be-created data matrix. The operators that implement these
operations are: :py:class:`~tidyms2.core.operators.DescriptorPatcher`,
:py:class:`~tidyms2.core.operators.AnnotationPatcher` and :py:class:`~tidyms2.core.operators.MissingImputer`.

Three data entities are associated with each one of this operators: :py:class:`~tidyms2.core.models.DescriptorPatch`,
:py:class:`~tidyms2.core.models.AnnotationPatch` and :py:class:`~tidyms2.core.models.FillValue`.

At first sight this restriction may seem excessive, but it will make sense once we need to consider that the main goal
of the assay workflow, which is to creation of the data matrix. A data matrix is an :math:`n \times m` matrix where
:math:`n` is the number of samples in the assay and :math:`m` is the number of :term:`feature groups <feature group>`
in the assay. A feature group is a collection of features from different samples that are originated from the same
chemical species. A :py:class:`~tidyms2.core.models.FeatureGroup` contains aggregated descriptors from all the
features in the group and a consensus annotation using information from all features. Features with the same
:py:attr:`~tidyms2.core.models.Annotation.group` belong to the same feature group. The group annotation is computed
by using a feature matching algorithm, which in terms of the data is an annotation patcher, because it patches the
annotation of all features. This is operations is required to create a data matrix as it defines the feature groups. The
:ref:`feature-groups` section for a detailed description on how feature groups are computed.

Once all features have a group assigned, the creation of the data matrix is straightforward: a feature descriptor is
assigned as the matrix value and is set in the corresponding data matrix entry using the sample id, group id pair of
the feature. It is important to note here that multiple feature from the same sample may share the feature group.
In this case, the matrix entry contains the additive contribution from each feature. Sharing a feature group in multiple
samples is allowed in order to consider the case in which features may need to be merged into a single feature.

After we presenting how the data matrix is created, we can resume the limitations imposed by the assay data flow.
As the data matrix is created using information from features descriptors and annotations, it does not make any sense
to update ROI or Feature data. This does not forbid using ROI data, feature data or even raw sample data for performing
assay operations, as long as the end result is the modification of either feature descriptors or annotations. We mentioned
feature matching as one example of annotation patcher. An example of descriptor patching is a peak alignment algorithm,
which computes corrected retention time for each chromatographic peak.

After feature groups are computed, it will often happen that in some samples there are not features belonging to a given
feature group. This will result in a missing entry in the data matrix. The missing imputer operation search a matrix
value for those missing entries.

TidyMS provides a built-in consensus annotator which takes annotations from individual features an compute annotations
for all feature groups.

.. _feature-groups:

Computing feature groups
------------------------