.. _sample-overview:

Sample pipeline
===============

The sample pipeline applies operations on data from individual samples. Data generated in the sample workflow is
stored using a :py:class:`~tidyms2.core.storage.SampleStorage` which provides an interface for reading and writing
sample data.  The preprocessing state of a sample is represented using the
:py:class:`~tidyms2.core.dataflow.SampleProcessStatus`.

.. _sample-data-model:

Sample data model
-----------------

Four entities model the data used during the sample preprocessing stage: :py:class:`~tidyms2.core.models.Sample`,
:py:class:`~tidyms2.core.models.Roi`, :py:class:`~tidyms2.core.models.Feature` and
:py:class:`~tidyms2.core.models.Annotation`. The following diagram showcases the relationships between these
entities:

.. figure:: assets/sample-models.png
        
    Sample data level models

A sample stores metadata from an individual measurement. It contains the required information to access the
sample raw data including path to the raw data file, MS level, start acquisition time and end acquisition time.
It also contains information associated with the assay design, such as sample group, sample order and sample batch.
The sample model can also store arbitrary sample metadata, as long as it is JSON serializable. A :term:`Region
Of Interest <ROI>` stores a region extracted from a sample raw data which may contain information about chemical
species of interest. There is a many-to-one relationship between ROIs and a sample. In an :term:`LC-MS` assay an
:term:`EIC` would be modelled as a ROI. A :term:`feature` defines where a chemical species is located in a ROI.
The term feature is borrowed from the machine learning field, and is used to highlight the fact that, even if a feature
is associated with a single chemical species, its chemical identity is not known. There is a many-to-one relationship
between features and a ROI. In an :term:`LC-MS` assay a chromatographic peak would be modelled as a Feature. A Feature
also defines how their properties, such as chromatographic peak width, are computed. We refer to these properties that
describe a feature as :term:`feature descriptors <feature descriptor>`. An annotation maps a feature to the ROI and
Sample that is associated with. It also provides chemical identity information such as its :term:`feature group`, ionic
charge and isotopologue annotation. There is a one-to-one relationship between a feature and an annotation.

Both ROI an Feature are abstract classes that must be implemented for each assay type (LC-MS, direct injection MS, ...).
The :ref:`extending-guide` guide contains detailed information on how to create or customize new ROI and Feature models.

.. _sample-data-flow:

Sample data flow
----------------

The sample data flow, which ensures data integrity during sample preprocessing, is enforced by defining a set of allowed
operations on sample data. Each operation is associated with an operator. The following operators are
available for sample data preprocessing: :py:class:`~tidyms2.core.operators.sample.RoiExtractor`,
:py:class:`~tidyms2.core.operators.sample.RoiTransformer`, :py:class:`~tidyms2.core.operators.sample.FeatureExtractor`,
:py:class:`~tidyms2.core.operators.sample.FeatureTransformer` and :py:class:`~tidyms2.core.operators.sample.SampleOperator`

The ROI extractor create ROIs using raw data. The ROI transformer apply transformations to existing ROIs.
A feature extractor search features on individual ROIs. The feature extractor also provides descriptor-based
filtering, allowing to keep or filter features based on descriptor values such as peak width or :term:`SNR`. A feature
transformer apply transformations to individual features. Finally, the sample operator applies an arbitrary operation
using all available data from a single sample. An example of a sample operator is the built-in isotopologue annotator,
which labels features as isotopologues and sets its charge state by updating each feature annotation.

All of these operators must be implemented for each ROI-feature pair. Refer to the :ref:`extending-guide`
for detailed information on how to create new operators.

Each one of these operators require a given sample data state to be applied to sample data. Otherwise, they will
fail. The sample data state is defined by a :py:class:`~tidyms2.core.dataflow.SampleProcessStatus`. The following
table contains the required sample data state by each operator:

.. list-table:: Sample data state required by each operator
   :widths: 25 25 25 25
   :header-rows: 1

   * - Operator
     - ROI extracted
     - Feature extracted
     - Isotopologue annotated
   * - RoiExtractor
     - No
     - No
     - No
   * - RoiTransformer
     - Yes
     - No
     - No
   * - FeatureExtractor
     - Yes
     - No
     - No
   * - FeatureTransformer
     - Yes
     - Yes
     - No
   * - IsotopologueAnnotator
     - Yes
     - Yes
     - No
   * - SampleOperator
     - 
     - 
     - 

After applying an operator to a sample, the sample state is updated to reflect the applied transformation.
Multiple operators are chained together to build the sample preprocessing
:py:class:`~tidyms2.core.operators.Pipeline`. A pipeline ensures that the data flow is correct before applying
any operation by checking the expected state of consecutive operators. For example, as feature extraction can be
applied only after applying ROI extraction, a pipeline can perform a fast sanity check on the operation order.

One important remark about sample pipelines, is that even if any number of operators can be used together,
a sample pipeline that is part of a preprocessing workflow needs to perform at the very least ROI extraction
and feature extraction operations, as they are required for performing assay-level operations, which will be
described in the next section.

The following diagram shows an example of how operators are organized into a sample preprocessing pipeline for
LC-MS data:

.. figure:: assets/lcms-sample-workflow.png
        
    Example of an LC-MS sample preprocessing pipeline