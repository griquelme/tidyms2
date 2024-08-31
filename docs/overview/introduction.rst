.. _introduction-overview:

Introduction
------------

In broad terms, the goal of mass spectrometry data preprocessing is, starting from raw data measurements, to extract a 
property associated with the abundance a chemical species in a sample. Usually, this property is compared across
multiple sample measured under similar experimental conditions. We refer to such collection of samples as an
:term:`assay`. Data from multiple chemical species are organized into a table or data matrix where each entry value
is the abundance associated with each sample, chemical species pair. We refer to all of the steps involved, as MS
data preprocessing.

We use an :term:`LC-MS` assay as an example of the preprocessing steps allow to build the data matrix from raw
data. This will help us identity some key concepts in MS data preprocessing. The first step in :term:`LC-MS` data
preprocessing is to create :term:`EIC` from each sample raw data. A peak detection algorithm is then applied to each
:term:`EIC` and each peak is, in principle, a single chemical species (this is not strictly true due to potential
coellution of isobars, or ionic species with similar :term:`m/z`). After chemical species are detected on all samples,
they must be grouped according to their chemical identity. Chromatographic peak alignment algorithms are commonly
used to this end. Once peaks are aligned, we can proceed to the creation of the data matrix using, for example, the
peak area as the value for each data matrix entry.

From this example we can extract some insights. First, there is a huge number of algorithms available in the literature
for creating :term:`EIC`, performing peak detection and peak alignment. There are also different ways in which
the peak area can be computed. What is more, this analysis is restricted to data processing algorithms for :term:`LC-MS`
only, and different strategies may apply for other analytical platforms such as Ambient MS, Direct infusion MS, or
Ion Mobility MS. It is clear that a flexible framework for processing mass spectrometry data should be able support
all of these multiple analytical platforms, algorithms, and conventions for computing properties from chemical species.

There are also computational considerations: as the MS instrument resolution and sensitivity increases, datasets become
larger in disk size. This results in increment of the data size of intermediary data processing data. Managing this data
in a reliable and efficient manner should be abstracted away from, for example, a developer working on a new peak detection
algorithm.

TidyMS was designed from its core with all of these considerations in mind to provide fast, reliable and reproducible
tools for MS data preprocessing workflows. These tools follow the TidyMS data model and data flow. The TidyMS data model
define what entities are created across a preprocessing workflow and how they are related. On the other hand, the data
flow defines what transformations are allowed throughout a preprocessing workflow, ensuring data consistency on all stages.
Under the hood, the TidyMS backend orchestrates the execution of data processing workflows in an efficient manner.

Data preprocessing framework
----------------------------

Data preprocessing in TidyMS is expressed in terms of workflows. A workflow is a specification of all the parameters of
each preprocessing step that allow to create a data matrix from raw experimental measurements. A workflow has three stages,
as described in the following diagram:

Sample operations use data from individual samples, e.g. creating chromatograms from a sample. Assay operations use data
from all samples, e.g., chromatographic peak alignment. Data matrix operations apply transformation on the data matrix
rows or columns. We now describe data models and the operations allowed by the data flow in each one one of these stages.