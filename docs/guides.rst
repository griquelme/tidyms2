.. _user-guide:

User guide
==========

TidyMS2 is an MS data processing library that was created with two goals in mind:

1. provide an easy two use toolkit for processing MS data, allowing users to quickly create data pipelines for
   entire datasets.
2. provide an extensible framework for implementing new data processing tools, such as processing workflows or
   algorithms.

To achieve this, TidyMS2 offers a high level API for processing entire MS datasets. This API is built on top of a
core API that provides a variety of utilities to model and process MS data. In these introductory guides We will
use a bottom-up approach to present the most important components of the low level API and then we will present
the high level API for processing entire datasets.

TidyMS2 is built around the idea of a **data model** which define how entities in a MS data processing pipeline
should be represented. In this data model, only a subset of operations are allowed, ensuring a consistent
**data flow** throughout the data pipeline. In these guides, an intuitive introduction to the TidyMS2 data model
and data flow is provided. For the interested reader, the :ref:`architecture overview guide <overview>` describes
the data model in detail.

For users interested in extending TidyMS2, we recommend reading the :ref:`architecture overview guide <overview>`
after this tutorial followed by the :ref:`extending-guide` guide.

To get started, first install TidyMS2 using any of these methods:

.. include:: installation.rst

You can then continue by reading these topic-specific tutorials. Tutorials can be read in any order, but it
is recommended to read them in the order listed below:

.. toctree::
   :maxdepth: 1

   guides/raw-data
   guides/chemistry
   guides/sample
   guides/assay
   guides/lcms
   guides/simulation
   guides/mzml
   guides/matrix