.. _assay-guide:

Processing multiple samples
===========================

.. note::
   This guide describes the general process for assay creation which involves
   creating each component needed by the assay class for data processing.
   This process is required for users that want to customize an assay or are
   interested in more advanced workflows. Most users can use instead helper
   functions defined for different processing workflows which allow to create
   an assay in an easy way. For example, for LC-MS data, the
   :py:func:`~tidyms2.lcms.create_lcms_assay` allows to create a preconfigured
   assay for LC-MS data. See the :ref:`processing-lcms-datasets` guide for more
   details.

We describe here how to process multiple samples. First, we will describe how to apply
sample operators to multiple samples and then how to apply assay operators that use
information from multiple samples.

The assay class
---------------

Before discussing sample processing, we need to present the
:py:class:`~tidyms2.core.assay.Assay` class, which manages complete preprocessing
workflows. The assay class is a composite class that orchestrates multiple sample
processing and storage. To create a new assay, several components needs to be
