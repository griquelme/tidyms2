.. _extending-guide:

Extending TidyMS2
=================

The :py:class:`~tidyms2.storage.memory.OnMemorySampleStorage` class implements the sample storage
protocol and is suited for most sample preprocessing workflows.

Implementation
--------------

TidyMS is built on top of some of the more battle tested Python libraries:

pydantic
    All TidyMS data models are defined as pydantic models, ensuring robust data validations along
    data preprocessing pipelines.
numpy
    raw spectral data and ROIs
scipy
    All numerical algorithms
scikit-learn
    Learning algorithms