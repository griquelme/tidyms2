TidyMS2
=======

Tools to process mass spectrometry data.

TidyMS2 is an upgraded version from [TidyMS](https://github.com/griquelme/tidyms), built from scratch and designed
to work with Python latest practices.

TidyMS offers a highly customizable framework for processing mass spectrometry datasets:

**For processing MS datasets:**

- tools for processing LC-MS datasets: build pipelines to define how to process your data and TidyMS manages the rest.
- utilities for working with chemical entities: formulas, formulas generators and formula scorers.
- test your pipelines with dataset simulation utilities
- data persistence for intermediate data and results 
- result visualization

**For creating new processing algorithms**

- a highly extensive data model for expressing features
- a data flow model that validates and manages data through your data pipeline. 

Installation
------------

TidyMS is installed using pip:

```shell
pip install tidyms2
```

Documentation
-------------

The library documentation is available [here](https://tidyms.readthedocs.io/en/latest/).

Getting help
------------

The library documentation contains tutorials on a variety on topics. If you weren't able to find an answer
to your problem, you can use the project [discussion board] (https://github.com/griquelme/tidyms2/discussions)

Development
-----------

If you encounter a problem or bug, you can report it using the [issue tracker](https://github.com/griquelme/tidyms2/issues).

Before submitting a new issue, please search the issue tracker to see if the problem has already been reported.

If your question is about how to achieve a specific task or use the library in a certain way, we recommend
posting it in the [discussions](https://github.com/griquelme/tidyms2/discussions) section.

When reporting an issue, it's helpful to include the following details:

- A code snippet that reproduces the problem.
- If an error occurs, please include the full traceback.
- A brief explanation of why the current behavior is incorrect or unexpected.

For guidelines on how to write an issue report, refer to this [post](https://matthewrocklin.com/minimal-bug-reports).


Contributing
------------

Checkout the developer and contributing guides in the library [documentation](https://tidyms.readthedocs.io/en/latest/)

Citation
--------

If you find TidyMS useful, we would appreciate citations:

Riquelme, G.; Zabalegui, N.; Marchi, P.; Jones, C.M.; Monge, M.E. A Python-Based Pipeline for Preprocessing
LC–MS Data for Untargeted Metabolomics Workflows. _Metabolites_ **2020**, 10, 416, doi:10.3390/metabo10100416.