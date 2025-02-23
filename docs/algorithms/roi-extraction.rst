.. _algorithms-roi-extraction:

`m/z` trace extraction in LC-MS
===============================

This guide describes theoretical background of :term:`ROI` extraction algorithm for LC-MS datasets,
implemented by the :py:class:`~tidyms2.lcms.LCTraceExtractor` operator.

A ROI is region extracted from raw sample data that may contain features. In LC-MS datasets, a ROI
is an :term:`m/z` trace. m/z traces are similar to chromatograms but with two differences:
information related to the m/z value used in each scan is included and the traces are defined in a
time window where m/z values were detected.

..  plot:: plots/roi_definition.py
    :caption: A m/z trace is comprises by three arrays storing m/z, time and intensity.

In TidyMS2, ROI extraction is done by using an approach similar to the one described by Tautenhahn
*et al* in [1]_, but with some modifications. m/z traces are created and extended connecting close
m/z values across successive scans using the following method:

1.  The m/z values in The first scan are used to initialize a list of ROI. If ``targeted_mz`` is
    used, the ROI are initialized using this list.
2.  m/z values from the next scan extend the ROIs if they are closer than ``tolerance`` to the
    mean m/z of the ROI. Values that don't match any ROI are used to create new ROIs and are
    appended to the ROI list. If ``targeted_mz`` is used, these values are discarded.
3.  If more than one m/z value is within the tolerance threshold, m/z and intensity values are
    computed according to the ``multiple_match`` strategy. Two strategies are available: merge
    multiple peaks into an average peak or use only the closest peak to extend the ROI and
    create new ROIs with the others.
4.  If a ROI can't be extended with any m/z value from the new scan, it is extended using NaNs.
5.  If there are more than ``max_missing`` consecutive NaN in a ROI, then the ROI is flagged as
    completed. If the maximum intensity of a completed ROI is greater than ``min_intensity`` and
    the number of points is greater than ``min_length``, then the ROI is flagged as valid.
    Otherwise, the ROI is discarded.
6.  Repeat from step 2 until no more new scans are available.

References
----------

..  [1] Tautenhahn, R., Böttcher, C. & Neumann, S. Highly sensitive feature
    detection for high resolution LC/MS. BMC Bioinformatics 9, 504 (2008).
    https://doi.org/10.1186/1471-2105-9-504
