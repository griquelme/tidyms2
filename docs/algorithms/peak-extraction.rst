.. _algorithms-peak-extraction:

Peak extraction
===============

This guide describes theoretical background of the :term:`feature` extraction algorithm for LC-MS
datasets, implemented by the :py:class:`~tidyms2.lcms.LCPeakExtractor` operator.

In LC-MS datasets, features are represented as peaks, which  are extracted from :term:`m/z`
traces, as described in the :ref:`algorithms-roi-extraction` guide.

A peak is represented by a start, apex and end, as shown in the following figure:

..  plot:: plots/peak_definition.py
    :caption: Peak start, apex and end.

In the first release of TidyMS, peak picking worked using a modified version of the CWT algorithm,
described in [1]_. In chromatographic data, and in particular in untargeted datasets, optimizing
the parameters to cover the majority of peaks present in the data can be a tricky process. Some of
the problems that may appear while using the CWT algorithm are:

1.  sometimes when a lot of peak overlap occurs, peaks are missing. This is because peaks are
    identified as local maximum in the ridge lines from the wavelet transform. If the widths
    selected don't have the adequate resolution, this local maximum may not be found. Also, it
    is possible to have more than one local maximum in a given ridgeline, which causes to
    select one of them using ad hoc rules.
2.  The Ricker wavelet is the most used wavelet to detect peaks, as it has been demonstrated
    to work very with gaussian peaks. In LC-MS data, is common to find peaks with a certain
    degree of asymmetry (eg. peak tailing). Using the Ricker wavelet in these cases, results
    in a wrong estimation of the peak extension, which in turn results in bad estimates for
    some descriptors such as peak area or peak width.
3.  The interaction between the parameters in the CWT algorithm is rather complex, and
    sometimes it is not very clear how they affect the peak picking process. The user must
    have a clear knowledge of the wavelet transform to interpret parameters such as the SNR.
    Also there are a lot of specific parameters to tune the detection of the ridgelines.

These reasons motivated us to replace the CWT peak picking function with a local maxima based
peak estimation. The current algorithm works as follows:

1.  The noise level in the m/z trace is estimated.
2.  Using the noise level estimation, each point in the m/z trace is classified as baseline o
    signal. Baseline points are interpolated to build the m/z trace baseline.
3.  Peaks apex are detected using :py:func:`scipy.signal.find_peaks`. Peaks with a prominence
    lower than three times the noise level or in regions classified as baseline are removed.
4.  For each peak its extension is determined by finding the closest baseline point to its left
    and right.
5.  If there are overlapping peaks (overlapping peak extensions), the extension is fixed by
    defining a boundary between the peaks as the minimum value between the apex of the two peaks.

We now describe each step in detail.

Noise estimation
----------------

To estimate the noise and baseline, the discrete signal :math:`x[n]` is modelled as three additive
components:

.. math::
    x[n] = s[n] + b[n] + e[n]

:math:`s` is the peak component, which is deterministic, non negative and small except regions
where peaks are present. The baseline :math:`b` is a smooth slow changing function. The noise
term :math:`e[n]` is assumed to be independent and identically distributed (iid) samples from a
gaussian distribution :math:`e[n] \sim N(0, \sigma)`.

If we consider the second finite difference of :math:`x[n]`, :math:`y[n]`:

.. math::
    y[n] = x[n] - 2 x[n-1] + x[n-2]

As :math:`b` is a slow changing function we can ignore its contribution. We expect that the
contribution from :math:`s` in the peak region is greater than the noise contribution, but if we
ignore higher values of :math:`y` we can focus on regions where :math:`s` is small we can say that
most of the variation in :math:`y` is due to the noise:

.. math::
    y[n] \approx (e[n] - 2 e[n-1] + e[n-2])

Within this approximation, we can say that :math:`y[n] \sim N(0, 2\sigma)`. The noise estimation
tries to exploit this fact, estimating the noise from the standard deviation of the second
difference of :math:`x`. The algorithm used can be summarized in the following steps:

1.  Compute the second difference of :math:`x`, :math:`y`.
2.  Set :math:`p=90`, the percentile of the data to evaluate.
3.  compute :math:`y_{p}` the p-th percentile of the absolute value of :math:`y`.
4.  Compute the mean :math:`\overline{y}` and standard deviation :math:`S_{y}` of :math:`y`
    restricted to elements with an absolute value lower than :math:`y_{p}`. This removes the
    contribution of :math:`s`.
5.  If :math:`|\overline{y}| \leq S_{y}` or :math:`p \leq 20` then the noise level is
    :math:`\sigma = 0.5 S_{y}`. Else decrease :math:`p` by 10 and go back to step 3.

If the contribution from :math:`s` is not completely removed, the noise estimation will be biased.
Despite this, this method gives a good enough approximation of the noise level that can be used to
remove noisy peaks.

Baseline  estimation
--------------------

Baseline estimation is done with the following approach: first, every point in :math:`x` is
classified as signal if a peak can potentially be found in the region or as or as baseline
otherwise. Then, the baseline is estimated for the whole signal by interpolating baseline points.

The main task of baseline estimation is then to perform this classification process. To do this,
all local extrema in the signal are searched (including first and last points). Then, we take all
closed intervals defined between consecutive local maxima and minima (or vice versa) and try to
evaluate if there is a significant contribution to the signal coming from :math:`s` in each
interval. If :math:`j` and :math:`k` are the indices defining one such interval, then the sum of
:math:`x` in the interval is:

.. math::
    \sum_{i=j}^{k}x[i] = \sum_{i=j}^{k} s[i] + b[i] + e[i]

If :math:`l = k - j` is the length of the interval, and assuming that :math:`b` is constant in the
interval we can write:

.. math::
    \sum_{i=j}^{k} x[i] - x[j] = \sum_{i=j}^{k} s[i] - s[j] +
    \sum_{i=j}^{k} e[i] -e[j]

.. math::
    a = \sum_{i=j}^{k} x[i] - x[j] = \sum_{i=j}^{k} s[i] - s[j] + e_{sum}

Where :math:`e_{sum} \sim N(0, \sqrt{2l}\sigma)` (we know :math:`\sigma` from the noise estimation).
We can get an idea of the contribution of :math:`s` by using the value of :math:`a` as follows: If
the signal term is contributing to :math:`a`, then the probability of obtaining a value greater than
:math:`a` from noise is going to be small. This can be computed in the following way:

.. math::
    P(|e_{sum}| > |a|)= \textrm{erfc} \left (\frac{|a|}{2\sqrt{l}\sigma}
    \right )

An interval is classified as baseline if this probability is greater than 0.05.

The following figure shows the result of the peak picking algorithm with different :term:`SNR`
levels, baseline shapes and peak widths.

..  plot:: plots/peak_extraction_example.py
    :caption: Peak detection and baseline estimation in noisy signals.

References
----------

..  [1] Pan Du, Warren A. Kibbe, Simon M. Lin, Improved peak detection in mass spectrum by
    incorporating continuous wavelet transform-based pattern matching, Bioinformatics, Volume
    22, Issue 17, 1 September 2006, Pages 2059-2065, https://doi.org/10.1093/bioinformatics/btl355
