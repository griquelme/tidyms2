.. _processing-individual-samples:

Processing individual samples
=============================

We will present here both the low and high level interfaces to process data from individual samples. Before
starting with these topics, we will create a simulated sample that will be used in the examples. This can be
done by using the simulation module:

.. code-block:: python

    from tidyms2.lcms.simulation import SimulatedLCMSSampleFactory

    factory_spec = {
        "config": {
            "min_int": 1.0,
            "n_scans": 40,
            "int_std": 0.0,
        },
        "adducts": [
            {
                "formula": "[C54H104O6]+",
                "rt": {"mean": 10.0},
                "abundance": {"mean": 1000.0},
                "n_isotopologues": 2,
            },
            {
                "formula": "[C27H40O2]+",
                "rt": {"mean": 20.0},
                "abundance": {"mean": 2000.0},
                "n_isotopologues": 2,
            },
            {
                "formula": "[C24H26O12]+",
                "rt": {"mean": 30.0},
                "abundance": {"mean": 3000.0},
                "n_isotopologues": 2,
            },
        ],
    }

    simulated_sample_factory = SimulatedLCMSSampleFactory(**factory_spec)
    sample = simulated_sample_factory(id="my_sample")

Simulated samples can be used as regular sample models to access "raw" data:

.. code-block:: python

    from tidyms2 import MSData

    data = MSData(sample)
    spectrum = data.get_spectrum(10)  # get the 10th scan from the simulated data

We are going to use this simulated sample on all the examples in this page. For more details about
how to create simulated samples, refer to :ref:`this guide <simulation-guide>`.

Low-level API
-------------

Basic data processing from raw data can be done using low level functions available in the `algorithms`
subpackage. For example, using the `algorithms.raw` module, we can create :term:`EIC` by providing a list of
`m/z` values:

.. code-block:: python

    from tidyms2.algorithms.raw import MakeChromatogramParameters, make_chromatograms
    from tidyms2.chem import Formula

    # we are using here m/z values from from two of the molecules included in the
    # simulated sample.
    mz1 = Formula("[C27H40O2]+").get_exact_mass()
    mz2 = Formula("[C54H104O6]+").get_exact_mass()

    params = MakeChromatogramParameters(mz=[mz1, mz2])
    chromatograms = make_chromatograms(data, params)

If you installed the plotting optional dependencies, you can plot the chromatograms using the following
function:

TODO: COMPLETE

The :py:func:`~tidyms2.algorithms.raw.accumulate_spectra` combines a series of successive spectra into a
single spectrum:

.. code-block:: python

    from tidyms2.algorithms.raw import AccumulateSpectraParameters, accumulate_spectra

    params = AccumulateSpectraParameters(start_time=10.0, end_time=15.0)
    accumulated_spectrum = accumulate_spectra(data, params)

The `algorithms` package provides several utilities to process data, such as untargeted :term:`ROI` creation
from raw data or peak picking routines, but you need to write the logic for moving and storing data around.
The high level API provides an easy to use way to process data and orchestrates data storage between
transformations.

High level API
--------------

The high level API is based on the TidyMS data model. The general idea is that sample
transformations are defined in a data pipeline, which is comprised by a sequence of
operators. Operators are applied one at a time to sample data. We will illustrate
data processing using individual operators first, and then we show how to apply
multiple operators using a pipeline.

The first thing to do to process sample data through a pipeline is to create a sample
data storage. The sample storage is a container class that store ROIs and features
extracted from raw data. We will use the :py:class:`~tidyms2.storage.OnMemorySampleStorage`
class for this:

.. code-block:: python

    from tidyms2.lcms import MZTrace, Peak
    from tidyms2.storage import OnMemorySampleStorage

    sample_data = OnMemorySampleStorage(sample, MZTrace, Peak)

Note that, besides the sample model, we also need to pass the type of data that we
want to store (MZTrace and Peak). During sample processing sample data is stored
either as ROI or features, two core entities of the TidyMS data model. A ROI is a
data subset extracted from raw data. A feature is a ROI subregion that contains
information associated with a chemical species. The sample data storage class manage
ROI/feature storage and retrieval extracted from raw data. For LC-MS data, ROI is
modeled as a the :py:class:`~tidyms2.core.models.MZTrace`, which is basically an m/z trace
containing m/z and intensity values from each scan. :py:class:`~tidyms2.lcms.Peak`
models a chromatographic peak detected in an m/z trace. We will use the LC-MS sample
operators :py:class:`~tidyms2.lcms.LCTraceExtractor` and
:py:class:`~tidyms2.lcms.LCPeakExtractor` to extract m/z traces from raw data
and then detect peaks on each m/z trace. This code snippet applies both operators to a
sample and then plots one of the detected peaks:

..  plot:: plots/single_sample_processing.py
    :include-source: true
    :caption: Simulated MS spectrum in profile mode.

Instead of applying operators one by one, multiple operators can be chained together
in a pipeline:

.. code-block:: python

  from tidyms2.core.operators.pipeline import Pipeline
  from tidyms2.lcms.operators.sample import LCTraceExtractor, LCTraceBaselineEstimator, LCPeakExtractor

  pipe = Pipeline("example_pipeline")
  pipe.add_operator(LCTraceExtractor(id="trace_extractor"))
  pipe.add_operator(LCTraceBaselineEstimator(id="baseline_estimator"))
  pipe.add_operator(LCPeakExtractor(id="peak_extractor"))

  pipe.apply(sample_data)

This is the approach used in multiple sample processing: a pipeline is designed for
processing samples and then it is applied to multiple samples.
