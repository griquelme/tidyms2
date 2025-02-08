.. _simulation-guide:

Simulating data
===============

This guide describes how to use the simulation utilities provided by the library.

Currently, only :term:`LC-MS` simulation is implemented. The aim of the simulation utilities is not
to synthesize signals that mimic `real world` signals, but to provide tools that can be used to
test and develop data pipelines. Despite this, the simulation utilities allow to take into account
several effects observed on real datasets:

- instrumental variations in :term:`m/z` and :term:`Rt`
- inter-sample variation in compounds abundance
- compound prevalence
- instrumental variation in spectra
- time-dependent signal loss
- inter-batch signal variation
- signal from multiple isotopologues

Two tools are provided for this end:

-   :py:class:`~tidyms2.simulation.lcms.SimulatedLCMSSampleFactory`: A factory class that creates
    :py:class:`~tidyms2.core.models.Sample` models with synthesized data that behaves in the same way
    as a sample with real data.
-   :py:func:`~tidyms2.simulation.lcms.simulate_data_matrix`: a function that creates a data matrix
    with simulated data.

We'll start by showing an example of how sample data can be created. Then, we will cover the
:py:class:`~tidyms2.simulation.lcms.SimulatedLCMSAdductSpec` which defines the properties of
features in simulated samples. Finally we will describe in detail sample and matrix data simulation.

Introductory example
--------------------

The following code block allows to create a simulated sample with three adducts:

.. code-block:: python

    from tidyms2.simulation.lcms import SimulatedLCMSSampleFactory
    from tidyms2.io import MSData

    adducts_spec = [
        {
            "formula": "[C54H104O6]+",
            "rt": {"mean": 10.0},
        },
        {
            "formula": "[C27H40O2]+",
            "rt": {"mean": 20.0},
        },
        {
            "formula": "[C24H26O12]+",
            "rt": {"mean": 30.0},
        },
    ]

    simulated_sample_factory = SimulatedLCMSSampleFactory(adducts=adducts_spec)
    sample = simulated_sample_factory(id="my_sample")

The factory class :py:class:`~tidyms2.simulation.lcms.SimulatedLCMSSampleFactory` allows to create
sample models that can be used pretty much in the same way as samples created with real data. For
example, we can create an MSData instance and retrieve "raw" data from each simulated scan:

.. code-block:: python

    simulated_sample_factory = SimulatedLCMSSampleFactory(**factory_spec)
    sample = simulated_sample_factory(id="my_sample")
    ms_data = MSData(sample)

    spectrum = ms_data.get_spectrum(10) # get the 10th scan from the simulated data

The following figure displays an extracted ion chromatogram created from this simulated sample:

..  plot:: plots/simulated-lcms-sample.py
    :caption: :term:`EIC` plot using simulated data.

The :py:class:`~tidyms2.simulation.lcms.SimulatedLCMSSampleFactory` contains two fields:

config
    contains the parameters related with simulating the data acquisition conditions, such as time
    resolution, number of scans, `m/z` grid resolution, etc... 
adducts
    contain a list adduct specifications, which define the features that will be observed in the
    data.


The sample factory configuration will be discussed in the :ref:`sample-simulation-guide` section.
We start by discussing the adduct specification in detail as it is the key component used in
both sample and matrix data simulation.

.. _adduct-spec-guide:

Adduct specification
--------------------

The observed signal :math:`x` is modelled using the following expression:

.. math::

    x_{0} = f * c * p_{k} + \epsilon

where :math:`f` is the response factor, :math:`c` is the abundance of the chemical species that generates the
adduct, :math:`p_{k}` is the relative abundance of the corresponding isotopologue and :math:`\epsilon` is an
additive noise term. Each one of these variables model different sources of variation observed in the data. The
:py:class:`~tidyms2.simulation.lcms.SimulatedLCMSAdductSpec` describes how each one of the terms in the equation
are computed.

The chemical characteristics are specified through the 
:py:attr:`~tidyms2.simulation.lcms.SimulatedLCMSAdductSpec.formula` and
:py:attr:`~tidyms2.simulation.lcms.SimulatedLCMSAdductSpec.n_isotopologues` fields, which define the
chemical formula, and the number of isotopologues to simulate (one by default). The chemical formula is a
string representation of a sum formula and may include a charge as shown in the :ref:`working-with-formulas-guide`
guide. Using these parameters the :math:`p_{k}` is computed for each simulated isotopologue.

The :py:class:`~tidyms2.simulation.base.AbundanceSpec` model defines how the abundance :math:`c` is computed
for a given sample. The mean abundance and variation is modeled as a gaussian distribution, allowing to emulate
variation between different observations. The :py:attr:`~tidyms2.simulation.base.AbundanceSpec.prevalence` field
allows to define the fraction of sample where the chemical species that generate the adduct will be observed.
By default a sample will have an abundance equal to 100, no variation between samples (:math:`\sigma=0`) and
a prevalence equal to 1.

The response factor :math:`f` can model time-dependent signal drift and inter-batch effects, but to keep the explanation
as simple as possible the details about this are moved into the :ref:`response-factor-simulation-guide` section.
If the default parameters are used, the response factor will be constant and equal to 1. 

The additive noise :math:`\epsilon` is computed using the :py:class:`~tidyms2.simulation.base.MeasurementNoiseSpec`,
which samples the noise from a Gaussian distribution, ensuring that the :term:`snr` follows the following
relation:

.. math::

        \textrm{SNR} = \max (\textrm{SNR}_{\textrm{min}}, \textrm{SNR}_{\textrm{base}} * p_{k})

where :math:`\textrm{SNR}` is the :term:`SNR` for the k-th isotopologue; :math:`\textrm{SNR}_{\textrm{base}}` is
the adduct base signal-to-noise ratio, and :math:`\textrm{SNR}_{\textrm{min}}` is the minimum signal-to-noise ratio
allowed. This allows to scale the :term:`snr` of less abundant isotopologues. For this computation we define
the :term:`snr` as follows:

.. math::

    \textrm{SNR} = \frac{c * f * p_{k}}{\sigma}

where :math:`\sigma` is the standard deviation of the distribution that generates the noise.
In the default configuration, the additive noise term is set to zero.

Finally, the chromatographic peak specification defines the properties of the chromatographic peak, such as
mean retention time, retention time variation across samples and chromatographic peak width. This specification
is defined by the :py:class:`tidyms2.simulation.lcms.RtSpec` model. The observed retention time is sampled from
a gaussian using the mean retention time and the retention time dispersion. The peak shape is modelled using the
following expression:

.. math::

    x(t) = x_{0} \exp \left ( - \frac{(t - \textrm{Rt})^{2}}{2w^{2}} \right )

where :math:`\textrm{Rt}` is the observed retention time for a sample and :math:`w` is the chromatographic peak
width parameter. In the default configuration, peaks will have a retention time equal to 10 and a peak width
equal to 3.

With this information, we can rewrite the adduct specification example to customize the appearance of the observed
signals:

.. code-block:: python

    adducts_spec = [
        {
            "formula": "[C54H104O6]+",
            "n_isotopologues": 3,
            "abundance": {
                "mean": 500,
                "std": 50,
            },
            "rt": {
                "mean": 10.0,
                "std": 1.0,
                "width": 4.0,
            },
            "noise": {
                "snr": 200,
            }
        },
        {
            "formula": "[C27H40O2]+",
            "n_isotopologues": 2,
            "abundance": {
                "mean": 200,
                "std": 50,
                "prevalence": 0.7,
            },
            "rt": {"mean": 20.0},
        },
        {
            "formula": "[C24H26O12]+",
            "n_isotopologues": 5,
            "rt": {"mean": 30.0},
        },
    ]

.. _sample-simulation-guide:

Sample simulation
-----------------

We already covered the adduct specification, which defines the properties of features that are
observed in a sample. The other configuration available for the sample factory is the data
acquisition specification, defined by the :py:class:`~tidyms2.simulation.base.DataAcquisitionSpec`
model. This models allows to define:

- instrumental noise (m/z and intensity)
- time resolution
- number of scans
- data mode (profile or centroid)

Instrumental noise is an additive noise added to m/z and intensity values on each scan. The values
are samples from a gaussian distribution using the fields :py:attr:`~tidyms2.simulation.base.DataAcquisitionSpec.mz_std` 
and :py:attr:`~tidyms2.simulation.base.DataAcquisitionSpec.int_std` respectively.

The number of scans and time resolution can be configured by the corresponding fields in the specification.
The :py:attr:`~tidyms2.simulation.base.DataAcquisitionSpec.min_int` removes points in spectra with
intensities lower than this value. This parameter is particularly useful for data simulation in profile mode,
as it drastically reduces the size of simulated spectra.

In the default configuration, the data is simulated in centroid mode, without `m/z` or intensity noise, a
time resolution equal to 1 and 100 scans.

The following example show simulated centroid data with a custom acquisition specification:

..  plot:: plots/simulated-lcms-sample-advanced.py
    :include-source: true
    :caption: :term:`EIC` plot using simulated data with a custom acquisition specification.

To simulate data in profile mode, an `m/z` grid specification must be provided. This specification is
defined by the :py:class:`~tidyms2.simulation.base.MZGridSpec` which defines the minimum and maximum
`m/z` values as well as the number of points in the grid. The generated grid is uniformly spaced, and
as MS data is sparse, it is useful to remove parts of the grid that do not contain useful information
using the :py:attr:`~tidyms2.simulation.base.DataAcquisitionSpec.min_int` field. Finally, the
:py:attr:`~tidyms2.simulation.base.DataAcquisitionSpec.mz_width` parameter controls the peak width
in the `m/z` dimension. This parameter allows to model signal overlap in the m/z domain, something that
cannot be done in centroid mode simulation. :py:attr:`~tidyms2.simulation.base.DataAcquisitionSpec.mz_width`
is ignored in centroid mode.

The following example show show an MS spectrum in profile mode:

..  plot:: plots/simulated-lcms-profile-spectrum.py
    :caption: Simulated MS spectrum in profile mode.

.. _matrix-simulation-guide:

Data matrix simulation
----------------------

Data matrix simulation is a lot simpler to set up as no data acquisition specification is required. In
this case, two things need to be defined:

- an adduct specification
- a list of samples to build the data matrix.

The following example shows a basic example of data matrix simulation:

..  plot:: plots/simulated-lcms-matrix-basic.py
    :include-source: true
    :caption: Data matrix value for a simulated feature.

The Abundance specification can be defined on a per-group basis, by passing a dictionary that maps sample
groups to abundance specifications. The following example defines two sample groups with different feature
mean values:


..  plot:: plots/simulated-lcms-matrix-multiple-groups.py
    :caption: Data matrix value for a simulated feature.


.. _response-factor-simulation-guide:

Computing the response factor
-----------------------------

The response factor :math:`f` observed on each sample is defined by the
:py:class:`~tidyms2.simulation.base.InstrumentResponseSpec` model, which can be customized for each adduct.
In the default configuration no time-dependent effects are included. The response factor can be used to
model time-dependent variation between samples, as described by the following equation:

.. math::

    f = \tilde{f} b ((1 - M) + M \exp (- i \lambda))

Where :math:`\tilde{f}` is the :py:attr:`~tidyms2.simulation.base.InstrumentResponseSpec.base_response_factor`,
:math:`M` is the :py:attr:`~tidyms2.simulation.base.InstrumentResponseSpec.max_sensitivity_loss`
and :math:`\lambda` is the :py:attr:`~tidyms2.simulation.base.InstrumentResponseSpec.sensitivity_decay`.

:math:`b` is the inter-batch variation, a random value sampled from a uniform distribution
with minimum equal to the :py:attr:`~tidyms2.simulation.base.InstrumentResponseSpec.interbatch_variation`
parameter and maximum equal to 1. This value is sampled once for each analytical batch value and applied
to all observations from that batch.

The following code example shows matrix data simulated with time-dependent effects:

..  plot:: plots/simulated-lcms-matrix-sensitivity-loss.py
    :caption: Data matrix value for a simulated feature with time-dependent variations.