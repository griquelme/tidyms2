.. _raw-data-guide:

Working with raw data
=====================

TidyMS2 provides utilities to read raw data in the mzML format. Refer to :ref:`this guide <mzml-guide>`
for details on converting files in proprietary, instrument specific formats to mzML.

The first core component that we will present is :py:class:`~tidyms2.MSData`. This class provides fast
index-based access to spectra and chromatogram data stored in a raw data file. The following code snippet
downloads an example mzML file and creates an :py:class:`~tidyms2.MSData` using it:

.. code-block:: python

    import pathlib

    from tidyms2 import MSData
    from tidyms2.io.datasets import download_dataset

    # you may need to provide a GitHub personal access token to download the dataset or 
    # download the data manually from
    # https://github.com/griquelme/tidyms-data/tree/master/test-nist-raw-data
    data_dir = pathlib.Path("data")
    token = None 
    download_dataset("test-nist-raw-data", data_dir, token=token)

    filename = "NZ_20200227_039.mzML"
    data = MSData(data_dir / filename)

    print(f"{filename} contains {data.get_n_spectra()} spectra.")

Individual :py:class:`~tidyms2.core.models.MSSpectrum` can be retrieved by index, i.e., the order in which
spectra appear in the raw data file:

.. code-block:: python

    index = 5
    spectrum = data.get_spectrum(index)

:py:class:`~tidyms2.core.models.MSSpectrum` is one of the data models that we mentioned above and it acts as
a container for m/z and spectral intensity in a single data scan.

.. code-block:: python

    print(spectrum.mz[:10])

MSData acts also as an spectra iterator, allowing to get all data from a file:

.. code-block:: python

    for sp in data:
        print(sp.index)

It is often the case that it is only required to iterate over a specific time window or MS level. This
information can be passed to the MSData constructor, but this information is better stored using
the :py:class:`~tidyms2.core.models.Sample` data model. This model uniquely defines a measurement and
it will be a core part to define datasets. A sample points to the raw data file associated with a
measurement, but it also includes information from MS level, start and end time and other metadata.
The following code snippet reads data defined using a sample model:

.. code-block:: python

    from tidyms2 import Sample

    sample = Sample(id="sample-39", path=data_dir / filename, ms_level=1, start_time=120.0, end_time=150.0)
    data = MSData(sample)

    for sp in data:
        print(f"Current scan time is {sp.time:.2f} s. MS level is {sp.ms_level}.")


The MSData reads data in a lazy manner, i.e., only the required scans are loaded into memory. In the default
configuration, once read, data is cached in to memory for faster retrieval. The MSData cache size can be configured
to limit the amount of memory used:

.. code-block:: python

    filename = "NZ_20200227_039.mzML"
    data = MSData(data_dir / filename, cache= 50 * 1024**2)   # maximum cache size of 50 MiB

Finally, a centroider function may be plugged into an MSData instance to convert profile data to centroid mode as it is
fetched from disk:

TODO: complete