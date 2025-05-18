.. _configuring-operators-guide:

Configuring operators
=====================

All operators are `pydantic models <https://docs.pydantic.dev/latest/>`_, and as such they can be instantiated in
a variety of ways. For example, we can use the ``model_validate`` method to
instantiate a new operator from a dictioanry:

.. code-block:: python

  from tidyms2.lcms.operators.sample import LCTraceExtractor

  op = LCTraceExtractor.model_validate({"id": "extractor". "tolerance": 0.01})

Besides this, all operators have the `from_defaults` method, which allows instanting
a new operator with sane default based on the data acquisition conditions. Three
constants are used for this, the :py:class:`~tidyms2.core.enums.MSInstrument`,
:py:class:`~tidyms2.core.enums.SeparationMode` and :py:class:`~tidyms2.core.enums.Polarity`
are used:

.. code-block:: python

  from tidyms2.core.enums import MSInstrument, SeparationMode, Polarity

  op = LCTraceExtractor.from_defaults(MSInstrument.QTOF, SeparationMode.UPLC, Polarity.NEGATIVE)

If an operator is stored in a data pipeline, it possible to retrieve it by id and
modify its attributes directly:

.. code-block:: python


  from tidyms2.core.operators.pipeline import Pipeline
  from tidyms2.lcms.operators.sample import LCTraceExtractor, LCTraceBaselineEstimator, LCPeakExtractor

  pipe = Pipeline("example_pipeline")
  pipe.add_operator(LCTraceExtractor(id="trace_extractor"))
  pipe.add_operator(LCTraceBaselineEstimator(id="baseline_estimator"))
  pipe.add_operator(LCPeakExtractor(id="peak_extractor"))


  op = pipe.get_operator("baseline_estimator")
  op.smoothing_strength = 2.0
