.. _core-api:

Core components
===============

.. _core-enums-api:

Enumerations
------------

.. autoclass:: tidyms2.core.enums.DataType()
    :members:

.. autoclass:: tidyms2.core.enums.MSInstrument()
    :members:

.. _core-models-api:

Data models
-----------

.. automodule:: tidyms2.core.models
   :members:
        Roi,
        AnnotableFeature,
        Feature,
        Sample,
   :show-inheritance:
   :inherited-members: BaseModel
   :exclude-members: model_computed_fields, model_config, model_fields


.. _core-operators-api:

Operators
---------

.. automodule:: tidyms2.core.operators
   :members:
        BaseOperator,
        SampleOperator,
        RoiExtractor,
        RoiTransformer,
        FeatureExtractor,
        FeatureTransformer
   :show-inheritance:
   :exclude-members: model_computed_fields, model_config, model_fields


.. _core-dataflow-api:

Data flow
---------

.. automodule:: tidyms2.core.dataflow
   :members:
        BaseProcessStatus,
        SampleProcessStatus,
        AssayProcessStatus,
        MatrixProcessStatus
   :show-inheritance:
   :inherited-members: BaseModel
   :exclude-members: model_computed_fields, model_config, model_fields