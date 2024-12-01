.. _contributing-guide:

Contributing
============

This guide describes cover several topics regarding contributing to TidyMS.


.. _bug-report:

Reporting a bug
---------------

If you encounter a problem or bug, you can report it using the `issue tracker <https://github.com/griquelme/tidyms2/issues>`_.

Before submitting a new issue, please search the issue tracker to see if the problem has already been reported.

If your question is about how to achieve a specific task or use the library in a certain way, we recommend
posting it in the `discussions <https://github.com/griquelme/tidyms2/discussions>`_ section.

When reporting an issue, it's helpful to include the following details:

- A code snippet that reproduces the problem.
- If an error occurs, please include the full traceback.
- A brief explanation of why the current behavior is incorrect or unexpected.

For guidelines on how to write an issue report, refer to this `post <https://matthewrocklin.com/minimal-bug-reports>`_.

.. _dev-environment:

Setting up the development environment
--------------------------------------

TidyMS2 is developed for Python >= 3.11, with `uv <https://docs.astral.sh/uv/>`_ as the build
and dependency management tool. A development installation recipe is provided in the project's Makefile:

.. code-block:: shell

    make dev-install

Executing this command will install the package along with all dependencies required for building the
documentation and running tests. Additionally, `pre-commit <https://pre-commit.com/>`_ hooks are configured
to enforce code quality standards.

uv is a pre-requisite for development installation. We recommend installing uv using the official
`installation method <https://docs.astral.sh/uv/getting-started/installation/#installing-uv>`_.
As an alternative, you can create a virtual environment and install uv in it:

.. code-block:: shell

    python -m venv .venv
    source .venv/bin/activate
    pip install uv
    make dev-install

Code quality standards
----------------------

`Ruff <https://astral.sh/ruff>`_ is used as the formatting tool. All commits should follow this formatting style to
maintain consistency across the codebase. Please ensure your code is formatted with Ruff before committing. The
following makefile recipe applies the format to the codebase:

.. code-block:: shell

    make format

`Pyright <https://github.com/microsoft/pyright>`_ is used for static type checks. Before creating a pull request,
please ensure that Pyright runs without errors or warnings. The following makefile recipe runs Pyright on the
codebase:

.. code-block:: shell

    make check-types

All commits MUST follow the `conventional commit <https://www.conventionalcommits.org/en/v1.0.0/>`_ style. The
`commitizen <https://commitizen-tools.github.io/commitizen/>`_ tool comes bundled with the development dependencies.
This tool SHOULD be used to generate conventional commit messages.

Continuous integration
----------------------

The following checks are performed on all pull requests:

- code style check
- static type check
- unit tests

After merging a pull request, and before creating a new release, integration tests are executed.


Contributing to the codebase
----------------------------

Contributions to the project should be made using a fork/pull request workflow. Ensure that all unit and integration
tests pass successfully before submitting a PR. Additionally, new features must be :ref:`tested <testing>` and
:ref:`documented <documentation>`. Refer to the respective sections in this guide for detailed instructions on
testing and documentation procedures.


Versioning and releases
-----------------------

The library follows the `semantic versioning <https://semver.org/>`_ convention, using the format
MAJOR.MINOR.PATCH for releases. Releases are created automatically with the release please GitHub action.
The version number is bumped automatically based on `conventional commit <https://www.conventionalcommits.org/en/v1.0.0/>`_
messages.


.. _testing:

Testing
-------

The testing suite for this library is based on `Pytest <https://docs.pytest.org/en/stable/index.html>`_.
Tests are organized into two categories: unit and integration tests.

**Unit tests**

Unit tests verify the functionality of individual components within the library. These tests are isolated
from external systems, such as databases or REST APIs; external interactions are simulated using mocks
when necessary. Unit tests can be executed using the following command:

.. code-block:: shell

    make unit-tests

**Integration tests**

Integration tests evaluate the library's behavior in real-world scenarios. These tests tend to have
longer execution times and are not run by default. To execute integration tests, use the following
command:

.. code-block:: shell

    make integration-tests

We aim to achieve 100 % code coverage on the code base. A code coverage report can be created with the
following command:

.. code-block:: shell

    make coverage

.. _documentation:

Improving the documentation
---------------------------

The library's documentation is generated using `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
Docstrings are written in the sphinx style.

All public modules, classes, methods, and functions must include a docstring. While docstrings for private
functions and magic methods are not strictly required, it is strongly encouraged to provide at least a brief
description of their purpose and usage.

Additionally, tutorials are encouraged to illustrate the intended usage and best practices for utilizing the
library's features.

To generate the HTML documentation, navigate to the docs directory and execute the following command:

.. code-block:: shell

    make html

Deprecating Features
--------------------

Feature deprecations must be communicated to users with a warning, indicating the version in which the feature
will be removed. Deprecations should be marked in the following ways:

-   In the issue tracker: Open an issue in the GitHub repository to announce the deprecation and its timeline.
-   In the code: Use the ``.. deprecated::`` directive in the relevant function or class docstring, indicating
    the version when the feature will be removed.
-   Provide alternatives: If available, recommend an alternative approach or feature in the deprecation message.