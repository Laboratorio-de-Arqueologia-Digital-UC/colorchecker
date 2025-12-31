Installation Guide
==================

Because of their size, the resources dependencies needed to run the various
examples and unit tests are not provided within the Pypi package. They are
separately available as
`Git Submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`__
when cloning the
`repository <https://github.com/colour-science/colour-checker-detection>`__.

Primary Dependencies
--------------------

**Colour - Checker Detection** requires various dependencies in order to run:

- `python >= 3.10, < 3.14 <https://www.python.org/download/releases>`__
- `colour-science >= 4.5 <https://pypi.org/project/colour-science>`__
- `imageio >= 2, < 3 <https://imageio.github.io>`__
- `numpy >= 1.24, < 3 <https://pypi.org/project/numpy>`__
- `opencv-python >= 4, < 5 <https://pypi.org/project/opencv-python>`__
- `scipy >= 1.10, < 2 <https://pypi.org/project/scipy>`__

Secondary Dependencies
~~~~~~~~~~~~~~~~~~~~~~

- `click >= 8, < 9 <https://pypi.org/project/click>`__
- `ultralytics >= 8, < 9 <https://pypi.org/project/ultralytics>`__

Installation
------------

Use **uv** for dependency management:

.. code-block:: bash

    uv sync

This will install the project and its dependencies (including `ultralytics` if optional groups are requested).

To run scripts:

.. code-block:: bash

    uv run python colour_checker_detection/test.py
