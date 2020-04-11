FlyingCircus
============

**FlyingCircus-Numeric** - FlyingCircus with NumPy/Scipy.

.. code::

     _____ _       _              ____ _
    |  ___| |_   _(_)_ __   __ _ / ___(_)_ __ ___ _   _ ___
    | |_  | | | | | | '_ \ / _` | |   | | '__/ __| | | / __|
    |  _| | | |_| | | | | | (_| | |___| | | | (__| |_| \__ \
    |_|   |_|\__, |_|_| |_|\__, |\____|_|_|  \___|\__,_|___/
             |___/         |___/
    NUMERIC


Overview
--------

This software provides a library of miscellaneous utilities / recipes for
generic computations with Python and NumPy / SciPy.
This code was originally included in
`FlyingCircus <https://pypi.python.org/pypi/flyingcircus>`__, but is now
living in a separate package to avoid pulling larger dependencies to those
users not in need of NumPy / SciPy functionalities.
It is relatively easy to extend and users are encouraged to tweak with it.

Most of the code is used in a number of projects where it is tested
against real-life scenarios.

All the code is tested against the examples in the documentation
(using `doctest <https://docs.python.org/3/library/doctest.html>`__).

The code has reached a reasonable level of maturity.
However, until it gets a wider adoption, some of the library components may
undergo some refactoring in the process of improving the code.
Changes will appear in the ``CHANGELOG.rst``.
Please file a bug report if you detect an undocumented refactoring.

Releases information are available through ``NEWS.rst``.

For a more comprehensive list of changes see ``CHANGELOG.rst`` (automatically
generated from the version control system).


Features
--------

The package (which requires both ``numpy`` and ``scipy``
-- as well as `FlyingCircus <https://pypi.python.org/pypi/flyingcircus>`__) --
contains a number of numerical functions, typically
working on or generating ``numpy.ndarray`` inputs, like:

-  ``sgngeomspace()``: generates geometrically / logarithmically spaced
   samples between signed start and stop endpoints.
-  ``unsqueeze()``: add singletons to the shape of an array to
   broadcast-match a given shape.
-  ``subst()``: conveniently substitute all occurrences of a value in an array.

etc.

These are meant to run both in Python 3

Installation
------------

The recommended way of installing the software is through
`PyPI <https://pypi.python.org/pypi/flyingcircus_numeric>`__:

.. code:: bash

    $ pip install flyingcircus_numeric

Alternatively, you can clone the source repository from
`GitHub <https://github.com/norok2/flyingcircus_numeric>`__:

.. code:: bash

    $ git clone git@github.com:norok2/flyingcircus_numeric.git
    $ cd flyingcircus_numeric
    $ pip install -e .

For more details see also ``INSTALL.rst``.


License
-------

This work is licensed through the terms and conditions of the
`GPLv3+ <http://www.gnu.org/licenses/gpl-3.0.html>`__ See the
accompanying ``LICENSE.rst`` for more details.


Acknowledgements
----------------

For a complete list of authors please see ``AUTHORS.rst``.

People who have influenced this work are acknowledged in ``THANKS.rst``.
