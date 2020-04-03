FlyingCircus
============

**FlyingCircus** - Everything you always wanted to have in Python.\*

(\*But were afraid to write)

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
It is relatively easy to extend and users are encouraged to tweak with it.

Most of the code is used in a number of projects where it is tested
against real-life scenarios.

As a result of the code maturity, some of the library components may
undergo (eventually heavy) refactoring.
While this is not expected, this will be documented.
Please file a bug report if you detect an undocumented refactoring.

Releases information are available through ``NEWS.rst``.

For a more comprehensive list of changes see ``CHANGELOG.rst`` (automatically
generated from the version control system).


Features
--------

The package contain two main sub-packages:

-  ``base``


The package ``base`` contains a number of generic functions like

-  ``multi_replace()``: performs multiple replacements in a string.
-  ``flatten()``: recursively flattens nested iterables, e.g.
   list of list of tuples to flat list).
-  ``uniques()``: extract unique items from an iterable while
   keeping the order of appearance.

etc.

The package ``extra`` (which requires both ``numpy`` and ``scipy``)
contains a number of numerical functions, typically
working on or generating ``numpy.ndarray`` inputs, like:

-  ``sgngeomspace()``: generates geometrically / logarithmically spaced
   samples between signed start and stop endpoints.
-  ``unsqueeze()``: add singletons to the shape of an array to
   broadcast-match a given shape.
-  ``subst()``: conveniently substitute all occurrences of a value in an array.

etc.

Additional packages may be added in the future.


Installation
------------

The recommended way of installing the software is through
`PyPI <https://pypi.python.org/pypi/flyingcircus>`__:

.. code:: bash

    $ pip install flyingcircus

Alternatively, you can clone the source repository from
`Bitbucket <https://bitbucket.org/norok2/flyingcircus>`__:

.. code:: bash

    $ git clone git@bitbucket.org:norok2/flyingcircus.git
    $ cd flyingcircus
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
