#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlyingCircus-Numeric - FlyingCircus with NumPy/SciPy
"""

# Copyright (c) Riccardo Metere <rick@metere.it>

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import datetime  # Basic date and time types
import inspect  # Inspect live objects
import os  # Miscellaneous operating system interfaces
import appdirs  # Determine appropriate platform-specific dirs
# import pkg_resources  # Manage package resource (from setuptools module)
import doctest  # Test interactive Python examples

# ======================================================================
# :: External Imports
import flyingcircus as fc  # Everything you always wanted to have in Python*
from flyingcircus import msg, dbg, fmt, fmtm, elapsed, report, pkg_paths
from flyingcircus import run_doctests
from flyingcircus import VERB_LVL, VERB_LVL_NAMES, D_VERB_LVL
from flyingcircus import HAS_JIT, jit

# ======================================================================
# :: Version
from flyingcircus_numeric._version import __version__

# ======================================================================
# :: Project Details
INFO = {
    'name': 'FlyingCircus_Numeric',
    'author': 'FlyingCircus developers',
    'contrib': (
        'Riccardo Metere <rick@metere.it>',
    ),
    'copyright': 'Copyright (C) 2014-2020',
    'license': 'GNU General Public License version 3 or later (GPLv3+)',
    'notice':
        """
This program is free software and it comes with ABSOLUTELY NO WARRANTY.
It is covered by the GNU General Public License version 3 (GPLv3+).
You are welcome to redistribute it under its terms and conditions.
        """,
    'version': __version__
}

# ======================================================================
# :: quick and dirty timing facility
_EVENTS = []

# ======================================================================
# Greetings
MY_GREETINGS = r"""
 _____ _       _              ____ _                    
|  ___| |_   _(_)_ __   __ _ / ___(_)_ __ ___ _   _ ___ 
| |_  | | | | | | '_ \ / _` | |   | | '__/ __| | | / __|
|  _| | | |_| | | | | | (_| | |___| | | | (__| |_| \__ \
|_|   |_|\__, |_|_| |_|\__, |\____|_|_|  \___|\__,_|___/
         |___/         |___/
                                                 NUMERIC                     
                                                        
"""

# generated with: figlet 'FlyingCircus' -f standard

# :: Causes the greetings to be printed any time the library is loaded.
# print(MY_GREETINGS)


# ======================================================================
PATH = pkg_paths(__file__, INFO['name'], INFO['author'], INFO['version'])

# ======================================================================
elapsed(os.path.basename(__file__))

# ======================================================================
# : populate flyingcircus namespace with submodules
from flyingcircus_numeric.base import *

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
