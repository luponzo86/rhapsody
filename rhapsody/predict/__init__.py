# -*- coding: utf-8 -*-
"""This subpackage contains the core Rhapsody class, the main interface and
relative functions needed for obtaining predictions from trained classifiers.
"""

__all__ = []

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

from . import core
from .core import *
__all__.extend(core.__all__)
__all__.append('core')

from . import main
from .main import *
__all__.extend(main.__all__)
__all__.append('main')

from . import figures
from .figures import *
__all__.extend(figures.__all__)
__all__.append('figures')
