# -*- coding: utf-8 -*-
"""This subpackage contains modules for the initial package configuration
and for accessing installation settings and other parameters.
"""

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

__all__ = []

from . import settings
from .settings import *
__all__.extend(settings.__all__)
__all__.append('settings')
