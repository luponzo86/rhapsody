# -*- coding: utf-8 -*-
"""This subpackage contains modules for the initial package configuration
and for accessing installation settings and other parameters.
"""

__all__ = []

from . import settings
from .settings import *
__all__.extend(settings.__all__)
__all__.append('settings')
