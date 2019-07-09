__all__ = []

from . import test_submodule
from .test_submodule import *
__all__.extend(test_submodule.__all__)
