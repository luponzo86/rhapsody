# -*- coding: utf-8 -*-
"""Rhapsody: a program for pathogenicity prediction of human missense
variants based on sequence, structure and dynamics of proteins
"""

from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'VERSION'), encoding='utf-8') as f:
    __version__ = f.read()

__release__ = __version__

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

__all__ = []

from . import utils
from .utils import *
__all__.extend(utils.__all__)
__all__.append('utils')

from . import train
from .train import *
__all__.extend(train.__all__)
__all__.append('train')

from . import features
from .features import *
__all__.extend(features.__all__)
__all__.append('features')

from . import predict
from .predict import *
__all__.extend(predict.__all__)
__all__.append('predict')
