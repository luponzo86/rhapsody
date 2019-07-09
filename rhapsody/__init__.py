from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, '../VERSION'), encoding='utf-8') as f:
    __version__ = f.read()

__release__ = __version__

__all__ = []

from . import settings
from .settings import *
__all__.extend(settings.__all__)

from . import PolyPhen2
from .PolyPhen2 import *
__all__.extend(PolyPhen2.__all__)

from . import Uniprot
from .Uniprot import *
__all__.extend(Uniprot.__all__)

from . import PDB
from .PDB import *
__all__.extend(PDB.__all__)

from . import EVmutation
from .EVmutation import *
__all__.extend(EVmutation.__all__)

from . import calcFeatures
from .calcFeatures import *
__all__.extend(calcFeatures.__all__)

from . import RFtraining
from .RFtraining import *
__all__.extend(RFtraining.__all__)

from . import rhapsody
from .rhapsody import *
__all__.extend(rhapsody.__all__)

from . import interfaces
from .interfaces import *
__all__.extend(interfaces.__all__)

from . import figures
from .figures import *
__all__.extend(figures.__all__)

from . import test_module
from .test_module import *
__all__.extend(test_module.__all__)
