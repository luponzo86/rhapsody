# -*- coding: utf-8 -*-
"""This subpackage contains modules for computing features from multiple
sources, e.g. Uniprot sequences, PDB structures, Pfam domains and
EVmutation precomputed data.
"""

__all__ = ['RHAPSODY_FEATS']

from . import Uniprot
from .Uniprot import *
__all__.extend(Uniprot.__all__)
__all__.append('Uniprot')

from . import PDB
from .PDB import *
__all__.extend(PDB.__all__)
__all__.append('PDB')

from . import PolyPhen2
from .PolyPhen2 import *
__all__.extend(PolyPhen2.__all__)
__all__.append('PolyPhen2')

from . import EVmutation
from .EVmutation import *
__all__.extend(EVmutation.__all__)
__all__.append('EVmutation')

from . import Pfam
from .Pfam import *
__all__.extend(Pfam.__all__)
__all__.append('Pfam')

from . import BLOSUM
from .BLOSUM import *
__all__.extend(BLOSUM.__all__)
__all__.append('BLOSUM')

# list of all available features in RHAPSODY
RHAPSODY_FEATS = {
    'PolyPhen2': set(PolyPhen2.PP2_FEATS),
    'PDB': set(PDB.PDB_FEATS),
    'Pfam': set(Pfam.PFAM_FEATS),
    'BLOSUM': set(BLOSUM.BLOSUM_FEATS),
    'EVmut': set(EVmutation.EVMUT_FEATS),
}
RHAPSODY_FEATS['all'] = set().union(*RHAPSODY_FEATS.values())
