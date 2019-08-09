# -*- coding: utf-8 -*-
"""This module defines a function for deriving features
from a precomputed BLOSUM substitution matrix."""

import numpy as np
from Bio.SubsMat.MatrixInfo import blosum62

__all__ = ['BLOSUM_FEATS', 'calcBLOSUMfeatures']

BLOSUM_FEATS = ['BLOSUM']
"""Features computed from BLOSUM62 substitution matrix."""


def calcBLOSUMfeatures(SAV_coords):
    feat_dtype = np.dtype([('BLOSUM', 'f')])
    features = np.zeros(len(SAV_coords), dtype=feat_dtype)
    for i, SAV in enumerate(SAV_coords):
        aa1 = SAV.split()[2]
        aa2 = SAV.split()[3]
        features[i] = blosum62.get((aa1, aa2), blosum62.get((aa2, aa1)))
    return features
