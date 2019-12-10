# -*- coding: utf-8 -*-
"""This module defines a function for computing conservation and
coevolution properties of an amino acid substitution from a Pfam
multiple sequence alignment."""

import os
import numpy as np
from tqdm import tqdm
from prody import LOGGER
from .Uniprot import UniprotMapping

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

__all__ = ['PFAM_FEATS', 'calcPfamFeatures']

PFAM_FEATS = ['entropy', 'ranked_MI']
"""List of features computed from Pfam multiple sequence alignments."""


def _calcEvolFeatures(PF_dict, pos):
    def calcNormRank(array, i):
        # returns rank in descending order
        order = array.argsort()
        ranks = order.argsort()
        return ranks[i]/len(ranks)
    entropy = 0.
    rankdMI = 0.
    # rankdDI = 0.
    n = 0
    for ID, Pfam in PF_dict.items():
        if isinstance(Pfam['mapping'], dict):
            n += 1
            indx = Pfam['mapping'][pos - 1]
            entropy += Pfam['entropy'][indx]
            rankdMI += calcNormRank(np.sum(Pfam['MutInfo'], axis=0), indx)
            # rankdDI += calcNormRank(np.sum(Pfam['DirInfo'], axis=0), indx)
    if n == 0:
        raise ValueError("Position couldn't be mapped on any Pfam domain")
    else:
        feats = (entropy/n, rankdMI/n)
        # feats = (entropy/n, rankdMI/n, rankdDI/n)
        return feats


def calcPfamFeatures(SAVs, status_file=None, status_prefix=None):
    LOGGER.info('Computing sequence properties from Pfam domains...')
    LOGGER.timeit('_calcPfamFeats')
    # sort SAVs, so to group together those
    # with identical accession number
    accs = [s.split()[0] for s in SAVs]
    sorting_map = np.argsort(accs)
    # define a structured array for features computed from Pfam
    num_SAVs = len(SAVs)
    feat_dtype = np.dtype([('entropy', 'f'), ('ranked_MI', 'f')])
    features = np.zeros(num_SAVs, dtype=feat_dtype)
    # define how to report progress
    if status_prefix is None:
        status_prefix = ''
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    if status_file is not None:
        status_file = open(status_file, 'w')
        progress_bar = tqdm(
            [(i, SAVs[i]) for i in sorting_map], file=status_file,
            bar_format=bar_format+'\n')
    else:
        progress_bar = tqdm(
            [(i, SAVs[i]) for i in sorting_map], bar_format=bar_format)
    # map to Pfam domains using UniprotMapping class
    cache = {'acc': None, 'obj': None, 'warn': ''}
    count = 0
    for indx, SAV in progress_bar:
        count += 1
        acc, pos, aa1, aa2 = SAV.split()
        pos = int(pos)
        # report progress
        progress_msg = f"{status_prefix}Mapping SAV '{SAV}' to Pfam"
        # LOGGER.info(f"[{count}/{num_SAVs}] {progress_msg}...")
        progress_bar.set_description(progress_msg)
        # map to Pfam domains using 'UniprotMapping' class
        if acc == cache['acc']:
            # use object from previous iteration
            obj = cache['obj']
        else:
            # save previous object
            if cache['obj'] is not None:
                cache['obj'].savePickle()
            cache['acc'] = acc
            # compute the new object
            try:
                obj = UniprotMapping(acc, recover_pickle=True)
            except Exception as e:
                obj = None
                cache['warn'] = str(e)
            cache['obj'] = obj
        # map specific SAV to Pfam and calculate features
        try:
            if not isinstance(obj, UniprotMapping):
                raise Exception(cache['warn'])
            # check if wt aa is correct
            wt_aa = obj.sequence[pos-1]
            if aa1 != wt_aa:
                msg = 'Incorrect wt aa ({} instead of {})'.format(aa1, wt_aa)
                raise Exception(msg)
            # compute Evol features from Pfam domains
            PF_dict = obj.calcEvolProperties(resid=pos)
            if PF_dict is None:
                raise Exception('No Pfam domain found.')
            n = len(PF_dict)
            if n > 1:
                LOGGER.warn(f'Multiple ({n}) Pfam domains found. '
                            'Average values for Evol features will be used.')
            # compute Evol features from Pfam domains
            _features = _calcEvolFeatures(PF_dict, pos)
        except Exception as e:
            LOGGER.warn('Unable to compute Pfam features: {}'.format(e))
            _features = np.nan
        # store computed features
        features[indx] = _features
        # in the final iteration of the loop, save last pickle
        if count == num_SAVs and cache['obj'] is not None:
            cache['obj'].savePickle()
    LOGGER.report('SAVs have been mapped on Pfam domains and sequence '
                  'properties have been computed in %.1fs.', '_calcPfamFeats')
    if status_file:
        os.remove(status_file.name)
    return features
