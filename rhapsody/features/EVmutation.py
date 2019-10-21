# -*- coding: utf-8 -*-
"""This module defines a function for deriving coevolutionary features
from precomputed EVmutation scores."""

import numpy as np
from glob import glob
from os.path import splitext, join, basename
from prody import SETTINGS, LOGGER

# extract precomputed EVmutation scores for given mutants
# NB:
# negative DeltaE_epist --> deleterious effect
# DeltaE_epist == 0     --> neutral effect (wild-type)
# positive DeltaE_epist --> neutral/benign effect


__all__ = ['EVMUT_FEATS', 'recoverEVmutFeatures']

EVMUT_FEATS = ['EVmut-DeltaE_epist', 'EVmut-DeltaE_indep',
               'EVmut-mut_aa_freq', 'EVmut-wt_aa_cons']
"""List of features derived from EVmutation database of precomputed
coevolution-based scores."""


def recoverEVmutFeatures(SAVs):
    """Compute EVmutation features by fetching precomputed scores from the
    downloaded local folder. If multiple values are found for a given variant,
    the average will be taken.

    :arg SAVs: list of SAV coordinates, e.g. ``'P17516 135 G E'``.
    :type SAVs: list or tuple of strings
    :return: an array of EVmutation features for each SAV
    :rtype: NumPy structured array
    """
    LOGGER.timeit('_EVmut')
    LOGGER.info('Recovering EVmutation data...')

    def find_matching_files(file_list, acc, pos):
        match_files = []
        for fname in [f for f in file_list if f.startswith(acc)]:
            basename = splitext(fname)[0]
            res_range = basename.split("_")[-1]
            res_i = int(res_range.split("-")[0])
            res_f = int(res_range.split("-")[1])
            if res_i <= int(pos) <= res_f:
                match_files.append(fname)
        return match_files

    feat_dtype = np.dtype([(f, 'f') for f in EVMUT_FEATS])
    features = np.zeros(len(SAVs), dtype=feat_dtype)
    features[:] = np.nan

    # recover EVmutation data
    EVmut_dir = SETTINGS.get('EVmutation_local_folder')
    if EVmut_dir is None:
        raise RuntimeError('EVmutation folder not set')
    file_list = [basename(f) for f in glob(join(EVmut_dir, '*.csv'))]
    if not file_list:
        raise RuntimeError('EVmutation folder does not contain any .csv files')
    for i, SAV in enumerate(SAVs):
        acc, pos, wt_aa, mut_aa = SAV.split()
        pos = int(pos)
#       LOGGER.info('Recovering EVmutation data for {}.'.format(SAV))
        # find files containing given SAV coordinates
        match_files = find_matching_files(file_list, acc, pos)
        # recover data and average them if multiple values are found
        mutant = f'{wt_aa}{pos}{mut_aa}'
        data = []
        for fname in match_files:
            with open(join(EVmut_dir, fname), 'r') as f:
                for line in f:
                    if line.startswith(mutant):
                        ll = line.strip().split(';')[4:8]
                        data.append(ll)
                        break
        data = np.array(data, dtype=float)
        if len(data) == 0:
            # LOGGER.warn(f"EVmutation data not found for '{SAV}'")
            continue
        else:
            features[i] = tuple(np.mean(data, axis=0))

    LOGGER.report('EVmutation scores recovered in %.1fs.', '_EVmut')
    return features


def calcEVmutPathClasses(EVmut_score):
    c = -SETTINGS.get('EVmutation_metrics')['optimal cutoff']
    EVmut_class = np.where(EVmut_score < c, 'deleterious', 'neutral')
    EVmut_class[np.isnan(EVmut_score)] = '?'
    return EVmut_class
