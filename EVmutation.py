import numpy as np
from os import listdir
from os.path import splitext, join
from prody import LOGGER

# extract precomputed EVmutation scores for given mutants
# NB:
# negative DeltaE_epist --> deleterious effect
# DeltaE_epist == 0     --> neutral effect (wild-type)
# positive DeltaE_epist --> neutral/benign effect 

__all__ = ['recoverEVmutFeatures']


EVMUT_FOLDER = '/home/lponzoni/Data/025-EVmutation/mutation_effects/'

EVMUT_FEATS = ['EVmut-DeltaE_epist', 'EVmut-DeltaE_indep',
               'EVmut-mut_aa_freq', 'EVmut-wt_aa_cons',]


def recoverEVmutFeatures(SAVs):
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
    file_list = listdir(EVMUT_FOLDER)
    for i, SAV in enumerate(SAVs):
        acc, pos, wt_aa, mut_aa, SAV_txt = SAV
#       LOGGER.info('Recovering EVmutation data for {}.'.format(SAV_txt))
        # find files containing given SAV coordinates
        match_files = find_matching_files(file_list, acc, pos)
        # recover data and average them if multiple values are found
        mutant = f'{wt_aa}{pos}{mut_aa}'
        data = []
        for fname in match_files:
            with open(join(EVMUT_FOLDER, fname), 'r') as f:
                for line in f:
                    if line.startswith(mutant):
                        l = line.strip().split(';')[4:8]
                        data.append(l)
                        break
        data = np.array(data, dtype=float)
        if len(data) == 0:
#           LOGGER.warn(f"EVmutation data not found for '{SAV_txt}'")
            continue
        else:
            features[i] = tuple(np.mean(data, axis=0))

    LOGGER.report('EVmutation scores recovered in %.1fs.', '_EVmut')
    return features 


