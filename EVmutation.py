import numpy as np
from os import listdir
from os.path import splitext, join
from prody import LOGGER


EVMUT_FOLDER = '/home/lponzoni/Data/025-EVmutation/mutation_effects/'

EVMUT_FEATS = ['EVmut-DeltaE_epist', 'EVmut-DeltaE_indep',
               'EVmut-mut_aa_freq', 'EVmut-wt_aa_cons',]
# EVMUT_FEATS += ['EVmut-DeltaE_epist_rank',
#                 'EVmut-DeltaE_epist_rank2',
#                 'EVmut-DeltaE_epist_norm',
#                 'EVmut-DeltaE_epist_norm2',]


def _calcEVmutFeatures(data, wt_aa, mut_aa):
    n = len(data)
    assert n==20, 'Incorrect number of mutants: {} instead of 20.'.format(n)
    assert set(data[:,3])==set('ACDEFGHIKLMNPQRSTVWY'), \
           'Invalid set of mutants'
    index_wt  = np.where(data[:,3]==wt_aa )[0][0]
    index_mut = np.where(data[:,3]==mut_aa)[0][0]
    # extract precomputed EVmut scores for given mutant
    precomputed_scores = data[index_mut, 4:8]
#     # compute normalized/ranked version of DeltaE_epist:
#     # negative DeltaE --> 1 (deleterious effect)
#     # DeltaE = 0 (wt) --> ? (neutral effect)
#     # positive DeltaE --> 0 (neutral/benign effect) 
#     DeltaE_epist = np.array(data[:, 4], dtype=float)
#     ### normalization
#     norm = np.ptp(DeltaE_epist)
#     DeltaE_epist_norm  = - DeltaE_epist[index_mut]/norm #[wt=0]
#     DeltaE_epist_norm2 = -(DeltaE_epist[index_mut]-np.max(DeltaE_epist))/norm
#     ### ranking
#     array = -DeltaE_epist
#     order = array.argsort()
#     ranks = order.argsort()
#     DeltaE_epist_rank  =  ranks[index_mut]/19.
#     DeltaE_epist_rank2 = (ranks[index_mut]-ranks[index_wt])/19. #[wt=0]
#     # output
#     feats = np.empty(8)
#     feats[:4] = precomputed_scores
#     feats[4]  = DeltaE_epist_rank
#     feats[5]  = DeltaE_epist_rank2
#     feats[6]  = DeltaE_epist_norm
#     feats[7]  = DeltaE_epist_norm2
    # output
    feats = np.empty(4)
    feats[:] = precomputed_scores
    return feats


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
        err_msg = "EVmutation data not found for '{}'".format(SAV_txt)
        # find files containing given SAV coordinates
        match_files = find_matching_files(file_list, acc, pos)
        if len(match_files) == 0:
#           LOGGER.warn(err_msg)
            continue
        # recover data and average them if multiple values are found
        feat_arrays = []
        for fname in match_files:
            full_path = join(EVMUT_FOLDER, fname)
            data = np.genfromtxt(full_path, delimiter=';', dtype=str)
            # find entries for requested position
            pos_str = str(pos)
            data = data[data[:,1]==pos_str]
            if len(data) == 0:
                continue
            # extract/compute features
            try:
                arr = _calcEVmutFeatures(data, wt_aa, mut_aa)
            except Exception as e:
                LOGGER.warn(str(e))
                continue
            else:
                feat_arrays.append(arr)
        if feat_arrays:
            avg_array = np.mean(feat_arrays, axis=0)
            features[i] = tuple(avg_array)
        else:
#           LOGGER.warn(err_msg)
            continue
    LOGGER.report('EVmutation scores recovered in %.1fs.', '_EVmut')
    return features 


