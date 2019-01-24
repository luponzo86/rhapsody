import numpy as np
from glob import glob
from os.path import splitext, join
from prody import LOGGER

# extract precomputed EVmutation scores for given mutants
# NB:
# negative DeltaE_epist --> deleterious effect
# DeltaE_epist == 0     --> neutral effect (wild-type)
# positive DeltaE_epist --> neutral/benign effect


__all__ = ['EVMUT_FEATS', 'recoverEVmutFeatures']

EVMUT_FEATS = ['EVmut-DeltaE_epist', 'EVmut-DeltaE_indep',
               'EVmut-mut_aa_freq', 'EVmut-wt_aa_cons',]


def pathEVmutationFolder(folder=None):
    """Returns or sets path of local folder where EVmutation data are stored.
    To release the current folder, pass an invalid path, e.g.
    ``folder=''``.
    """
    if folder is None:
        folder = SETTINGS.get('EVmutation_local_folder')
        if folder:
            if isdir(folder):
                return folder
            else:
                LOGGER.warn('Local folder {} is not accessible.'
                            .format(repr(folder)))
    else:
        if isdir(folder):
            folder = abspath(folder)
            LOGGER.info('Local EVmutation folder is set: {}'.
                        format(repr(folder)))
            SETTINGS['EVmutation_local_folder'] = folder
            SETTINGS.save()
        else:
            current = SETTINGS.pop('EVmutation_local_folder')
            if current:
                LOGGER.info('EVmutation folder {0} is released.'
                            .format(repr(current)))
                SETTINGS.save()
            else:
                raise IOError('{} is not a valid path.'
                              .format(repr(folder)))


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
    EVmut_dir = pathEVmutationFolder()
    if EVmut_dir is None:
        raise RuntimeError('EVmutation folder not set')
    file_list = glob(join(EVmut_dir, '*.csv'))
    if not file_list:
        raise RuntimeError('EVmutation folder does not contain any csv files')
    for i, SAV in enumerate(SAVs):
        acc, pos, wt_aa, mut_aa, SAV_txt = SAV
#       LOGGER.info('Recovering EVmutation data for {}.'.format(SAV_txt))
        # find files containing given SAV coordinates
        match_files = find_matching_files(file_list, acc, pos)
        # recover data and average them if multiple values are found
        mutant = f'{wt_aa}{pos}{mut_aa}'
        data = []
        for fname in match_files:
            with open(join(EVmut_dir, fname), 'r') as f:
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


