import numpy as np
from prody import LOGGER
from Bio.SubsMat.MatrixInfo import blosum62
from .PolyPhen2 import *
from .PDB import *
from .Uniprot import *
from .EVmutation import *

__all__ = ['RHAPSODY_FEATS', 'calcPolyPhen2features', 'calcPDBfeatures',
           'calcPfamFeatures', 'calcBLOSUMfeatures',
           'buildFeatMatrix']


# list of all available features in RHAPSODY
RHAPSODY_FEATS = {
    'PolyPhen2': {'wt_PSIC', 'Delta_PSIC'},
    'PDB': set(PDB_FEATS),
    'Pfam': {'entropy', 'ranked_MI'},
    'BLOSUM': {'BLOSUM'},
    'EVmut': set(EVMUT_FEATS),
}
RHAPSODY_FEATS['all'] = set().union(*RHAPSODY_FEATS.values())


def calcPolyPhen2features(PolyPhen2output):
    # define a datatype for sequence-conservation features
    # extracted from the output of PolyPhen-2
    feat_dtype = np.dtype([('wt_PSIC', 'f'),
                           ('Delta_PSIC', 'f')])
    # import selected quantities from PolyPhen-2's output
    # into a structured array
    f_l = PolyPhen2output[['Score1', 'dScore']]
    f_t = [tuple(np.nan if x == '?' else x for x in l) for l in f_l]
    features = np.array(f_t, dtype=feat_dtype)
    LOGGER.info("Sequence-conservation features have been "
                "retrieved from PolyPhen-2's output.")
    return features


def calcPDBfeatures(mapped_SAVs, sel_feats=None, custom_PDB=None,
                    refresh=False):
    LOGGER.info('Computing structural and dynamical features '
                'from PDB structures...')
    LOGGER.timeit('_calcPDBFeats')
    if sel_feats is None:
        sel_feats = RHAPSODY_FEATS['PDB']
    # define a structured array for features computed from PDBs
    num_SAVs = len(mapped_SAVs)
    feat_dtype = np.dtype([(f, 'f') for f in sel_feats])
    features = np.zeros(num_SAVs, dtype=feat_dtype)
    # compute PDB features using PDBfeatures class
    if custom_PDB is None:
        # sort SAVs, so to group together those
        # belonging to the same PDB
        PDBID_list = [r[2][:4] if r[3] != 0 else '' for r in mapped_SAVs]
        sorting_map = np.argsort(PDBID_list)
    else:
        # no need to sort when using a custom PDB or PDBID
        sorting_map = range(num_SAVs)
    cache = {'PDBID': None, 'chain': None, 'obj': None}
    count = 0
    for indx, SAV in [(i, mapped_SAVs[i]) for i in sorting_map]:
        count += 1
        if SAV['PDB size'] == 0:
            # SAV could not be mapped to PDB
            _features = np.nan
            SAV_coords = SAV['SAV coords']
            LOGGER.info(f"[{count}/{num_SAVs}] SAV '{SAV_coords}' "
                        "couldn't be mapped to PDB")
        else:
            parsed_PDB_coords = SAV['PDB SAV coords'].split()
            PDBID, chID = parsed_PDB_coords[:2]
            resid = int(parsed_PDB_coords[2])
            LOGGER.info("[{}/{}] Analizing mutation site {}:{} {}..."
                        .format(count, num_SAVs, PDBID, chID, resid))
            # chID == "?" stands for "empty space"
            chID = " " if chID == "?" else chID
            if PDBID == cache['PDBID']:
                # use PDBfeatures instance from previous iteration
                obj = cache['obj']
            else:
                # save previous mapping to pickle
                if cache['obj'] is not None and custom_PDB is None:
                    cache['obj'].savePickle()
                cache['PDBID'] = PDBID
                cache['chain'] = chID
                try:
                    # instantiate new PDBfeatures object
                    if custom_PDB is None:
                        obj = PDBfeatures(PDBID, recover_pickle=not(refresh))
                    else:
                        obj = PDBfeatures(custom_PDB, recover_pickle=False)
                except Exception as e:
                    obj = None
                    LOGGER.warn(str(e))
                cache['obj'] = obj
            # compute PDB features
            if obj is None:
                _features = np.nan
            else:
                feat_dict = obj.calcSelFeatures(chID, resid=resid,
                                                sel_feats=sel_feats)
                # check for error messages
                _features = []
                for name in feat_dtype.names:
                    feat_array = feat_dict[name]
                    if isinstance(feat_array, str):
                        # print error message
                        LOGGER.warn('{}: {}'.format(name, feat_array))
                        _features.append(np.nan)
                    else:
                        # sometimes resid maps to multiple indices:
                        # we will only consider the first one
                        _features.append(feat_array[0])
                _features = tuple(_features)
        # store computed features
        features[indx] = _features
        # in the final iteration of the loop, save last pickle
        if count == num_SAVs and cache['obj'] is not None \
           and custom_PDB is None:
            cache['obj'].savePickle()
    LOGGER.report('PDB features have been computed in %.1fs.', '_calcPDBFeats')
    return features


def _calcEvolFeatures(PF_dict, pos):
    def calcNormRank(array, i):
        # returns rank in descending order
        order = array.argsort()
        ranks = order.argsort()
        return ranks[i]/len(ranks)
    entropy = 0.
    rankdMI = 0.
    rankdDI = 0.
    n = 0
    for ID, Pfam in PF_dict.items():
        if isinstance(Pfam['mapping'], dict):
            n += 1
            indx     = Pfam['mapping'][pos - 1]
            entropy += Pfam['entropy'][indx]
            rankdMI += calcNormRank(np.sum(Pfam['MutInfo'], axis=0), indx)
##          rankdDI += calcNormRank(np.sum(Pfam['DirInfo'], axis=0), indx)
    if n == 0:
        raise ValueError("Position couldn't be mapped on any Pfam domain")
    else:
        feats = (entropy/n, rankdMI/n)
##      feats = (entropy/n, rankdMI/n, rankdDI/n)
        return feats


def calcPfamFeatures(SAVs):
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
    # map to Pfam domains using UniprotMapping class
    cache = {'acc': None, 'obj': None, 'warn': ''}
    count = 0
    for indx, SAV in [(i, SAVs[i]) for i in sorting_map]:
        count += 1
        acc, pos, aa1, aa2 = SAV.split()
        pos = int(pos)
        LOGGER.info(f"[{count}/{num_SAVs}] Mapping SAV '{SAV}' to Pfam...")
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
                LOGGER.warn('Multiple ({}) Pfam domains found. '.format(n) + \
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
    LOGGER.report('SAVs have been mapped on Pfam domains and sequence ' + \
                  'properties have been computed in %.1fs.', '_calcPfamFeats')
    return features


def calcBLOSUMfeatures(SAV_coords):
    feat_dtype = np.dtype([('BLOSUM', 'f')])
    features = np.zeros(len(SAV_coords), dtype=feat_dtype)
    for i, SAV in enumerate(SAV_coords):
        aa1 = SAV.split()[2]
        aa2 = SAV.split()[3]
        features[i] = blosum62.get((aa1, aa2), blosum62.get((aa2, aa1)))
    return features


def buildFeatMatrix(featset, all_features):
    n_rows = len(all_features[0])
    n_cols = len(featset)
    feat_matrix = np.zeros((n_rows, n_cols))
    for j, featname in enumerate(featset):
        # find structured array containing a specific feature
        arrays = [a for a in all_features if featname in a.dtype.names]
        if len(arrays) == 0:
            raise RuntimeError('Invalid feature name: {}'.format(name))
        if len(arrays) >  1:
            LOGGER.warn('Multiple values for feature {}'.format(name))
        array = arrays[0]
        feat_matrix[:, j] = array[featname]
    return feat_matrix
