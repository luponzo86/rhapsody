import numpy as np
import pickle
from os.path import isfile
from prody import LOGGER, SETTINGS, Atomic, queryUniprot
from .settings import DEFAULT_FEATSETS
from .Uniprot import *
from .PolyPhen2 import *
from .EVmutation import *
from .calcFeatures import *

__all__ = ['Rhapsody', 'seqScanning', 'printSAVlist', 'mapSAVs2PDB',
           'calcPredictions']


class Rhapsody:
    """A class implementing the Rhapsody algorithm for pathogenicity
    prediction of human missense variants and that can also be used to
    compare results from other prediction tools, namely PolyPhen-2 and
    EVmutation.
    """

    def __init__(self):

        # masked NumPy array that will contain all info abut SAVs
        self.data = None
        self.data_dtype = np.dtype([
            # original Uniprot SAV coords, extracted from
            # PolyPhen-2's output or imported directly
            ('SAV coords', 'U50'),
            # "official" Uniprot SAV identifiers and corresponding
            # PDB coords (if found, otherwise message errors)
            ('unique SAV coords', 'U50'),
            ('PDB SAV coords', 'U100'),
            # number of residues in PDB structure (0 if not found)
            ('PDB size', 'i4'),
            # true labels provided by the user and
            # only needed when exporting training data
            ('true labels', 'i4'),
            # SAV found in the training dataset will be marked as
            # 'known_del' or 'known_neu', otherwise as 'new'
            ('training info', 'U12'),
            # predictions from main classifier
            ('main score', 'f4'),
            ('main path. prob.', 'f4'),
            ('main path. class', 'U12'),
            # predictions from auxiliary classifier
            ('aux score', 'f4'),
            ('aux path. prob.', 'f4'),
            ('aux path. class', 'U12'),
            # predictions from PolyPhen-2 and EVmutation
            ('PolyPhen-2 score', 'f4'),
            ('PolyPhen-2 path. class', 'U12'),
            ('EVmutation score', 'f4'),
            ('EVmutation path. class', 'U12')
        ])

        # filename of the classifier pickle
        self.classifier = None
        # tuple of feature names
        self.featSet = None
        # dictionary of properties of the trained classifier
        self.CVsummary = None
        # custom PDB structure used for PDB features calculation
        self.customPDB = None
        # structured array containing parsed PolyPhen-2 output
        self.PP2output = None
        # numpy array (num_SAVs)x(num_features)
        self.featMatrix = None

        # structured array containing predictions
        self.predictions    = None
        # structured arrays of auxiliary predictions from a subclassifier
        self.auxPreds       = None
        # original and auxiliary predictions combined
        self.mixPreds       = None

    def setCustomPDB(self, custom_PDB):
        if self.featSet is not None:
            if not RHAPSODY_FEATS['PDB'].intersection(self.featSet):
                LOGGER.warn('The given feature set does not require '
                            'a PDB structure.')
                return
        assert self.customPDB is None, 'Custom PDB structure already set.'
        assert isinstance(custom_PDB, (str, Atomic)), \
               'Please provide a PDBID, a filename or an Atomic instance.'
        self.customPDB = custom_PDB

    def setFeatSet(self, featset):
        assert self.featSet is None, 'Feature set already set.'
        if isinstance(featset, str):
            assert featset in ['all', 'full', 'reduced', 'EVmut']
            if featset == 'all':
                featset = sorted(list(RHAPSODY_FEATS['all']))
            else:
                featset == DEFAULT_FEATSETS[featset]
        assert all([f in RHAPSODY_FEATS['all'] for f in featset]), \
               'Invalid list of features'
        self.featSet = tuple(featset)

    def setTrueLabels(self, true_label_dict):
        # NB: PolyPhen-2 may reshuffle or discard entries, that's why it is
        # better to ask for a dictionary...
        assert self.data is not None, 'SAVs not set.'
        assert set(self.data['SAV coords']).issubset(
                   set(true_label_dict.keys())), 'Some labels are missing.'
        assert set(true_label_dict.values()).issubset(
                   {-1, 0, 1}), 'Invalid labels.'
        true_labels = [true_label_dict[s] for s in self.data['SAV coords']]
        self.data['true labels'] = tuple(true_labels)

    def queryPolyPhen2(self, x, scanning=False, filename='rhapsody-SAVs.txt'):
        assert self.PP2output is None, "PolyPhen-2's output already imported."
        SAV_file = ''
        if scanning:
            # 'x' is a string, e.g. 'P17516' or 'P17516 135'
            SAV_list = seqScanning(x)
            SAV_file = printSAVlist(SAV_list, filename)
        elif isinstance(x, str) and isfile(x):
            # 'x' is a filename, with line format 'P17516 135 G E'
            SAV_file = x
        else:
            # 'x' is a list, tuple or single string of SAV coordinates
            SAV_file = printSAVlist(x, filename)
        # submit query to PolyPhen-2
        try:
            PP2_output = queryPolyPhen2(SAV_file)
        except:
            err = ('Unable to get a response from PolyPhen-2. Please click '
                   '"Check Status" on the server homepage \n'
                   '(http://genetics.bwh.harvard.edu/pph2) \n'
                   'and try again when "Load" is "Low" and "Health" is 100%')
            raise RuntimeError(err)
        return self.importPolyPhen2output(PP2_output)

    def importPolyPhen2output(self, PP2output):
        assert self.PP2output is None, "PolyPhen-2's output already imported."
        assert self.data is None, 'SAVs already set.'
        self.PP2output = parsePP2output(PP2output)
        # store SAV coords
        nSAVs = len(self.PP2output)
        self.data = np.ma.masked_all(nSAVs, dtype=self.data_dtype)
        self.data['SAV coords'] = getSAVcoords(self.PP2output)['text']
        return self.PP2output

    def getUniprot2PDBmap(self, filename='rhapsody-Uniprot2PDB.txt',
                          header=True):
        """Maps each SAV to the corresponding resid in a PDB chain.
        """
        assert self.data is not None, "SAVs not set."
        if self.data['PDB SAV coords'].count() == 0:
            # compute mapping
            m = mapSAVs2PDB(self.data['SAV coords'], custom_PDB=self.customPDB)
            self.data['unique SAV coords'] = m['unique SAV coords']
            self.data['PDB SAV coords'] = m['PDB SAV coords']
            self.data['PDB size'] = m['PDB size']
        # print to file, if requested
        if filename is not None:
            with open(filename, 'w') as f:
                if header:
                    f.write('# SAV coords           '
                            'Uniprot coords         '
                            'PDB/ch/res/aa/size \n')
                for s in self.data:
                    orig_SAV = s['SAV coords'] + ','
                    U_coords = s['unique SAV coords'] + ','
                    if s['PDB size'] == 0:
                        PDB_coords = s['PDB SAV coords']
                    else:
                        PDB_coords = s['PDB SAV coords'] + ' ' + s['PDB size']
                    f.write(f'{orig_SAV:<22} {U_coords:<22} {PDB_coords:<}\n')
        return self.data[['SAV coords', 'unique SAV coords',
                          'PDB SAV coords', 'PDB size']]

    def calcFeatures(self, filename='rhapsody-features.txt'):
        if self.featMatrix is None:
            self.featMatrix = self._calcFeatMatrix()
        # print to file, if requested
        if filename is not None:
            h = ''
            for i, feat in enumerate(self.featSet):
                if len(feat) > 13:
                    feat = feat[:10] + '...'
                if i == 0:
                    h += f'{feat:>13}'
                else:
                    h += f' {feat:>15}'
            np.savetxt(filename, self.featMatrix, fmt='%15.3e', header=h)
        return self.featMatrix

    def _calcFeatMatrix(self):
        assert self.data is not None, 'SAVs not set.'
        assert self.featSet is not None, 'Feature set not set.'
        # list of structured arrays that will contain all computed features
        all_feats = []
        if RHAPSODY_FEATS['PP2'].intersection(self.featSet):
            # retrieve sequence-conservation features from PolyPhen-2's output
            assert self.PP2output is not None, \
                   "Please import PolyPhen-2's output first."
            f = calcPP2features(self.PP2output)
            all_feats.append(f)
        sel_PDBfeats = RHAPSODY_FEATS['PDB'].intersection(self.featSet)
        if sel_PDBfeats:
            # map SAVs to PDB structures
            Uniprot2PDBmap = self.getUniprot2PDBmap()
            # compute structural and dynamical features from a PDB structure
            f = calcPDBfeatures(Uniprot2PDBmap, sel_feats=sel_PDBfeats,
                                custom_PDB=self.customPDB)
            all_feats.append(f)
        if RHAPSODY_FEATS['BLOSUM'].intersection(self.featSet):
            # retrieve BLOSUM values
            f = calcBLOSUMfeatures(self.data['SAV coords'])
            all_feats.append(f)
        if RHAPSODY_FEATS['Pfam'].intersection(self.featSet):
            # compute sequence properties from Pfam domains
            f = calcPfamFeatures(self.data['SAV coords'])
            all_feats.append(f)
        if RHAPSODY_FEATS['EVmut'].intersection(self.featSet):
            # recover EVmutation data
            f = recoverEVmutFeatures(self.data['SAV coords'])
            all_feats.append(f)
        # build matrix of selected features
        return buildFeatMatrix(self.featSet, all_feats)

    def exportTrainingData(self):
        assert self.data is not None, 'SAVs not set.'
        assert self.data['true labels'].count() > 0, 'True labels not set.'
        if self.featMatrix is None:
            self.featMatrix = self._calcFeatMatrix()
        dt = np.dtype([('SAV_coords', '<U50'), ('Uniprot2PDB', '<U100'),
                       ('PDB_length', '<i2'), ('true_label', '<i2')] +
                      [(f, '<f4') for f in self.featSet])
        num_SAVs = len(self.data)
        trainData = np.empty(num_SAVs, dtype=dt)
        trainData['SAV_coords'] = self.data['SAV coords']
        if self.data['PDB SAV coords'].count() > 0:
            trainData['Uniprot2PDB'] = self.data['PDB SAV coords']
            trainData['PDB_length'] = self.data['PDB size']
        trainData['true_label'] = self.data['true labels']
        for i, f in enumerate(self.featSet):
            trainData[f] = self.featMatrix[:, i]
        return trainData

    def importClassifier(self, classifier, force_env=None):
        assert self.classifier is None, 'Classifier already set.'
        assert isfile(classifier), 'Please provide a valid filename.'
        assert force_env in [None, 'chain', 'reduced', 'sliced']
        clsf_dict = pickle.load(open(classifier, 'rb'))
        self.classifier = classifier
        featset = clsf_dict['features']
        LOGGER.info(f"Imported feature set: '{featset[0]}'")
        for f in featset[1:]:
            LOGGER.info(' '*22 + f"'{f}'")
        if force_env is not None:
            # force a given ENM environment model
            for i, f in enumerate(featset):
                if f in RHAPSODY_FEATS['PDB'] and (f.startswith('ANM')
                                                   or f.startswith('GNM')):
                    old_env = f.split('-')[-1]
                    featset[i] = f.replace(old_env, force_env)
            LOGGER.info(f"Modified feature set: '{featset[0]}'")
            for f in featset[1:]:
                LOGGER.info(' '*22 + f"'{f}'")
        self.setFeatSet(featset)
        self.CVsummary = clsf_dict['CV summary']
        del clsf_dict

    def calcPredictions(self):
        assert self.predictions is None, 'Predictions already computed.'
        assert self.classifier is not None, 'Classifier not set.'
        assert self.featMatrix is not None, 'Features not computed.'
        p = calcPredictions(self.featMatrix, self.classifier,
                            SAV_coords=self.data['SAV coords'])
        self.predictions = p
        return self.predictions

    def calcAuxPredictions(self, aux_clsf, force_env=None):
        assert self.predictions is not None, 'Primary predictions not found.'
        assert self.featMatrix  is not None, 'Features not computed.'
        assert force_env in [None, 'chain', 'reduced', 'sliced']
        # import feature subset
        clsf_dict = pickle.load(open(aux_clsf, 'rb'))
        LOGGER.info('Auxiliary Random Forest classifier imported.')
        feat_subset = clsf_dict['features']
        if force_env is not None:
            # force a given ENM environment model
            for i, f in enumerate(feat_subset):
                if f in RHAPSODY_FEATS['PDB'] and \
                   (f.startswith('ANM') or f.startswith('GNM')):
                    old_env = f.split('-')[-1]
                    feat_subset[i] = f.replace(old_env, force_env)
        assert all(f in self.featSet for f in feat_subset), \
               'The new set of features must be a subset of the original one.'
        # reduce original feature matrix
        sel = [i for i,f in enumerate(self.featSet) if f in feat_subset]
        fm = self.featMatrix[:, sel]
        p_a = calcPredictions(fm, clsf_dict,
                              SAV_coords=self.data['SAV coords'])
        if p_a is None:
            LOGGER.warn('No additional predictions.')
            return None
        self.auxPreds = p_a
        p_o = self.predictions
        self.mixPreds = np.where(np.isnan(p_o['score']), p_a, p_o)
        return self.auxPreds, self.mixPreds

    def calcEVmutPredictions(self):
        if self.data['EVmutation score'].count() == 0:
            EVmut_feats = recoverEVmutFeatures(self.data['SAV coords'])
            EVmut_score = EVmut_feats['EVmut-DeltaE_epist']
            c = -SETTINGS.get('EVmutation_metrics')['optimal cutoff']
            EVmut_class = np.where(EVmut_score < c, 'deleterious', 'neutral')
            self.data['EVmutation score'] = EVmut_score
            self.data['EVmutation path. class'] = EVmut_class
        return self.data[['SAV coords', 'EVmutation score',
                          'EVmutation path. class']]

    def printPredictions(self, format="auto", header=True,
                         filename='rhapsody-predictions.txt'):
        assert format in ["auto", "full", "aux", "mixed", "both"]
        assert isinstance(filename, str)
        assert isinstance(header, bool)
        if format != "both":
            # select what predictions to print
            if format == "auto":
                preds = self.mixPreds
                if self.mixPreds is None:
                    preds = self.predictions
            elif format == "full":
                preds = self.predictions
            elif format == "aux":
                preds = self.auxPreds
            elif format == "mixed":
                preds = self.mixPreds
            # check if predictions are computed
            if preds is None:
                raise RuntimeError('Predictions not computed')
            # print
            with open(filename, 'w') as f:
                h = '# SAV coords           score   prob    class        info\n'
                if header:
                    f.write(h)
                for SAV, p in zip(self.data['SAV coords'], preds):
                    p_cols = '{:<5.3f}   {:<5.3f}   {:12s} {:12s}'.format(*p)
                    f.write(f'{SAV:22} {p_cols} \n')
        else:
            if self.mixPreds is None:
                raise RuntimeError('Auxiliary predictions not computed')
            # print both full and aux predictions in a more detailed format
            with open(filename, 'w') as f:
                h  = '# SAV coords           '
                h += 'full-classifier predictions        '
                h += 'reduced-classifier predictions \n'
                if header:
                    f.write(h)
                SAVs = self.data['SAV coords']
                p_o  = self.predictions
                p_a  = self.auxPreds
                p_m  = self.mixPreds
                for SAV, t_o, t_a, t_m in zip(SAVs, p_o, p_a, p_m):
                    f.write(f'{SAV:22} ')
                    f.write(f'{t_m[0]:<5.3f}  {t_m[1]:<5.3f}  {t_m[2]:12s}')
                    if np.isnan(t_o['score']) and not np.isnan(t_a['score']):
                        f.write('   <--   ')
                    else:
                        f.write('   x--   ')
                    f.write(f'{t_a[0]:<5.3f}  {t_a[1]:<5.3f}  {t_a[2]:12s}\n')

    def savePickle(self, filename='rhapsody-pickle.pkl'):
        f = pickle.dump(self, open(filename, "wb"))
        return f


#############################################################################


def seqScanning(Uniprot_coord):
    '''Returns a list of SAVs. If the string 'Uniprot_coord' is just a Uniprot ID,
    the list will contain all possible amino acid substitutions at all positions
    in the sequence. If 'Uniprot_coord' also includes a specific position, the list
    will only contain all possible amino acid variants at that position.
    '''
    assert isinstance(Uniprot_coord, str), "Must be a string."
    coord = Uniprot_coord.strip().split()
    assert len(coord)<3, "Invalid format. Examples: 'Q9BW27' or 'Q9BW27 10'."
    Uniprot_record = queryUniprot(coord[0])
    sequence = Uniprot_record['sequence   0'].replace("\n","")
    if len(coord) == 1:
        positions = range(len(sequence))
    else:
        positions = [int(coord[1]) - 1]
    SAV_list = []
    acc = coord[0]
    for i in positions:
        wt_aa = sequence[i]
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            if aa == wt_aa:
                continue
            s = ' '.join([acc, str(i+1), wt_aa, aa])
            SAV_list.append(s)
    return SAV_list


def printSAVlist(input_SAVs, filename):
    if isinstance(input_SAVs, str):
        input_SAVs = [input_SAVs,]
    with open(filename, 'w', 1) as f:
        for i, line in enumerate(input_SAVs):
            m = f'error in SAV {i}: '
            assert isinstance(line, str), f'{m} not a string'
            assert len(line) < 25, f'{m} too many characters'
            print(line, file=f)
    return filename


def mapSAVs2PDB(SAV_coords, custom_PDB=None):
    LOGGER.info('Mapping SAVs to PDB structures...')
    LOGGER.timeit('_map2PDB')
    # sort SAVs, so to group together those
    # with identical accession number
    accs = [s.split()[0] for s in SAV_coords]
    sorting_map = np.argsort(accs)
    # define a structured array
    PDBmap_dtype = np.dtype([('orig. SAV coords', 'U25'),
                             ('unique SAV coords', 'U25'),
                             ('PDB SAV coords', 'U100'),
                             ('PDB size', 'i')])
    num_SAVs = len(SAV_coords)
    mapped_SAVs = np.zeros(num_SAVs, dtype=PDBmap_dtype)
    # map to PDB using Uniprot class
    cache = {'acc': None, 'obj': None}
    count = 0
    for indx, SAV in [(i, SAV_coords[i]) for i in sorting_map]:
        count += 1
        acc, pos, aa1, aa2 = SAV.split()
        pos = int(pos)
        LOGGER.info(f"[{count}/{num_SAVs}] Mapping SAV '{SAV}' to PDB...")
        # map Uniprot to PDB chains
        if acc == cache['acc']:
            # use mapping from previous iteration
            U2P_map = cache['obj']
        else:
            # save previous mapping
            if isinstance(cache['obj'], UniprotMapping):
                cache['obj'].savePickle()
            cache['acc'] = acc
            # compute the new mapping
            try:
                U2P_map = UniprotMapping(acc, recover_pickle=True)
                if custom_PDB is not None:
                    LOGGER.info('Aligning Uniprot sequence to custom PDB...')
                    U2P_map.alignCustomPDB(custom_PDB, 'all')
            except Exception as e:
                U2P_map = str(e)
            cache['obj'] = U2P_map
        # map specific SAV
        try:
            if isinstance(U2P_map, str):
                raise RuntimeError(U2P_map)
            # check wt aa
            if not 0 < pos <= len(U2P_map.sequence):
                raise ValueError('Index out of range')
            wt_aa = U2P_map.sequence[pos-1]
            if aa1 != wt_aa:
                raise ValueError(f'Incorrect wt aa: {aa1} instead of {wt_aa}')
            # map to PDB. Format: [('2DZF', 'A', 150, 'N', 335)]
            if custom_PDB is None:
                r = U2P_map.mapSingleResidue(pos, check_aa=True)
            else:
                r = U2P_map.mapSingleRes2CustomPDBs(pos, check_aa=True)
            if len(r) == 0:
                raise RuntimeError('Unable to map SAV to PDB')
            else:
                PDBID, chID, resid, aa, PDB_size = r[0]
                # NB: check for blank "chain" field
                if chID.strip() == '':
                    chID = '?'
                res_map = f'{PDBID} {chID} {resid} {aa}'
        except Exception as e:
                res_map = str(e)
                PDB_size = 0
        # store SAVs mapped on PDB chains and unique Uniprot coordinates
        if isinstance(U2P_map, str):
            uniq_coords = U2P_map
        else:
            uniq_coords = f'{U2P_map.uniq_acc} {pos} {aa1} {aa2}'
        mapped_SAVs[indx] = (SAV, uniq_coords, res_map, PDB_size)
    # save last pickle
    if isinstance(cache['obj'], UniprotMapping):
        cache['obj'].savePickle()
    LOGGER.report('SAVs have been mapped to PDB in %.1fs.', '_map2PDB')
    return mapped_SAVs


def calcPredictions(feat_matrix, clsf, SAV_coords=None):
    assert SAV_coords is None or len(SAV_coords)==len(feat_matrix)

    # import classifier and other info
    if isinstance(clsf, dict):
        clsf_dict = clsf
    else:
        LOGGER.timeit('_import_clsf')
        clsf_dict = pickle.load(open(clsf, 'rb'))
        LOGGER.report('Random Forest classifier imported in %.1fs.',
                      '_import_clsf')
    classifier = clsf_dict['trained RF']
    opt_cutoff = clsf_dict['CV summary']['optimal cutoff']
    path_curve = clsf_dict['CV summary']['path. probability']
    train_data = clsf_dict['training dataset']

    LOGGER.timeit('_preds')

    # define a structured array for storing predictions
    pred_dtype = np.dtype([('score', 'f'),
                           ('path. probability', 'f'),
                           ('path. class', 'U12'),
                           ('training info', 'U12')])
    predictions = np.zeros(len(feat_matrix), dtype=pred_dtype)

    # select rows where all features are well-defined
    sel_rows = [i for i,r in enumerate(feat_matrix) if all(~np.isnan(r))]
    n_pred = len(sel_rows)
    if n_pred == 0:
        LOGGER.warning('No predictions could be computed.')
        proba = None
    else:
        # compute predictions
        sliced_feat_matrix = feat_matrix[sel_rows]
        proba = classifier.predict_proba(sliced_feat_matrix)

    # output
    J, err_bar = opt_cutoff
    Jminus = J - err_bar
    Jplus  = J + err_bar
    k = 0
    for i in range(len(feat_matrix)):
        # determine SAV status
        if SAV_coords is None:
            SAV_status = '?'
        elif SAV_coords[i] in train_data['del. SAVs']:
            SAV_status = 'known_del'
        elif SAV_coords[i] in train_data['neu. SAVs']:
            SAV_status = 'known_neu'
        else:
            SAV_status = 'new'
        # determine pathogenicity prob. and class
        if i not in sel_rows:
            predictions[i] = (np.nan, np.nan, '?', SAV_status)
        else:
            # retrieve score returned by RF
            score = proba[k, 1]
            # assign pathogenicity probability by interpolating
            # the pathogenicity profile computed during CV
            path_prob = np.interp(score, path_curve[0], path_curve[1])
            # assign class of pathogenicity based on Youden's cutoff
            if score > Jplus:
                path_class = "deleterious"
            elif score > J:
                path_class = "prob.delet."
            elif score >= Jminus:
                path_class = "prob.neutral"
            else:
                path_class = "neutral"
            # store values
            predictions[i] = (score, path_prob, path_class, SAV_status)
            k = k + 1
    LOGGER.report(f'{n_pred} predictions computed in %.1fs.', '_preds')

    return predictions
