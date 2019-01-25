import numpy as np
import pickle
from os.path import abspath, isdir, isfile
from prody import LOGGER, SETTINGS, Atomic
from prody import queryUniprot
from .Uniprot import *
from .PolyPhen2 import *
from .EVmutation import *
from .calcFeatures import *

__all__ = ['Rhapsody', 'pathRhapsodyFolder', 'seqScanning',
           'printSAVlist', 'mapSAVs2PDB', 'calcPredictions']


class Rhapsody:
    """A class for running calculations and handling results
    from RHAPSODY.
    """
    def __init__(self):
        # filename of the classifier pickle
        self.classifier     = None
        # tuple of feature names
        self.featSet        = None
        # dictionary of properties of the trained classifier
        self.CVsummary      = None
        # custom PDB structure used for PDB features calculation
        self.customPDB      = None
        # tuple of dicts containing parsed PolyPhen-2 output
        self.PP2output      = None
        # structured array containing EVmutation data
        self.EVmutFeats     = None
        # structured array containing original Uniprot SAV coords,
        # extracted from PolyPhen-2's output or imported directly
        self.SAVcoords      = None
        # tuple of tuples containing:
        # * original Uniprot SAV coords (str)
        # * unique Uniprot SAV coords, with format: (acc, pos, wt_aa, mut_aa)
        # * PDB coords, with format: (PDBID, chain, resid, wt_aa, PDB_length)
        # If an error occurs, unique Uniprot coords and/or PDB coords may
        # be replaced by a string describing the error.
        self.Uniprot2PDBmap = None
        # numpy array (num_SAVS)x(num_features)
        self.featMatrix     = None
        # structured array containing predictions
        self.predictions    = None
        # structured arrays of auxiliary predictions from a subclassifier
        self.auxPreds       = None
        # original and auxiliary predictions combined
        self.mixPreds       = None

    def importClassifier(self, classifier):
        assert self.classifier is None, 'Classifier already set.'
        assert isfile(classifier), 'Please provide a valid filename.'
        clsf_dict = pickle.load(open(classifier, 'rb'))
        self.classifier = classifier
        featset = clsf_dict['features']
        self.setFeatSet(featset)
        self.CVsummary = clsf_dict['CV summary']
        del clsf_dict
        LOGGER.info("Imported feature set: '{}'".format(featset[0]))
        for f in featset[1:]:
            LOGGER.info(' '*22 + "'{}'".format(f))

    def setCustomPDB(self, custom_PDB):
        if self.featSet is not None:
            if not RHAPSODY_FEATS['PDB'].intersection(self.featSet):
                LOGGER.warn('The given feature set does not require ' +\
                'a PDB structure.')
                return
        assert self.customPDB is None, 'Custom PDB structure already set.'
        assert isinstance(custom_PDB, (str, Atomic)), \
               'Please provide a PDBID, a filename or an Atomic instance.'
        self.customPDB = custom_PDB

    def setFeatSet(self, featset):
        assert self.featSet is None, 'Feature set already set.'
        assert isinstance(featset, (tuple, list)), \
               'Please provide a tuple or a list.'
        self.featSet = tuple(featset)

    def queryPolyPhen2(self, x, scanning=False, filename='rhapsody-SAVs.txt'):
        assert self.PP2output is None, "PolyPhen-2's output already imported."
        assert isinstance(x, (str, list, tuple))
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
        PP2_output = queryPolyPhen2(SAV_file)
        self.importPolyPhen2output(PP2_output)

    def importPolyPhen2output(self, PP2output):
        assert self.PP2output is None, "PolyPhen-2's output already imported."
        self.PP2output = parsePP2output(PP2output)
        self.SAVcoords = getSAVcoords(self.PP2output)
        return self.PP2output

    def calcEVmutationFeats(self):
        if self.EVmutFeats is None:
            self.EVmutFeats = recoverEVmutFeatures(self.SAVcoords)
        return self.EVmutFeats

    def getUniprot2PDBmap(self, filename='rhapsody-Uniprot2PDB.txt'):
        """Maps each SAV to the corresponding resid in a PDB chain.
        The format is: (PDBID, chainID, resid, wild-type aa, length).
        """
        assert self.SAVcoords is not None, "Uniprot coordinates not set."
        if self.Uniprot2PDBmap is None:
            m = mapSAVs2PDB(self.SAVcoords, custom_PDB=self.customPDB)
            self.Uniprot2PDBmap = m
        # print to file, if requested
        if filename is not None:
            with open(filename, 'w') as f:
                for t in self.Uniprot2PDBmap:
                    orig_SAV = '{},'.format(t[0])
                    if isinstance(t[1], tuple):
                        U_coords = '{} {} {} {},'.format(*t[1])
                    else:
                        U_coords = '{},'.format(t[1])
                    if isinstance(t[2], tuple):
                        PDB_coords = '{} {} {} {} {}'.format(*t[2])
                    else:
                        PDB_coords = '{}'.format(t[2])
                    o = (orig_SAV, U_coords, PDB_coords)
                    f.write('{:<22} {:<22} {:<} \n'.format(*o))
        return self.Uniprot2PDBmap

    def calcFeatures(self, filename='rhapsody-features.txt'):
        if self.featMatrix is None:
            self.featMatrix = self._calcFeatMatrix()
        # print to file, if requested
        if filename is not None:
            np.savetxt(filename, self.featMatrix, fmt='%10.3e')
        return self.featMatrix

    def _calcFeatMatrix(self):
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
            mapped_SAVs = tuple(t[2] for t in Uniprot2PDBmap)
            # compute structural and dynamical features from a PDB structure
            f = calcPDBfeatures(mapped_SAVs, sel_feats=sel_PDBfeats,
                                custom_PDB=self.customPDB)
            all_feats.append(f)
        if RHAPSODY_FEATS['BLOSUM'].intersection(self.featSet):
            assert self.SAVcoords is not None, 'Uniprot coords not set.'
            # retrieve BLOSUM values
            f = calcBLOSUMfeatures(self.SAVcoords)
            all_feats.append(f)
        if RHAPSODY_FEATS['Pfam'].intersection(self.featSet):
            assert self.SAVcoords is not None, 'Uniprot coords not set.'
            # compute sequence properties from Pfam domains
            f = calcPfamFeatures(self.SAVcoords)
            all_feats.append(f)
        if RHAPSODY_FEATS['EVmut'].intersection(self.featSet):
            assert self.SAVcoords is not None, 'Uniprot coords not set.'
            # recover EVmutation data
            f = recoverEVmutFeatures(self.SAVcoords)
            all_feats.append(f)
        # build matrix of selected features
        return buildFeatMatrix(self.featSet, all_feats)

    def calcPredictions(self):
        assert self.predictions is None, 'Predictions already computed.'
        assert self.classifier is not None, 'Classifier not set.'
        assert self.featMatrix is not None, 'Features not computed.'
        p = calcPredictions(self.featMatrix, self.classifier,
                            SAV_coords=self.SAVcoords['text'])
        self.predictions = p
        return self.predictions

    def calcAuxPredictions(self, aux_clsf):
        assert self.predictions is not None, 'Primary predictions not found.'
        assert self.featMatrix  is not None, 'Features not computed.'
        # import feature subset
        clsf_dict = pickle.load(open(aux_clsf, 'rb'))
        LOGGER.info('Auxiliary Random Forest classifier imported.')
        feat_subset = tuple(clsf_dict['features'])
        assert all(f in self.featSet for f in feat_subset), \
               'The new set of features must be a subset of the original one.'
        # reduce original feature matrix
        sel = [i for i,f in enumerate(self.featSet) if f in feat_subset]
        fm = self.featMatrix[:, sel]
        p_a = calcPredictions(fm, clsf_dict, SAV_coords=self.SAVcoords['text'])
        if p_a is None:
            LOGGER.warn('No additional predictions.')
            return None
        self.auxPreds = p_a
        p_o = self.predictions
        self.mixPreds = np.where(np.isnan(p_o['score']), p_a, p_o)
        return self.auxPreds, self.mixPreds

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
                for SAV, p in zip(self.SAVcoords['text'], preds):
                    p_cols = '{:<5.3f}   {:<5.3f}   {:12s} {:12s}'.format(*p)
                    f.write(f'{SAV:22} {p_cols} \n')
        else:
            if self.mixPreds is None:
                raise RuntimeError('Auxiliary predictions not computed')
            # print both full and aux predictions in a more detailed format
            with open(filename, 'w') as f:
                h  = '# SAV coords           '
                h += 'final predictions                auxiliary predictions\n'
                if header:
                    f.write(h)
                SAVs = self.SAVcoords['text']
                p_o  = self.predictions
                p_a  = self.auxPreds
                p_m  = self.mixPreds
                for SAV, t_o, t_a, t_m in zip(SAVs, p_o, p_a, p_m):
                    f.write(f'{SAV:22} ')
                    f.write(f'{t_m[0]:<5.3f}  {t_m[1]:<5.3f}  {t_m[2]:12s}')
                    if np.isnan(t_o['score']) and not np.isnan(t_a['score']):
                        f.write('  <--  ')
                    else:
                        f.write('  x--  ')
                    f.write(f'{t_a[0]:<5.3f}  {t_a[1]:<5.3f}  {t_a[2]:12s}\n')



    def savePickle(self, filename='rhapsody-pickle.pkl'):
        f = pickle.dump(self, open(filename, "wb"))
        return f


#############################################################################


def pathRhapsodyFolder(folder=None):
    """Returns or sets path of local folder where files and pickles necessary
    to run Rhapsody will be stored. To release the current folder, pass an
    invalid path, e.g. ``folder=''``.
    """
    if folder is None:
        folder = SETTINGS.get('rhapsody_local_folder')
        if folder:
            if isdir(folder):
                return folder
            else:
                LOGGER.warn('Local folder {} is not accessible.'
                            .format(repr(folder)))
    else:
        if isdir(folder):
            folder = abspath(folder)
            LOGGER.info('Local Rhapsody folder is set: {}'.
                        format(repr(folder)))
            SETTINGS['rhapsody_local_folder'] = folder
            SETTINGS.save()
        else:
            current = SETTINGS.pop('rhapsody_local_folder')
            if current:
                LOGGER.info('Rhapsody folder {0} is released.'
                            .format(repr(current)))
                SETTINGS.save()
            else:
                raise IOError('{} is not a valid path.'
                              .format(repr(folder)))


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
    assert isinstance(input_SAVs, (str, list, tuple))
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
    sorting_map = np.argsort(SAV_coords['acc'])
    # map to PDB using Uniprot class
    num_SAVs = len(SAV_coords)
    mapped_SAVs = [None]*num_SAVs
    cache = {'acc': None, 'obj': None}
    count = 0
    for indx, SAV in [(i, SAV_coords[i]) for i in sorting_map]:
        count += 1
        acc, pos, aa1, aa2, SAV_str = SAV
        LOGGER.info("[{}/{}] Mapping SAV '{}' to PDB..."
                    .format(count, num_SAVs, SAV_str))
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
                raise ValueError('Incorrect wt aa: ' + \
                                 '{} instead of {}'.format(aa1, wt_aa))
            # map to PDB. Format: [('2DZF', 'A', 150, 'N', 335)]
            if custom_PDB is None:
                r = U2P_map.mapSingleResidue(pos, check_aa=True)
            else:
                r = U2P_map.mapSingleRes2CustomPDBs(pos, check_aa=True)
            if len(r) == 0:
                raise RuntimeError('Unable to map SAV to PDB')
            else:
                res_map = r[0]
        except Exception as e:
                res_map = str(e)
        # store SAVs mapped on PDB chains and unique Uniprot coordinates
        if isinstance(U2P_map, str):
            uniq_coords = U2P_map
        else:
            uniq_coords = (U2P_map.uniq_acc, pos, aa1, aa2)
        mapped_SAVs[indx] = (SAV_str, uniq_coords, res_map)
        # in the final iteration of the loop, save last pickle
        if count == num_SAVs and cache['obj'] is not None:
            cache['obj'].savePickle()
    LOGGER.report('SAVs have been mapped to PDB in %.1fs.', '_map2PDB')
    return tuple(mapped_SAVs)


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
                           ('path. probability' , 'f'),
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

    # check if SAVs are found in the training dataset
    known_SAVs = {}
    if SAV_coords is not None:
        f = lambda x: 'known_del' if x==1 else 'known_neu'
        known_SAVs = {l['SAV_coords']: f(l['true_label']) for l in train_data}

    # output
    J, err_bar = opt_cutoff
    Jminus = J - err_bar
    Jplus  = J + err_bar
    k = 0
    for i in range(len(feat_matrix)):
        # determine SAV status
        if SAV_coords is None:
            SAV_status = '?'
        else:
            SAV_status = known_SAVs.get(SAV_coords[i], 'new')
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
    LOGGER.report('{} predictions computed in %.1fs.'.format(n_pred),
                  '_preds')

    return predictions
