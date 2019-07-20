import numpy as np
import warnings
import pickle
from os.path import isfile
from prody import LOGGER, SETTINGS, Atomic, queryUniprot
from .settings import DEFAULT_FEATSETS
from .Uniprot import *
from .PolyPhen2 import queryPolyPhen2, parsePolyPhen2output, getSAVcoords
from .EVmutation import recoverEVmutFeatures
from .calcFeatures import RHAPSODY_FEATS
from .calcFeatures import calcPolyPhen2features, calcPfamFeatures
from .calcFeatures import calcPDBfeatures, calcBLOSUMfeatures
from .calcFeatures import buildFeatMatrix

__all__ = ['Rhapsody', 'seqScanning', 'printSAVlist', 'mapSAVs2PDB',
           'calcPredictions']


class Rhapsody:
    """A class implementing the Rhapsody algorithm for pathogenicity
    prediction of human missense variants and that can also be used to
    compare results from other prediction tools, namely PolyPhen-2 and
    EVmutation.
    """

    def __init__(self, query=None, query_type='SAVs', queryPolyPhen2=True):

        assert query_type in ('SAVs', 'PolyPhen2')
        assert isinstance(queryPolyPhen2, bool)

        if query is None:
            # a SAV list can be uploaded later with setSAVs()
            # (useful when PolyPhen-2 features are not needed)
            self.query = None
            self.saturation_mutagenesis = None
        elif query_type == 'PolyPhen2':
            # 'query' must be a filename containing PolyPhen-2's output
            self.importPolyPhen2output(query)
        elif queryPolyPhen2:
            # 'query' can be a filename, list, tuple or string
            # containing SAV coordinates, or just a single string with
            # the Uniprot accession number of a sequence (with or without
            # a specified position) for which a complete scanning of all
            # mutations will be computed
            self.queryPolyPhen2(query)
        else:
            # as above, but without querying PolyPhen-2
            self.setSAVs(query)

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
            ('aux. score', 'f4'),
            ('aux. path. prob.', 'f4'),
            ('aux. path. class', 'U12'),
            # string indicating the best prediction set
            # ('main' or 'aux') to use for a given SAV
            ('best classifier', 'U4'),
            # predictions from PolyPhen-2 and EVmutation
            ('PolyPhen-2 score', 'f4'),
            ('PolyPhen-2 path. class', 'U12'),
            ('EVmutation score', 'f4'),
            ('EVmutation path. class', 'U12')
        ])
        # number of SAVs
        self.numSAVs = None
        # structured array containing parsed PolyPhen-2 output
        self.PolyPhen2output = None
        # custom PDB structure used for PDB features calculation
        self.customPDB = None
        # NumPy array (num_SAVs)x(num_features)
        self.featMatrix = None
        # classifiers and main feature set
        self.classifier = None
        self.aux_classifier = None
        self.featSet = None

    def _isColSet(self, column):
        assert self.data is not None, 'Data array not initialized.'
        return self.data[column].count() != 0

    def _isSaturationMutagenesis(self):
        assert self._isColSet('SAV coords'), 'SAV list not set.'
        if self.saturation_mutagenesis is None:
            self.saturation_mutagenesis = False
            try:
                SAVs = self.getUniqueSAVcoords()
                SAV_list = list(SAVs['SAV coords'])
                acc = SAVs[0]['Uniprot ID']
                pos = list(set(SAVs['position']))
                if len(pos) == 1:
                    query = f'{acc} {pos[0]}'
                else:
                    query = acc
                generated_SAV_list = seqScanning(query)
                if SAV_list == generated_SAV_list:
                    self.saturation_mutagenesis = True
            except Exception as e:
                LOGGER.warn(f'Not a saturation mutagenesis list: {e}')
        return self.saturation_mutagenesis

    def setSAVs(self, query):
        # 'query' can be a filename, list, tuple or string
        # containing SAV coordinates, or just a single string with
        # the Uniprot accession number of a sequence (with or without
        # a specified position) for which a complete scanning of all
        # mutations will be computed
        assert self.data is None, 'SAV list already set.'
        SAV_dtype = [
            ('acc', 'U10'),
            ('pos', 'i'),
            ('wt_aa', 'U1'),
            ('mut_aa', 'U1')
        ]
        if isinstance(query, str):
            if isfile(query):
                # 'query' is a filename, with line format 'P17516 135 G E'
                SAVs = np.loadtxt(query, dtype=SAV_dtype)
                SAV_list = ['{} {} {} {}'.format(*s) for s in SAVs]
            elif len(query.split()) < 3:
                # single Uniprot acc (+ pos), e.g. 'P17516' or 'P17516 135'
                SAV_list = seqScanning(query)
                self.saturation_mutagenesis = True
            else:
                # single SAV
                SAV = np.array(query.split(), dtype=SAV_dtype)
                SAV_list = ['{} {} {} {}'.format(*SAV)]
        else:
            # 'query' is a list or tuple of SAV coordinates
            SAVs = np.array([tuple(s.split()) for s in query], dtype=SAV_dtype)
            SAV_list = ['{} {} {} {}'.format(*s) for s in SAVs]
        # store SAV coordinates
        numSAVs = len(SAV_list)
        data = np.ma.masked_all(numSAVs, dtype=self.data_dtype)
        data['SAV coords'] = SAV_list
        self.data = data
        self.numSAVs = len(SAV_list)

    def queryPolyPhen2(self, query, filename='rhapsody-SAVs.txt'):
        assert self.data is None, 'SAV list already set.'
        assert self.PolyPhen2output is None, "PolyPhen-2 output " \
                                             "already imported."
        if isinstance(query, str) and isfile(query):
            # 'query' is a filename
            SAV_file = query
        elif isinstance(query, str) and len(query.split()) < 3:
            # single Uniprot acc (+ pos), e.g. 'P17516' or 'P17516 135'
            SAV_list = seqScanning(query)
            SAV_file = printSAVlist(SAV_list, filename)
        else:
            # 'query' is a list, tuple or single string of SAV coordinates
            SAV_file = printSAVlist(query, filename)
        # submit query to PolyPhen-2
        try:
            PolyPhen2_output = queryPolyPhen2(SAV_file)
        except Exception as e:
            err = (f'Unable to get a response from PolyPhen-2: {e} \n'
                   'Please click "Check Status" on the server homepage \n'
                   '(http://genetics.bwh.harvard.edu/pph2) \n'
                   'and try again when "Load" is "Low" and "Health" is 100%')
            raise RuntimeError(err)
        # import PolyPhen-2 output
        self.importPolyPhen2output(PolyPhen2_output)
        return self.PolyPhen2output

    def importPolyPhen2output(self, filename):
        assert self.data is None, 'SAV list already set.'
        assert self.PolyPhen2output is None, ("PolyPhen-2 output "
                                              "already imported.")
        self.PolyPhen2output = parsePolyPhen2output(filename)
        # store SAV coords
        self.setSAVs(getSAVcoords(self.PolyPhen2output)['text'])
        return self.PolyPhen2output

    def getSAVcoords(self):
        # they could also *not* be in Uniprot format, e.g.
        # 'rs397518423' or 'chr5:80390175 G/A'
        return np.array(self.data['SAV coords'])

    def setFeatSet(self, featset):
        assert self.featSet is None, 'Feature set already set.'
        if isinstance(featset, str):
            assert featset in ['all', 'full', 'reduced', 'EVmut']
            if featset == 'all':
                featset = sorted(list(RHAPSODY_FEATS['all']))
            else:
                featset == DEFAULT_FEATSETS[featset]
        if any([f not in RHAPSODY_FEATS['all'] for f in featset]):
            raise RuntimeError('Invalid list of features.')
        if len(set(featset)) != len(featset):
            raise RuntimeError('Duplicate features in feature set.')
        self.featSet = tuple(featset)

    def setCustomPDB(self, custom_PDB):
        if self.featSet is not None:
            if not RHAPSODY_FEATS['PDB'].intersection(self.featSet):
                LOGGER.warn('The given feature set does not require '
                            'a PDB structure.')
                return
        assert self.customPDB is None, 'Custom PDB structure already set.'
        assert isinstance(custom_PDB, (str, Atomic)), (
            'Please provide a PDBID, a filename or an Atomic instance.')
        self.customPDB = custom_PDB

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

    def getUniprot2PDBmap(self, filename='rhapsody-Uniprot2PDB.txt',
                          print_header=True, refresh=False):
        """Maps each SAV to the corresponding resid in a PDB chain.
        """
        assert self.data is not None, "SAVs not set."
        if not self._isColSet('PDB SAV coords'):
            # compute mapping
            m = mapSAVs2PDB(self.data['SAV coords'], custom_PDB=self.customPDB,
                            refresh=refresh)
            self.data['unique SAV coords'] = m['unique SAV coords']
            self.data['PDB SAV coords'] = m['PDB SAV coords']
            self.data['PDB size'] = m['PDB size']
        # print to file, if requested
        if filename is not None:
            with open(filename, 'w') as f:
                if print_header:
                    f.write('# SAV coords           '
                            'Uniprot coords         '
                            'PDB/ch/res/aa/size \n')
                for s in self.data:
                    orig_SAV = s['SAV coords'] + ','
                    U_coords = s['unique SAV coords'] + ','
                    if s['PDB size'] != 0:
                        PDB_coords = (s['PDB SAV coords'] + ' '
                                      + str(s['PDB size']))
                    else:
                        # print error message
                        PDB_coords = s['PDB SAV coords']
                    f.write(f'{orig_SAV:<22} {U_coords:<22} {PDB_coords:<}\n')
        return np.array(self.data[['SAV coords', 'unique SAV coords',
                                   'PDB SAV coords', 'PDB size']])

    def getPDBcoords(self):
        self.getUniprot2PDBmap(filename=None)
        dt = np.dtype([
            ('SAV coords', 'U50'),
            ('PDB SAV coords', 'U100'),
            ('PDBID', 'U12'),
            ('chain', 'U1'),
            ('resid', 'i4'),
            ('resname', 'U1'),
            ('PDB size', 'i4')
        ])
        PDBcoords = np.zeros(self.numSAVs, dtype=dt)
        PDBcoords['SAV coords'] = self.data['SAV coords']
        PDBcoords['PDB SAV coords'] = self.data['PDB SAV coords']
        fields = [
            row['PDB SAV coords'].split() if row['PDB size'] > 0
            else ['?', '?', -999, '?'] for row in self.data
        ]
        PDBcoords['PDBID'] = [r[0] for r in fields]
        PDBcoords['chain'] = [r[1] for r in fields]
        PDBcoords['resid'] = [r[2] for r in fields]
        PDBcoords['resname'] = [r[3] for r in fields]
        PDBcoords['PDB size'] = self.data['PDB size']
        return PDBcoords

    def getUniqueSAVcoords(self):
        self.getUniprot2PDBmap(filename=None)
        dt = np.dtype([
            ('SAV coords', 'U50'),
            ('unique SAV coords', 'U50'),
            ('Uniprot ID', 'U10'),
            ('position', 'i4'),
            ('wt. aa', 'U1'),
            ('mut. aa', 'U1')
        ])
        uSAVcoords = np.zeros(self.numSAVs, dtype=dt)
        for i, SAV in enumerate(self.data):
            try:
                uSAVcoords[i] = tuple(
                    [SAV['SAV coords'], SAV['unique SAV coords']] +
                    SAV['unique SAV coords'].split()
                )
            except Exception as e:
                LOGGER.warn(f'Invalid Uniprot coordinates at line {i}: {e}')
                uSAVcoords[i] = tuple(['?', -999, '?', '?'])
        return uSAVcoords

    def calcFeatures(self, filename='rhapsody-features.txt', refresh=False):
        if self.featMatrix is None:
            self.featMatrix = self._calcFeatMatrix(refresh=refresh)
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

    def _calcFeatMatrix(self, refresh=False):
        assert self.data is not None, 'SAVs not set.'
        assert self.featSet is not None, 'Feature set not set.'
        # list of structured arrays that will contain all computed features
        all_feats = []
        if RHAPSODY_FEATS['PolyPhen2'].intersection(self.featSet):
            # retrieve sequence-conservation features from PolyPhen-2's output
            assert self.PolyPhen2output is not None, \
                   "Please import PolyPhen-2's output first."
            f = calcPolyPhen2features(self.PolyPhen2output)
            all_feats.append(f)
        sel_PDBfeats = RHAPSODY_FEATS['PDB'].intersection(self.featSet)
        if sel_PDBfeats:
            # map SAVs to PDB structures
            Uniprot2PDBmap = self.getUniprot2PDBmap(refresh=refresh)
            # compute structural and dynamical features from a PDB structure
            f = calcPDBfeatures(Uniprot2PDBmap, sel_feats=sel_PDBfeats,
                                custom_PDB=self.customPDB, refresh=refresh)
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

    def exportTrainingData(self, refresh=False):
        assert self.data is not None, 'SAVs not set.'
        assert self._isColSet('true labels'), 'True labels not set.'
        if self.featMatrix is None:
            self.featMatrix = self._calcFeatMatrix(refresh=refresh)
        dt = np.dtype([('SAV_coords', '<U50'), ('Uniprot2PDB', '<U100'),
                       ('PDB_length', '<i2'), ('true_label', '<i2')] +
                      [(f, '<f4') for f in self.featSet])
        num_SAVs = len(self.data)
        trainData = np.empty(num_SAVs, dtype=dt)
        trainData['SAV_coords'] = self.data['SAV coords']
        if self._isColSet('PDB SAV coords'):
            trainData['Uniprot2PDB'] = self.data['PDB SAV coords']
            trainData['PDB_length'] = self.data['PDB size']
        trainData['true_label'] = self.data['true labels']
        for i, f in enumerate(self.featSet):
            trainData[f] = self.featMatrix[:, i]
        return trainData

    def importClassifiers(self, classifier, aux_classifier=None,
                          force_env=None):
        assert self.classifier is None, 'Classifiers already set.'
        assert force_env in [None, 'chain', 'reduced', 'sliced'], \
               "Invalid 'force_env' value"
        # import main classifier
        p = pickle.load(open(classifier, 'rb'))
        featset = p['features']
        main_clsf = {
            'path': classifier,
            'CV summary': p['CV summary'],
            'featset': p['features']
        }
        if force_env:
            # force a given ENM environment model
            featset = self._replaceEnvModel(featset, force_env)
            main_clsf['mod. featset'] = featset
        # import auxiliary classifier
        if aux_classifier is None:
            aux_clsf = None
            aux_featset = []
        else:
            p = pickle.load(open(aux_classifier, 'rb'))
            aux_featset = p['features']
            aux_clsf = {
                'path': aux_classifier,
                'CV summary': p['CV summary'],
                'featset': p['features']
            }
            if any(f not in main_clsf['featset'] for f in aux_clsf['featset']):
                raise ValueError('The auxiliary feature set must be a '
                                 'subset of the main one.')
            if force_env:
                # force a given ENM environment model
                aux_featset = self._replaceEnvModel(aux_featset, force_env)
                aux_clsf['mod. featset'] = aux_featset
        # print featset
        LOGGER.info('Imported feature set:')
        for i, f in enumerate(featset):
            note1 = '*' if f in aux_featset else ''
            if f != main_clsf['featset'][i]:
                original_env = main_clsf['featset'][i].split('-')[-1]
                note2 = f"(originally '-{original_env}')"
            else:
                note2 = ''
            LOGGER.info(f"   '{f}'{note1} {note2}")
        if aux_clsf:
            LOGGER.info("   (* auxiliary feature set)")
        # store classifiers and main feature set
        self.classifier = main_clsf
        self.aux_classifier = aux_clsf
        self.setFeatSet(featset)

    def _replaceEnvModel(self, featset, new_env):
        new_env = '-' + new_env
        new_featset = []
        for i, f in enumerate(featset):
            if any(f.endswith(e) for e in ['-chain', '-reduced', '-sliced']):
                old_env = '-' + f.split('-')[-1]
                new_featset.append(f.replace(old_env, new_env))
            else:
                new_featset.append(f)
        return new_featset

    def _calcPredictions(self, refresh=False):
        assert self.classifier is not None, 'Classifier not set.'
        if self._isColSet('main score'):
            return
        # compute features
        self.calcFeatures(refresh=refresh)
        # compute main predictions
        preds = calcPredictions(self.featMatrix, self.classifier['path'],
                                SAV_coords=self.data['SAV coords'])
        self.data['training info'] = preds['training info']
        self.data['main score'] = preds['score']
        self.data['main path. prob.'] = preds['path. probability']
        self.data['main path. class'] = preds['path. class']
        self.data['best classifier'] = 'main'
        if self.aux_classifier:
            # reduce original feature matrix
            aux_fs = self.aux_classifier.get('mod. featset',
                                             self.aux_classifier['featset'])
            sel = [i for i, f in enumerate(self.featSet) if f in aux_fs]
            fm = self.featMatrix[:, sel]
            # compute auxiliary predictions
            aux_preds = calcPredictions(fm, self.aux_classifier['path'],
                                        SAV_coords=self.data['SAV coords'])
            self.data['aux. score'] = aux_preds['score']
            self.data['aux. path. prob.'] = aux_preds['path. probability']
            self.data['aux. path. class'] = aux_preds['path. class']
            # select best classifier for each SAV
            main_score = self.data['main score']
            self.data['best classifier'] = np.where(np.isnan(main_score),
                                                    'aux.', 'main')

    def _calcPolyPhen2Predictions(self):
        assert self.PolyPhen2output is not None, 'PolyPhen-2 output not found.'
        if self._isColSet('PolyPhen-2 score'):
            return
        PP2_score = [x if x != '?' else 'nan' for x in
                     self.PolyPhen2output['pph2_prob']]
        PP2_class = self.PolyPhen2output['pph2_class']
        self.data['PolyPhen-2 score'] = PP2_score
        self.data['PolyPhen-2 path. class'] = PP2_class

    def _calcEVmutationPredictions(self):
        if self._isColSet('EVmutation score'):
            return
        EVmut_feats = recoverEVmutFeatures(self.data['SAV coords'])
        EVmut_score = EVmut_feats['EVmut-DeltaE_epist']
        c = -SETTINGS.get('EVmutation_metrics')['optimal cutoff']
        EVmut_class = np.where(EVmut_score < c, 'deleterious', 'neutral')
        EVmut_class[np.isnan(EVmut_score)] = '?'
        self.data['EVmutation score'] = EVmut_score
        self.data['EVmutation path. class'] = EVmut_class

    def getPredictions(self, SAV='all', classifier='best',
                       PolyPhen2=True, EVmutation=True,
                       PDBcoords=False, refresh=False):
        assert classifier in ['best', 'main', 'aux'], "Invalid 'classifier'."
        if classifier == 'aux' and self.aux_classifier is None:
            raise ValueError('Auxiliary classifier not found.')
        # initialize output array
        cols = [
            ('SAV coords', 'U50'),
            ('training info', 'U12'),
            ('score', 'f4'),
            ('path. prob.', 'f4'),
            ('path. class', 'U12')
        ]
        if PolyPhen2:
            cols.extend([
                ('PolyPhen-2 score', 'f4'),
                ('PolyPhen-2 path. class', 'U12')
            ])
        if EVmutation:
            cols.extend([
                ('EVmutation score', 'f4'),
                ('EVmutation path. class', 'U12')
            ])
        if PDBcoords:
            cols.append(
                ('PDB SAV coords', 'U100')
            )
        # get Rhapsody predictions
        self._calcPredictions(refresh=refresh)
        output = np.empty(self.numSAVs, dtype=np.dtype(cols))
        output['SAV coords'] = self.data['SAV coords']
        output['training info'] = self.data['training info']
        for s in ['score', 'path. prob.', 'path. class']:
            if classifier == 'best':
                output[s] = np.where(self.data['best classifier'] == 'main',
                                     self.data[f'main {s}'],
                                     self.data[f'aux. {s}'])
            elif classifier == 'main':
                output[s] = self.data[f'main {s}']
            else:
                output[s] = self.data[f'aux. {s}']
        # get PolyPhen-2 predictions
        if PolyPhen2:
            self._calcPolyPhen2Predictions()
            for s in ['PolyPhen-2 score', 'PolyPhen-2 path. class']:
                output[s] = self.data[s]
        # get EVmutation predictions
        if EVmutation:
            self._calcEVmutationPredictions()
            for s in ['EVmutation score', 'EVmutation path. class']:
                output[s] = self.data[s]
        # get PDB coordinates
        if PDBcoords:
            self.getUniprot2PDBmap(filename=None)
            output['PDB SAV coords'] = self.data['PDB SAV coords']
        # return output
        if SAV == 'all':
            return output
        elif isinstance(SAV, int):
            return output[SAV]
        elif SAV in output['SAV coords']:
            return output[output['SAV coords'] == SAV][0]
        else:
            raise ValueError('Invalid SAV.')

    def _calcResAvg(self, array, dtype='float'):
        assert dtype in ['float', 'int', 'str']
        array = array.copy()
        m = array.reshape((-1, 19)).T
        if dtype == 'float':
            return np.nanmean(m, axis=0)
        else:
            uniq_rows = np.unique(m, axis=0)
            if len(uniq_rows) != 1:
                raise RuntimeError('Invalid saturation mutagenesis list')
            return uniq_rows[0]

    def getResAvgPredictions(self, resid=None, classifier='best',
                             PolyPhen2=True, EVmutation=True,
                             refresh=False):
        if not self._isSaturationMutagenesis():
            return None
        # initialize output array
        cols = [
            ('sequence index', 'i4'),
            ('PDB resid', 'i4'),
            ('wt. aa', 'U1'),
            ('score', 'f4'),
            ('path. prob.', 'f4'),
            ('path. class', 'U12')
        ]
        if PolyPhen2:
            cols.extend([
                ('PolyPhen-2 score', 'f4'),
                ('PolyPhen-2 path. class', 'U12')
            ])
        if EVmutation:
            cols.extend([
                ('EVmutation score', 'f4'),
                ('EVmutation path. class', 'U12')
            ])
        output = np.empty(int(self.numSAVs/19), dtype=np.dtype(cols))
        # fetch unique SAV coords, PDB coords and predictions
        uSAVc = self.getUniqueSAVcoords()
        PDBc = self.getPDBcoords()
        preds = self.getPredictions(classifier=classifier, PolyPhen2=PolyPhen2,
                                    EVmutation=EVmutation, refresh=refresh)
        # compute residue-averaged quantities
        output['sequence index'] = self._calcResAvg(uSAVc['position'], 'int')
        output['PDB resid'] = self._calcResAvg(PDBc['resid'], 'int')
        output['wt. aa'] = self._calcResAvg(uSAVc['wt. aa'], 'str')
        # NB: I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            output['score'] = self._calcResAvg(preds['score'])
            pp = self._calcResAvg(preds['path. prob.'])
            pc = np.where(pp > 0.5, 'deleterious', 'neutral')
            pc = np.where(np.isnan(pp), '?', pc)
            output['path. prob.'] = pp
            output['path. class'] = pc
            if PolyPhen2:
                ps = self._calcResAvg(preds['PolyPhen-2 score'])
                pc = np.where(ps > 0.5, 'deleterious', 'neutral')
                pc = np.where(np.isnan(ps), '?', pc)
                output['PolyPhen-2 score'] = ps
                output['PolyPhen-2 path. class'] = pc
            if EVmutation:
                ps = self._calcResAvg(preds['EVmutation score'])
                cutoff = -SETTINGS.get('EVmutation_metrics')['optimal cutoff']
                pc = np.where(ps < cutoff, 'deleterious', 'neutral')
                pc = np.where(np.isnan(ps), '?', pc)
                output['EVmutation score'] = ps
                output['EVmutation path. class'] = pc
        if resid is None:
            return output
        elif isinstance(resid, int):
            return output[output['PDB resid'] == resid][0]
        else:
            raise ValueError('Invalid resid.')

    def printPredictions(self, classifier='best',
                         PolyPhen2=True, EVmutation=True,
                         filename='rhapsody-predictions.txt',
                         print_header=True):
        assert classifier in ['best', 'main', 'aux', 'both']
        if classifier != 'both':
            preds = self.getPredictions(classifier=classifier,
                                        PolyPhen2=PolyPhen2,
                                        EVmutation=EVmutation)
            with open(filename, 'w') as f:
                if print_header:
                    header = '{:25} {:15} {:6} {:6} {:14}'.format(
                        '# SAV coords',
                        'training info',
                        'score',
                        'prob.',
                        'class'
                    )
                    if PolyPhen2:
                        header += 'PolyPhen-2 score/class   '
                    if EVmutation:
                        header += 'EVmutation score/class'
                    f.write(header + '\n')
                for SAV in preds:
                    row = '{:25} {:15} {:<5.3f}  {:<5.3f}  {:14}'.format(
                        SAV['SAV coords'],
                        SAV['training info'],
                        SAV['score'],
                        SAV['path. prob.'],
                        SAV['path. class']
                    )
                    if PolyPhen2:
                        row += '{:<5.3f}    {:16}'.format(
                            SAV['PolyPhen-2 score'],
                            SAV['PolyPhen-2 path. class']
                        )
                    if EVmutation:
                        row += '{:<7.3f}    {:12}'.format(
                            SAV['EVmutation score'],
                            SAV['EVmutation path. class']
                        )
                    f.write(row + '\n')
        else:
            # print both main and aux predictions in a more detailed format
            self.getPredictions(classifier='aux', PolyPhen2=PolyPhen2,
                                EVmutation=EVmutation)
            with open(filename, 'w') as f:
                if print_header:
                    header = '{:25} {:15} {:33} {:30}'.format(
                        '# SAV coords',
                        'training info',
                        'main classifier predictions',
                        'aux. classifier predictions')
                    if PolyPhen2:
                        header += 'PolyPhen-2 score/class   '
                    if EVmutation:
                        header += 'EVmutation score/class'
                    f.write(header + '\n')
                for SAV in self.data:
                    row = '{:25} {:15} {:<5.3f}  {:<5.3f}  {:15}'.format(
                        SAV['SAV coords'],
                        SAV['training info'],
                        SAV['main score'],
                        SAV['main path. prob.'],
                        SAV['main path. class']
                    )
                    if np.isnan(SAV['main score']) and \
                       not np.isnan(SAV['aux. score']):
                        row += '<--'
                    else:
                        row += 'x--'
                    row += '  {:<5.3f}  {:<5.3f}  {:16}'.format(
                        SAV['aux. score'],
                        SAV['aux. path. prob.'],
                        SAV['aux. path. class']
                    )
                    if PolyPhen2:
                        row += '{:<5.3f}    {:16}'.format(
                            SAV['PolyPhen-2 score'],
                            SAV['PolyPhen-2 path. class']
                        )
                    if EVmutation:
                        row += '{:<7.3f}    {:12}'.format(
                            SAV['EVmutation score'],
                            SAV['EVmutation path. class']
                        )
                    f.write(row + '\n')

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
    assert len(coord) < 3, "Invalid format. Examples: 'Q9BW27' or 'Q9BW27 10'."
    Uniprot_record = queryUniprot(coord[0])
    sequence = Uniprot_record['sequence   0'].replace("\n", "")
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
        input_SAVs = [input_SAVs]
    with open(filename, 'w', 1) as f:
        for i, line in enumerate(input_SAVs):
            m = f'error in SAV {i}: '
            assert isinstance(line, str), f'{m} not a string'
            assert len(line) < 25, f'{m} too many characters'
            print(line, file=f)
    return filename


def mapSAVs2PDB(SAV_coords, custom_PDB=None, refresh=False):
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
    nSAVs = len(SAV_coords)
    mapped_SAVs = np.zeros(nSAVs, dtype=PDBmap_dtype)
    # map to PDB using Uniprot class
    cache = {'acc': None, 'obj': None}
    count = 0
    for indx, SAV in [(i, SAV_coords[i]) for i in sorting_map]:
        count += 1
        acc, pos, aa1, aa2 = SAV.split()
        pos = int(pos)
        LOGGER.info(f"[{count}/{nSAVs}] Mapping SAV '{SAV}' to PDB...")
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
                U2P_map = UniprotMapping(acc, recover_pickle=not(refresh))
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
    n = sum(mapped_SAVs['PDB size'] != 0)
    LOGGER.report(f'{n} out of {nSAVs} SAVs have been mapped to PDB in %.1fs.',
                  '_map2PDB')
    return mapped_SAVs


def calcPredictions(feat_matrix, clsf, SAV_coords=None):
    assert SAV_coords is None or len(SAV_coords) == len(feat_matrix)

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
    sel_rows = [i for i, r in enumerate(feat_matrix) if all(~np.isnan(r))]
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
    Jplus = J + err_bar
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
