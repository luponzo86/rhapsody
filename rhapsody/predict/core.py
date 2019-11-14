# -*- coding: utf-8 -*-
"""This module defines the main class used for running the pre-trained
classifiers and organizing its predictions."""

import numpy as np
import warnings
import pickle
import prody as pd
from os.path import isfile
from prody import LOGGER, SETTINGS
from ..features import Uniprot, PDB, PolyPhen2, EVmutation, Pfam, BLOSUM
from ..features import RHAPSODY_FEATS
from ..utils.settings import DEFAULT_FEATSETS

__all__ = ['Rhapsody', 'calcPredictions']


class Rhapsody:
    """A class implementing the Rhapsody algorithm for pathogenicity
    prediction of human missense variants and that can also be used to
    compare results from other prediction tools, namely PolyPhen-2 and
    EVmutation.
    """

    def __init__(self, query=None, query_type='SAVs', queryPolyPhen2=True,
                 **kwargs):
        """ Initialize a Rhapsody object with a list of SAVs (optional).

        :arg query: Single Amino Acid Variants (SAVs) in Uniprot coordinates.

          - If **None**, the SAV list can be imported at a later moment, by
            using ``.importPolyPhen2output()``, ``.queryPolyPhen2()`` or
            ``.setSAVs()``

          - if *query_type* = ``'SAVs'`` (default), *query* should be a
            filename, a string or a list/tuple of strings, containing Uniprot
            SAV coordinates, with the format ``'P17516 135 G E'``. The string
            could also be just a single Uniprot sequence identifier (e.g.
            ``'P17516'``), or the coordinate of a specific site in a sequence
            (e.g. ``'P17516 135'``), in which case all possible 19 amino acid
            substitutions at the specified positions will be analyzed.
          - if *query_type* = ``'PolyPhen2'``, *query* should be a filename
            containing the output from PolyPhen-2, usually named
            :file:`pph2-full.txt`
        :type query: str, list

        :arg query_type: ``'SAVs'`` or ``'PolyPhen2'``
        :type query_type: str

        :arg queryPolyPhen2: if ``True``, the PolyPhen-2 online tool will be
            queryied with the list of SAVs
        :type queryPolyPhen2: bool
        """

        assert query_type in ('SAVs', 'PolyPhen2')
        assert isinstance(queryPolyPhen2, bool)
        valid_kwargs = [
            'status_file_Uniprot',
            'status_file_PDB',
            'status_file_Pfam',
            'status_prefix_Uniprot',
            'status_prefix_PDB',
            'status_prefix_Pfam']
        assert all([k in valid_kwargs for k in kwargs])

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
        # structured array containing additional precomputed features
        self.extra_features = None
        # NumPy array (num_SAVs)x(num_features)
        self.featMatrix = None
        # classifiers and main feature set
        self.classifier = None
        self.aux_classifier = None
        self.featSet = None
        # options
        self.options = kwargs

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
                generated_SAV_list = Uniprot.seqScanning(query)
                if SAV_list == generated_SAV_list:
                    self.saturation_mutagenesis = True
                else:
                    raise RuntimeError('Missing SAVs detected.')
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
                SAV_list = Uniprot.seqScanning(query)
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
        fix_isoforms = False
        if isinstance(query, str) and isfile(query):
            # 'query' is a filename
            SAV_file = query
        elif isinstance(query, str) and len(query.split()) < 3:
            # single Uniprot acc (+ pos), e.g. 'P17516' or 'P17516 135'
            SAV_list = Uniprot.seqScanning(query)
            SAV_file = Uniprot.printSAVlist(SAV_list, filename)
            # only when submitting a saturation mutagenesis list, try and
            # fix possible wrong isoforms used by PolyPhen-2
            fix_isoforms = True
        else:
            # 'query' is a list, tuple or single string of SAV coordinates
            SAV_file = Uniprot.printSAVlist(query, filename)
        # submit query to PolyPhen-2
        try:
            PolyPhen2_output = PolyPhen2.queryPolyPhen2(
                SAV_file, fix_isoforms=fix_isoforms)
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
        pp2_output = PolyPhen2.parsePolyPhen2output(filename)
        # store SAV coords
        self.setSAVs(PolyPhen2.getSAVcoords(pp2_output)['text'])
        self.PolyPhen2output = pp2_output
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
        # check for unrecognized features
        known_feats = RHAPSODY_FEATS['all']
        if self.extra_features is not None:
            known_feats = known_feats.union(self.extra_features.dtype.names)
        for f in featset:
            if f not in known_feats:
                raise RuntimeError(f"Unknown feature: '{f}'")
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
        assert isinstance(custom_PDB, (str, pd.Atomic)), (
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
            m = Uniprot.mapSAVs2PDB(
                self.data['SAV coords'], custom_PDB=self.customPDB,
                status_file=self.options.get('status_file_Uniprot'),
                status_prefix=self.options.get('status_prefix_Uniprot'),
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
            ('PDBID', 'U100'),
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

    def _buildFeatMatrix(self, featset, all_features):
        n_rows = len(all_features[0])
        n_cols = len(featset)
        feat_matrix = np.zeros((n_rows, n_cols))
        for j, featname in enumerate(featset):
            # find structured array containing a specific feature
            arrays = [a for a in all_features if featname in a.dtype.names]
            if len(arrays) == 0:
                raise RuntimeError(f'Invalid feature name: {featname}')
            if len(arrays) > 1:
                LOGGER.warn(f'Multiple values for feature {featname}')
            array = arrays[0]
            feat_matrix[:, j] = array[featname]
        return feat_matrix

    def _calcFeatMatrix(self, refresh=False):
        assert self.data is not None, 'SAVs not set.'
        assert self.featSet is not None, 'Feature set not set.'
        # list of structured arrays that will contain all computed features
        all_feats = []
        if RHAPSODY_FEATS['PolyPhen2'].intersection(self.featSet):
            # retrieve sequence-conservation features from PolyPhen-2's output
            assert self.PolyPhen2output is not None, \
                   "Please import PolyPhen-2's output first."
            f = PolyPhen2.calcPolyPhen2features(self.PolyPhen2output)
            all_feats.append(f)
        sel_PDBfeats = RHAPSODY_FEATS['PDB'].intersection(self.featSet)
        if sel_PDBfeats:
            # map SAVs to PDB structures
            Uniprot2PDBmap = self.getUniprot2PDBmap(refresh=refresh)
            # compute structural and dynamical features from a PDB structure
            f = PDB.calcPDBfeatures(
                Uniprot2PDBmap, sel_feats=sel_PDBfeats,
                custom_PDB=self.customPDB, refresh=refresh,
                status_file=self.options.get('status_file_PDB'),
                status_prefix=self.options.get('status_prefix_PDB'))
            all_feats.append(f)
        if RHAPSODY_FEATS['BLOSUM'].intersection(self.featSet):
            # retrieve BLOSUM values
            f = BLOSUM.calcBLOSUMfeatures(self.data['SAV coords'])
            all_feats.append(f)
        if RHAPSODY_FEATS['Pfam'].intersection(self.featSet):
            # compute sequence properties from Pfam domains
            f = Pfam.calcPfamFeatures(
                self.data['SAV coords'],
                status_file=self.options.get('status_file_Pfam'),
                status_prefix=self.options.get('status_prefix_Pfam'))
            all_feats.append(f)
        if RHAPSODY_FEATS['EVmut'].intersection(self.featSet):
            # recover EVmutation data
            f = EVmutation.recoverEVmutFeatures(self.data['SAV coords'])
            all_feats.append(f)
        if self.extra_features is not None:
            all_feats.append(self.extra_features)
        # build matrix of selected features
        featm = self._buildFeatMatrix(self.featSet, all_feats)
        return featm

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

    def importPrecomputedFeatures(self, features_dict):
        assert isinstance(features_dict, dict)
        # import additional precomputed features
        default_feats = RHAPSODY_FEATS['all']
        if any([f in default_feats for f in features_dict]):
            ff = default_feats.intersection(set(features_dict))
            raise ValueError('Cannot import precomputed features already '
                             f"in Rhapsody's default list of features: {ff}")
        # store precomputed features in a structured array
        if self.numSAVs is None:
            raise RuntimeError('SAVs need to be imported first')
        dt = [(f, 'f4') for f in features_dict]
        extra_feats = np.empty(self.numSAVs, dtype=np.dtype(dt))
        for feat, array in features_dict.items():
            extra_feats[feat] = array
        self.extra_features = extra_feats

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
        EVmut_feats = EVmutation.recoverEVmutFeatures(self.data['SAV coords'])
        EVmut_score = EVmut_feats['EVmut-DeltaE_epist']
        EVmut_class = EVmutation.calcEVmutPathClasses(EVmut_score)
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

    def _calcResAvg(self, array):
        array = array.copy()
        m = array.reshape((-1, 19)).T
        if array.dtype.name.startswith('float'):
            return np.nanmean(m, axis=0)
        else:
            uniq_rows = np.unique(m, axis=0)
            if len(uniq_rows) != 1:
                raise RuntimeError('Invalid saturation mutagenesis list')
            return uniq_rows[0]

    def getResAvgPredictions(self, resid=None, classifier='best',
                             PolyPhen2=True, EVmutation=True, refresh=False):
        if not self._isSaturationMutagenesis():
            return None
        # initialize output array
        cols = [
            ('sequence index', 'i4'),
            ('PDB SAV coords', 'U100'),
            ('PDBID', 'U100'),
            ('chain', 'U1'),
            ('resid', 'i4'),
            ('resname', 'U1'),
            ('PDB size', 'i4'),
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
        output['sequence index'] = self._calcResAvg(uSAVc['position'])
        for field in ['PDB SAV coords', 'PDBID', 'chain',
                      'resid', 'resname', 'PDB size']:
            output[field] = self._calcResAvg(PDBc[field])
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
            return output[output['resid'] == resid][0]
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

    def writePDBs(self, PDBID=None, predictions='best', path_prob=True,
                  filename_prefix='rhapsody-PDB', refresh=False):
        assert predictions in ['best', 'main', 'aux',
                               'PolyPhen-2', 'EVmutation']
        if not self._isSaturationMutagenesis():
            LOGGER.warn('This function is available only when performing '
                        'saturation mutagenesis analysis')
            return None
        # select prediction set to be printed on PDB file
        kwargs = {'classifier': 'main', 'PolyPhen2': False,
                  'EVmutation': False, 'refresh': refresh}
        if predictions in ['best', 'main', 'aux']:
            kwargs['classifier'] = predictions
            array = self.getResAvgPredictions(**kwargs)
            if path_prob:
                sel_preds = 'path. prob.'
            else:
                sel_preds = 'score'
        elif predictions == 'PolyPhen-2':
            kwargs['PolyPhen2'] = True
            array = self.getResAvgPredictions(**kwargs)
            sel_preds = 'PolyPhen-2 score'
        else:
            kwargs['EVmutation'] = True
            array = self.getResAvgPredictions(**kwargs)
            sel_preds = 'EVmutation score'
        # select PDB structures to be printed
        PDBIDs = set(array[array['PDB size'] > 0]['PDBID'])
        if PDBID is None:
            PDBIDs = list(PDBIDs)
        elif PDBID in PDBIDs:
            PDBIDs = [PDBID, ]
        else:
            raise ValueError('Invalid PDBID')
        # write residue-averaged predictions on B-factor column of PDB file
        output_dict = {}
        for id in PDBIDs:
            # import PDB structure
            if self.customPDB is not None:
                if isinstance(self.customPDB, pd.Atomic):
                    pdb = self.customPDB
                else:
                    pdb = pd.parsePDB(self.customPDB, model=1)
                fname = f'{filename_prefix}_custom.pdb'
            else:
                pdb = pd.parsePDB(id, model=1)
                fname = f'{filename_prefix}_{id}.pdb'
            # find chains in PDB
            PDBchids = set(pdb.getChids())
            # find chains used for predictions
            array_id = array[array['PDBID'] == id]
            chids = set(array_id['chain'])
            # replace B-factor column in each chain
            for chid in PDBchids:
                PDBresids = pdb[chid].getResnums()
                new_betas = np.array([np.nan]*len(PDBresids))
                if chid in chids:
                    array_ch = array_id[array_id['chain'] == chid]
                    for l in array_ch:
                        new_betas[PDBresids == l['resid']] = l[sel_preds]
                pdb[chid].setBetas(new_betas)
            # write PDB to file
            f = pd.writePDB(fname, pdb)
            self.__replaceNanInPDBBetaColumn(fname)
            output_dict[f] = pdb
            LOGGER.info(f'Predictions written to PDB file {fname}')
        return output_dict

    def __replaceNanInPDBBetaColumn(self, filename):
        # In the current implementation of Prody, you cannot set an empty
        # string in the B-factor column...
        with open(filename, 'r') as file:
            filedata = file.readlines()
        with open(filename, 'w') as file:
            for line in filedata:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    line = line.replace(' nan', '    ')
                file.write(line)

    def savePickle(self, filename='rhapsody-pickle.pkl'):
        f = pickle.dump(self, open(filename, "wb"))
        return f


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
    train_data = clsf_dict['CV summary']['training dataset']

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
    delSAVs = train_data['SAV_coords'][train_data['true_label'] == 1]
    neuSAVs = train_data['SAV_coords'][train_data['true_label'] == 0]
    k = 0
    for i in range(len(feat_matrix)):
        # determine SAV status
        if SAV_coords is None:
            SAV_status = '?'
        elif SAV_coords[i] in delSAVs:
            SAV_status = 'known_del'
        elif SAV_coords[i] in neuSAVs:
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
