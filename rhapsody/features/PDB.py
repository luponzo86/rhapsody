# -*- coding: utf-8 -*-
"""This module defines a class that organizes the calculation of
PDB-based structural and dynamical features in a single place, and a
function for using the latter on a list of PDB SAV coordinates."""

import numpy as np
import pickle
import datetime
import os
from tqdm import tqdm
from prody import Atomic, parsePDB, writePDB, LOGGER, SETTINGS
from prody import GNM, ANM, calcSqFlucts
from prody import calcPerturbResponse, calcMechStiff
# from prody import calcMBS
from prody import reduceModel, sliceModel
from prody import execDSSP, parseDSSP

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

__all__ = ['STR_FEATS', 'DYN_FEATS', 'PDB_FEATS',
           'PDBfeatures', 'calcPDBfeatures']

MAX_NUM_RESIDUES = 17000
"""Hard-coded maximum size of PDB structures that can be handled by the
class :class:`PDBfeatures()`."""

STR_FEATS = ['SASA', 'SASA_in_complex', 'Delta_SASA']
"""List of available structural features."""

DYN_FEATS = ['GNM_MSF', 'ANM_MSF',
             'GNM_effectiveness', 'GNM_sensitivity',
             'ANM_effectiveness', 'ANM_sensitivity',
             'stiffness']  # , 'MBS']
"""List of available dynamical features."""

PDB_FEATS = STR_FEATS + [f + e for f in DYN_FEATS
                         for e in ['-chain', '-reduced', '-sliced']]
"""List of available PDB-based structural and dynamical features. The latter
can be computed by using three different environment models."""


class PDBfeatures:
    """A class for deriving structural and dynamical properties from a
    PDB structure.

    :arg PDB: an object or a PDB code identifying a PDB structure.
    :type PDB: :class:`Atomic`, str
    :arg n_modes: number of GNM/ANM modes to be computed.
    :type n_modes: int, str
    :arg recover_pickle: whether or not to recover precomputed pickle, if found
    :type recover_pickle: bool
    """

    def __init__(self, PDB, n_modes='all', recover_pickle=False, **kwargs):
        assert isinstance(PDB, (str, Atomic)), \
               'PDB must be either a PDBID or an Atomic instance.'
        assert type(recover_pickle) is bool
        # definition and initialization of variables
        if isinstance(PDB, str):
            self.PDBID = PDB
            self._pdb = None
        else:
            self.PDBID = None
            self._pdb = PDB.copy()
        self.n_modes = n_modes
        self.chids = None
        self.resids = None
        self.feats = None
        self._gnm = None
        self._anm = None
        self.timestamp = None
        if recover_pickle:
            try:
                self.recoverPickle(**kwargs)
            except Exception as e:
                LOGGER.warn(f'Unable to recover pickle: {e}')
                self.refresh()
        else:
            self.refresh()
        return

    def getPDB(self):
        """Returns the parsed PDB structure as an :class:`AtomGroup` object."""
        if self._pdb is None:
            self._pdb = parsePDB(self.PDBID, model=1)
        return self._pdb

    def refresh(self):
        """Deletes all precomputed ENM models and features, and resets
        time stamp."""
        pdb = self.getPDB()
        self.chids = set(pdb.ca.getChids())
        self.resids = {chID: pdb[chID].ca.getResnums()
                       for chID in self.chids}
        self._gnm = {}
        self._anm = {}
        for env in ['chain', 'reduced', 'sliced']:
            self._gnm[env] = {chID: None for chID in self.chids}
            self._anm[env] = {chID: None for chID in self.chids}
        self.feats = {chID: {} for chID in self.chids}
        self.timestamp = str(datetime.datetime.utcnow())
        return

    def recoverPickle(self, folder=None, filename=None, days=30, **kwargs):
        """Looks for precomputed pickle for the current PDB structure.

        :arg folder: path of folder where pickles are stored. If not specified,
            pickles will be searched for in the local Rhapsody installation
            folder.
        :type folder: str
        :arg filename: name of the pickle. If not specified, the default
            filename ``'PDBfeatures-[PDBID].pkl'`` will be used. If a PDBID is
            not found, user must specify a valid filename.
        :type filename: str
        :arg days: number of days after which a pickle will be considered too
            old and won't be recovered.
        :type days: int
        """
        if folder is None:
            # define folder where to look for pickles
            folder = SETTINGS.get('rhapsody_local_folder')
            if folder is None:
                folder = '.'
            else:
                folder = os.path.join(folder, 'pickles')
        if filename is None:
            # use the default filename, if possible
            if self.PDBID is not None:
                filename = 'PDBfeatures-' + self.PDBID + '.pkl'
            else:
                # when a custom structure is used, there is no
                # default filename: the user should provide it
                raise ValueError('Please provide a filename.')
        pickle_path = os.path.join(folder, filename)
        if not os.path.isfile(pickle_path):
            raise IOError("File '{}' not found".format(filename))
        recovered_self = pickle.load(open(pickle_path, "rb"))
        # check consistency of recovered data
        if self.PDBID is None:
            if self._pdb != recovered_self._pdb:
                raise ValueError('Incompatible PDB structure in recovered pickle.')
        elif self.PDBID != recovered_self.PDBID:
            raise ValueError('PDBID in recovered pickle ({}) does not match.'
                             .format(recovered_self.PDBID))
        if self.n_modes != recovered_self.n_modes:
            raise ValueError('Num. of modes in recovered pickle ({}) does not match.'
                             .format(recovered_self.n_modes))
        # check timestamp and ignore pickles that are too old
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        t_old = datetime.datetime.strptime(
            recovered_self.timestamp, date_format)
        t_now = datetime.datetime.utcnow()
        Delta_t = datetime.timedelta(days=days)
        if t_old + Delta_t < t_now:
            raise RuntimeError('Pickle was too old and was ignored.')
        # import recovered data
        self.chids = recovered_self.chids
        self.resids = recovered_self.resids
        self.feats = recovered_self.feats
        self._gnm = recovered_self._gnm
        self._anm = recovered_self._anm
        self.timestamp = recovered_self.timestamp
        LOGGER.info("Pickle '{}' recovered.".format(filename))
        return

    def savePickle(self, folder=None, filename=None):
        """Stores a pickle of the current class instance. The pickle will
        contain all information and precomputed features, but not GNM and ANM
        models. In case a PDBID is missing, the parsed PDB :class:`AtomGroup`
        is stored as well.

        :arg folder: path of the folder where the pickle will be saved. If not
            specified, the local Rhapsody installation folder will be used.
        :type folder: str
        :arg filename: name of the pickle. By default, the pickle will be
            saved as ``'PDBfeatures-[PDBID].pkl'``. If a PDBID is not defined,
            the user must provide a filename.
        :type filename: str
        :return: pickle path
        :rtype: str
        """
        if folder is None:
            # define folder where to look for pickles
            folder = SETTINGS.get('rhapsody_local_folder')
            if folder is None:
                folder = '.'
            else:
                folder = os.path.join(folder, 'pickles')
        if filename is None:
            # use the default filename, if possible
            if self.PDBID is None:
                # when a custom structure is used, there is no
                # default filename: the user should provide it
                raise ValueError('Please provide a filename.')
            filename = 'PDBfeatures-' + self.PDBID + '.pkl'
        pickle_path = os.path.join(folder, filename)
        # do not store GNM and ANM instances.
        # If a valid PDBID is present, do not store parsed PDB
        # as well, since it can be easily fetched again
        cache = (self._pdb, self._gnm, self._anm)
        if self.PDBID is not None:
            self._pdb = None
        self._gnm = {}
        self._anm = {}
        for env in ['chain', 'reduced', 'sliced']:
            self._gnm[env] = {chID: None for chID in self.chids}
            self._anm[env] = {chID: None for chID in self.chids}
        # write pickle
        pickle.dump(self, open(pickle_path, "wb"))
        # restore temporarily cached data
        self._pdb, self._gnm, self._anm = cache
        LOGGER.info("Pickle '{}' saved.".format(filename))
        return pickle_path

    def resetTimestamp(self):
        self.timestamp = str(datetime.datetime.utcnow())

    def setNumModes(self, n_modes):
        """Sets the number of ENM modes to be computed. If different from
        the number provided at instantiation, any precomputed features will
        be deleted."""
        if n_modes != self.n_modes:
            self.n_modes = n_modes
            self.refresh()
        return

    def _checkNumCalphas(self, ag):
        n_ca = ag.ca.numAtoms()
        if n_ca > MAX_NUM_RESIDUES:
            m = f'Too many C-alphas: {n_ca}. Max. allowed: {MAX_NUM_RESIDUES}'
            raise RuntimeError(m)

    def calcGNM(self, chID, env='chain'):
        """Builds GNM model for the selected chain.

        :arg chID: chain identifier
        :type chID: str
        :arg env: environment model, i.e. ``'chain'``, ``'reduced'`` or
            ``'sliced'``
        :type env: str
        :return: GNM model
        :rtype: :class:`GNM`
        """
        assert env in ['chain', 'reduced', 'sliced']
        gnm_e = self._gnm[env]
        n = self.n_modes
        if gnm_e[chID] is None:
            pdb = self.getPDB()
            if env == 'chain':
                ca = pdb.ca[chID]
                self._checkNumCalphas(ca)
                gnm = GNM()
                gnm.buildKirchhoff(ca)
                gnm.calcModes(n_modes=n)
                gnm_e[chID] = gnm
            else:
                ca = pdb.ca
                self._checkNumCalphas(ca)
                gnm_full = GNM()
                gnm_full.buildKirchhoff(ca)
                if env == 'reduced':
                    sel = 'chain ' + chID
                    gnm, _ = reduceModel(gnm_full, ca, sel)
                    gnm.calcModes(n_modes=n)
                    gnm_e[chID] = gnm
                else:
                    gnm_full.calcModes(n_modes=n)
                    for c in self.chids:
                        sel = 'chain ' + c
                        gnm, _ = sliceModel(gnm_full, ca, sel)
                        gnm_e[c] = gnm
        return self._gnm[env][chID]

    def calcANM(self, chID, env='chain'):
        """Builds ANM model for the selected chain.

        :arg chID: chain identifier
        :type chID: str
        :arg env: environment model, i.e. ``'chain'``, ``'reduced'`` or
            ``'sliced'``
        :type env: str
        :return: ANM model
        :rtype: :class:`ANM`
        """
        assert env in ['chain', 'reduced', 'sliced']
        anm_e = self._anm[env]
        n = self.n_modes
        if anm_e[chID] is None:
            pdb = self.getPDB()
            if env == 'chain':
                ca = pdb.ca[chID]
                self._checkNumCalphas(ca)
                anm = ANM()
                anm.buildHessian(ca)
                anm.calcModes(n_modes=n)
                anm_e[chID] = anm
            else:
                ca = pdb.ca
                self._checkNumCalphas(ca)
                anm_full = ANM()
                anm_full.buildHessian(ca)
                if env == 'reduced':
                    sel = 'chain ' + chID
                    anm, _ = reduceModel(anm_full, ca, sel)
                    anm.calcModes(n_modes=n)
                    anm_e[chID] = anm
                else:
                    anm_full.calcModes(n_modes=n)
                    for c in self.chids:
                        sel = 'chain ' + c
                        anm, _ = sliceModel(anm_full, ca, sel)
                        anm_e[c] = anm
        return self._anm[env][chID]

    def calcGNMfeatures(self, chain='all', env='chain', GNM_PRS=True):
        """Computes GNM-based features.

        :arg chain: chain identifier
        :type chain: str
        :arg env: environment model, i.e. ``'chain'``, ``'reduced'`` or
            ``'sliced'``
        :type env: str
        :arg GNM_PRS: whether or not to compute features based on Perturbation
            Response Scanning analysis
        :type GNM_PRS: bool
        """
        assert env in ['chain', 'reduced', 'sliced']
        assert type(GNM_PRS) is bool
        # list of features to be computed
        features = ['GNM_MSF-'+env]
        if GNM_PRS:
            features += ['GNM_effectiveness-'+env, 'GNM_sensitivity-'+env]
        # compute features (if not precomputed)
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        for chID in chain_list:
            d = self.feats[chID]
            if all([f in d for f in features]):
                continue
            try:
                gnm = self.calcGNM(chID, env=env)
            except Exception as e:
                if (isinstance(e, MemoryError)):
                    msg = 'MemoryError'
                else:
                    msg = str(e)
                for f in features:
                    d[f] = msg
                    LOGGER.warn(msg)
                    continue
            key_msf = 'GNM_MSF-' + env
            if key_msf not in d:
                try:
                    d[key_msf] = calcSqFlucts(gnm)
                except Exception as e:
                    msg = str(e)
                    d[key_msf] = msg
                    LOGGER.warn(msg)
            key_eff = 'GNM_effectiveness-' + env
            if key_eff in features and key_eff not in d:
                key_sns = 'GNM_sensitivity-' + env
                try:
                    prs_mtrx, eff, sns = calcPerturbResponse(gnm)
                    d[key_eff] = eff
                    d[key_sns] = sns
                except Exception as e:
                    msg = str(e)
                    d[key_eff] = msg
                    d[key_sns] = msg
                    LOGGER.warn(msg)
        return

    def calcANMfeatures(self, chain='all', env='chain',
                        ANM_PRS=True, stiffness=True, MBS=False):
        """Computes ANM-based features.

        :arg chain: chain identifier
        :type chain: str
        :arg env: environment model, i.e. ``'chain'``, ``'reduced'`` or
            ``'sliced'``
        :type env: str
        :arg ANM_PRS: whether or not to compute features based on Perturbation
            Response Scanning analysis
        :type ANM_PRS: bool
        :arg stiffness: whether or not to compute stiffness with MechStiff
        :type stiffness: bool
        :arg MBS: whether or not to compute Mechanical Bridging Score
        :type MBS: bool
        """
        assert env in ['chain', 'reduced', 'sliced']
        for k in ANM_PRS, stiffness, MBS:
            assert type(k) is bool
        # list of features to be computed
        features = ['ANM_MSF-'+env]
        if ANM_PRS:
            features += ['ANM_effectiveness-'+env, 'ANM_sensitivity-'+env]
        if MBS:
            features += ['MBS-'+env]
        if stiffness:
            features += ['stiffness-'+env]
        # compute features (if not precomputed)
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        for chID in chain_list:
            d = self.feats[chID]
            if all([f in d for f in features]):
                continue
            try:
                anm = self.calcANM(chID, env=env)
            except Exception as e:
                if (isinstance(e, MemoryError)):
                    msg = 'MemoryError'
                else:
                    msg = str(e)
                for f in features:
                    d[f] = msg
                    LOGGER.warn(msg)
                continue
            key_msf = 'ANM_MSF-' + env
            if key_msf not in d:
                try:
                    d[key_msf] = calcSqFlucts(anm)
                except Exception as e:
                    msg = str(e)
                    d[key_msf] = msg
                    LOGGER.warn(msg)
            key_eff = 'ANM_effectiveness-' + env
            if key_eff in features and key_eff not in d:
                key_sns = 'ANM_sensitivity-' + env
                try:
                    prs_mtrx, eff, sns = calcPerturbResponse(anm)
                    d[key_eff] = eff
                    d[key_sns] = sns
                except Exception as e:
                    msg = str(e)
                    d[key_eff] = msg
                    d[key_sns] = msg
                    LOGGER.warn(msg)
            key_mbs = 'MBS-' + env
            if key_mbs in features and key_mbs not in d:
                try:
                    pdb = self.getPDB()
                    ca = pdb[chID].ca
                    d[key_mbs] = calcMBS(anm, ca, cutoff=15.)
                except Exception as e:
                    msg = str(e)
                    d[key_mbs] = msg
                    LOGGER.warn(msg)
            key_stf = 'stiffness-' + env
            if key_stf in features and key_stf not in d:
                try:
                    pdb = self.getPDB()
                    ca = pdb[chID].ca
                    stiff_mtrx = calcMechStiff(anm, ca)
                    d[key_stf] = np.mean(stiff_mtrx, axis=0)
                except Exception as e:
                    msg = str(e)
                    d[key_stf] = msg
                    LOGGER.warn(msg)
        return

    def _launchDSSP(self, ag):
        LOGGER.info('Running DSSP...')
        LOGGER.timeit('_DSSP')
        pdb_file = writePDB('_temp.pdb', ag, secondary=False)
        try:
            dssp_file = execDSSP(pdb_file, outputname='_temp')
        except EnvironmentError:
            raise EnvironmentError("dssp executable not found: please install "
                                   "with 'sudo apt install dssp'")
        ag = parseDSSP(dssp_file, ag)
        os.remove('_temp.pdb')
        os.remove('_temp.dssp')
        LOGGER.report('DSSP finished in %.1fs.', '_DSSP')
        return ag

    def calcDSSP(self, chain='whole'):
        """Runs DSSP on the PDB structure.

        :arg chain: chain identifier. If ``'whole'``, the whole complex will
            be considered
        :type chain: str
        :return: modified PDB object with DSSP properties added as additional
            attributes, accessible via method :func:`getData()`
        :rtype: :class:`AtomGroup`
        """
        if chain == 'whole':
            # compute DSSP on the whole complex
            ag = self.getPDB()
            if ag.getData('dssp_acc') is None:
                ag = self._launchDSSP(ag)
        else:
            # compute DSSP on single chain
            pdb = self.getPDB()
            ag = pdb[chain].copy()
            ag = self._launchDSSP(ag)
        return ag

    def calcSASA(self, chain='all'):
        """Computes Solvent Accessible Surface Area of single chains
        with DSSP algorithm.

        :arg chain: chain identifier
        :type chain: str
        """
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        for chID in chain_list:
            d = self.feats[chID]
            if 'SASA' not in d:
                #  SASA will be computed on single chains
                try:
                    ag = self.calcDSSP(chain=chID)
                    d['SASA'] = ag.ca.getData('dssp_acc')
                except Exception as e:
                    msg = str(e)
                    d['SASA'] = msg
                    LOGGER.warn(msg)
        return

    def calcDeltaSASA(self, chain='all'):
        """Computes the difference between Solvent Accessible Surface Area of
        an isolated chain and of the same chain seen in the complex.

        :arg chain: chain identifier
        :type chain: str
        """
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain, ]
        if any(['SASA_in_complex' not in self.feats[c] for c in self.chids]):
            try:
                # compute SASA for chains in the complex
                ag = self.calcDSSP(chain='whole')
                for chID in self.chids:
                    SASA_c = ag[chID].ca.getData('dssp_acc')
                    self.feats[chID]['SASA_in_complex'] = SASA_c
                    self.feats[chID].pop('Delta_SASA', None)
            except Exception as e:
                msg = str(e)
                for chID in self.chids:
                    d = self.feats[chID]
                    d['SASA_in_complex'] = msg
                    d['Delta_SASA'] = msg
                    LOGGER.warn(msg)
        for chID in chain_list:
            d = self.feats[chID]
            if 'SASA' not in d:
                # SASA of isolated chains
                try:
                    ag = self.calcDSSP(chain=chID)
                    d['SASA'] = ag.ca.getData('dssp_acc')
                except Exception as e:
                    msg = str(e)
                    d['SASA'] = msg
                    d['Delta_SASA'] = msg
                    LOGGER.warn(msg)
            if 'Delta_SASA' not in d:
                # compute difference between SASA of the isolated chain
                # and SASA of the chain in the complex
                try:
                    d['Delta_SASA'] = d['SASA'] - d['SASA_in_complex']
                except Exception as e:
                    msg = str(e)
                    d['Delta_SASA'] = str(msg)
                    LOGGER.warn(msg)
        return

    def _findIndex(self, chain, resid):
        indices = np.where(self.resids[chain] == resid)[0]
        if len(indices) > 1:
            LOGGER.warn('Multiple ({}) residues with resid {} found.'
                        .format(len(indices), resid))
        return indices

    def calcSelFeatures(self, chain='all', resid=None, sel_feats=None):
        """Computes selected PDB-based features for all chains in the PDB
        structure, for a specific chain or for a single residue. Available
        features are listed in :func:`PDB_FEATS`.

        :arg chain: chain identifier
        :type chain: str
        :arg resid: residue number. If selected, a single chain must be also
            specified
        :type resid: int
        :arg sel_feats: list of feature names. If **None**, all
            :func:`PDB_FEATS` will be computed
        :type env: list of str
        :return: a dictionary, containing names and values (or error messages)
            of selected features, for each chain or residue
        :rtype: dict
        """
        if resid is not None and chain == 'all':
            raise ValueError('Please select a single chain.')
        if sel_feats is None:
            sel_feats = PDB_FEATS
        assert all([f in PDB_FEATS for f in sel_feats])
        # compute only selected features
        if 'Delta_SASA' in sel_feats:
            self.calcDeltaSASA(chain)
        elif 'SASA' in sel_feats:
            self.calcSASA(chain)
        for env in ['chain', 'reduced', 'sliced']:
            s = '-' + env
            l = [f.replace(s, '') for f in sel_feats if f.endswith(s)]
            if l == []:
                continue
            GNM_PRS, ANM_PRS, stiffness, MBS = (False,)*4
            if 'GNM_effectiveness' in l or 'GNM_sensitivity' in l:
                GNM_PRS = True
            if 'ANM_effectiveness' in l or 'ANM_sensitivity' in l:
                ANM_PRS = True
            if 'MBS' in l:
                MBS = True
            if 'stiffness' in l:
                stiffness = True
            self.calcGNMfeatures(chain, env=env, GNM_PRS=GNM_PRS)
            self.calcANMfeatures(chain, env=env, ANM_PRS=ANM_PRS,
                                 MBS=MBS, stiffness=stiffness)
        # return different outputs depending on options
        _feats = {}
        for c in self.chids:
            d = self.feats[c]
            _feats[c] = {k: v for k, v in d.items() if k in sel_feats}
        if chain == 'all':
            return _feats
        elif resid is None:
            return _feats[chain]
        else:
            d = _feats[chain]
            indices = self._findIndex(chain, resid)
            output = {}
            for k in d:
                feat_array = d[k]
                if isinstance(feat_array, str):
                    # return error message instead of array
                    output[k] = feat_array
                else:
                    # return 2-D array
                    output[k] = np.array([feat_array[i] for i in indices])
            return output


def calcPDBfeatures(mapped_SAVs, sel_feats=None, custom_PDB=None,
                    refresh=False, status_file=None, status_prefix=None):
    LOGGER.info('Computing structural and dynamical features '
                'from PDB structures...')
    LOGGER.timeit('_calcPDBFeats')
    if sel_feats is None:
        sel_feats = PDB_FEATS
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
    # define how to report progress
    if status_prefix is None:
        status_prefix = ''
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    if status_file is not None:
        status_file = open(status_file, 'w')
        progress_bar = tqdm(
            [(i, mapped_SAVs[i]) for i in sorting_map], file=status_file,
            bar_format=bar_format+'\n')
    else:
        progress_bar = tqdm(
            [(i, mapped_SAVs[i]) for i in sorting_map], bar_format=bar_format)
    cache = {'PDBID': None, 'chain': None, 'obj': None}
    count = 0
    for indx, SAV in progress_bar:
        count += 1
        if SAV['PDB size'] == 0:
            # SAV could not be mapped to PDB
            _features = np.nan
            SAV_coords = SAV['SAV coords']
            progress_msg = f"{status_prefix}No PDB for SAV '{SAV_coords}'"
        else:
            parsed_PDB_coords = SAV['PDB SAV coords'].split()
            PDBID, chID = parsed_PDB_coords[:2]
            resid = int(parsed_PDB_coords[2])
            progress_msg = status_prefix + \
                f'Analizing mutation site {PDBID}:{chID} {resid}'
            # chID == "?" stands for "empty space"
            chID = " " if chID == "?" else chID
        # report progress
        # LOGGER.info(f"[{count}/{num_SAVs}] {progress_msg}...")
        progress_bar.set_description(progress_msg)
        # compute PDB features, if possible
        if SAV['PDB size'] != 0:
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
    if status_file:
        os.remove(status_file.name)
    return features
