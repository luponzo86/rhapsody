from prody import Atomic, parsePDB, writePDB, LOGGER, SETTINGS
from prody import GNM, ANM, calcSqFlucts
from prody import calcPerturbResponse, calcMBS, calcMechStiff
from prody import reduceModel, sliceModel
from prody import execDSSP, parseDSSP
import numpy as np
import pickle
import datetime
import os

__all__ = ['PDBfeatures']

STR_FEATS = ['SASA', 'SASA_in_complex', 'Delta_SASA']
DYN_FEATS = ['GNM_MSF', 'ANM_MSF',
             'GNM_effectiveness', 'GNM_sensitivity',
             'ANM_effectiveness', 'ANM_sensitivity',
             'MBS', 'stiffness'] 
PDB_FEATS = STR_FEATS + [f + e for f in DYN_FEATS
                         for e in ['-chain', '-reduced', '-sliced']]

class PDBfeatures:
    
    def __init__(self, PDB, n_modes='all',
                 recover_pickle=False, **kwargs):
        assert isinstance(PDB, (str, Atomic)), \
               'PDB must be either a PDBID or an Atomic instance.'
        assert type(recover_pickle) is bool
        # definition and initialization of variables
        if isinstance(PDB, str):
            self.PDBID = PDB
            self._pdb  = None
        else:
            self.PDBID = None
            self._pdb  = PDB.copy()
        self.n_modes = n_modes
        self.chids   = None
        self.resids  = None
        self.feats   = None
        self._gnm    = None
        self._anm    = None
        self.timestamp = None
        if recover_pickle:
            try:
                self.recoverPickle(**kwargs)
            except Exception as e:
                LOGGER.warn('Unable to recover pickle: %s' %e)
                self.refresh()
        else:
            self.refresh()
        return

    def getPDB(self):
        if self._pdb is None:
            self._pdb = parsePDB(self.PDBID, model=1)
        return self._pdb

    def refresh(self):
        pdb = self.getPDB()
        self.chids = set(pdb.ca.getChids())
        self.resids = {chID: pdb[chID].ca.getResnums()
                       for chID in self.chids}
        self._gnm  = {}
        self._anm  = {}
        for env in ['chain', 'reduced', 'sliced']:
            self._gnm[env]  = {chID: None for chID in self.chids}
            self._anm[env]  = {chID: None for chID in self.chids}
        self.feats = {chID: {} for chID in self.chids}
        self.timestamp = str(datetime.datetime.utcnow())
        return

    def recoverPickle(self, folder=None, filename=None, days=30, **kwargs):
        if folder is None:
            # define folder where to look for pickles
            folder = SETTINGS.get('rhapsody_local_folder', '.')
        if filename is None:
            # use the default filename, if possible
            if self.PDBID is not None:
                filename = 'PDBfeatures-'+ self.PDBID +'.pkl'
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
        t_old = datetime.datetime.strptime(recovered_self.timestamp, date_format)
        t_now = datetime.datetime.utcnow()
        Delta_t = datetime.timedelta(days=days)
        if t_old + Delta_t < t_now:
            raise RuntimeError('Pickle was too old and was ignored.')
        # import recovered data 
        self.chids  = recovered_self.chids
        self.resids = recovered_self.resids
        self.feats  = recovered_self.feats
        self._gnm   = recovered_self._gnm
        self._anm   = recovered_self._anm
        self.timestamp = recovered_self.timestamp
        LOGGER.info("Pickle '{}' recovered.".format(filename))
        return

    def savePickle(self, folder=None, filename=None):
        if folder is None:
            # define folder where to look for pickles
            folder = SETTINGS.get('rhapsody_local_folder', '.')
        if filename is None:
            # use the default filename, if possible
            if self.PDBID is None:
                # when a custom structure is used, there is no
                # default filename: the user should provide it
                raise ValueError('Please provide a filename.')
            filename = 'PDBfeatures-'+ self.PDBID +'.pkl'
        pickle_path = os.path.join(folder, filename)
        # do not store GNM and ANM instances.
        # If a valid PDBID is present, do not store parsed PDB
        # as well, since it can be easily fetched again
        cache = (self._pdb, self._gnm, self._anm)
        if self.PDBID is not None:
            self._pdb = None
        self._gnm  = {}
        self._anm  = {}
        for env in ['chain', 'reduced', 'sliced']:
            self._gnm[env] = {chID: None for chID in self.chids}
            self._anm[env] = {chID: None for chID in self.chids}
        # write pickle
        pickle.dump(self, open(pickle_path, "wb"))
        # restore temporarily cached data
        self._pdb, self._gnm, self._anm = cache
        LOGGER.info("Pickle '{}' saved.".format(filename))
        return pickle_path

    def setNumModes(self, n_modes):
        """This operation will delete precomputed features.
        """
        if n_modes != self.n_modes:
            self.n_modes = n_modes
            self.refresh()
        return 
        
    def calcGNM(self, chID, env='chain'):
        assert env in ['chain', 'reduced', 'sliced']
        gnm_e =  self._gnm[env]
        n = self.n_modes
        if gnm_e[chID] is None:
            pdb = self.getPDB()
            if env == 'chain':
                ca = pdb.ca[chID]
                gnm = GNM()
                gnm.buildKirchhoff(ca)
                gnm.calcModes(n_modes=n)
                gnm_e[chID] = gnm
            else:
                ca = pdb.ca
                gnm_full = GNM()
                gnm_full.buildKirchhoff(ca)
                if env == 'reduced':
                    sel = 'chain '+ chID
                    gnm, _ = reduceModel(gnm_full, ca, sel)
                    gnm.calcModes(n_modes=n)
                    gnm_e[chID] = gnm
                else:
                    gnm_full.calcModes(n_modes=n)
                    for c in self.chids:
                        sel = 'chain '+ c
                        gnm, _ = sliceModel(gnm_full, ca, sel)
                        gnm_e[c] = gnm
        return self._gnm[env][chID]

    def calcANM(self, chID, env='chain'):
        assert env in ['chain', 'reduced', 'sliced']
        anm_e = self._anm[env]
        n = self.n_modes
        if anm_e[chID] is None:
            pdb = self.getPDB()
            if env == 'chain':
                ca = pdb.ca[chID]
                anm = ANM()
                anm.buildHessian(ca)
                anm.calcModes(n_modes=n)
                anm_e[chID] = anm
            else:
                ca = pdb.ca
                anm_full = ANM()
                anm_full.buildHessian(ca)
                if env == 'reduced':
                    sel = 'chain '+ chID
                    anm, _ = reduceModel(anm_full, ca, sel)
                    anm.calcModes(n_modes=n)
                    anm_e[chID] = anm
                else:
                    anm_full.calcModes(n_modes=n)
                    for c in self.chids:
                        sel = 'chain '+ c
                        anm, _ = sliceModel(anm_full, ca, sel)
                        anm_e[c] = anm
        return self._anm[env][chID]    

    def calcGNMfeatures(self, chain='all', env='chain', GNM_PRS=True):
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
            chain_list = [chain,]
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
                        ANM_PRS=True, MBS=True, stiffness=True):
        assert env in ['chain', 'reduced', 'sliced']
        for k in ANM_PRS, MBS, stiffness:
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
            chain_list = [chain,]
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
        try:
            pdb_file  = writePDB('_temp.pdb', ag, secondary=False)
            dssp_file = execDSSP(pdb_file, outputname='_temp')
            ag = parseDSSP(dssp_file, ag)
        except:
            raise
        finally:
            os.remove('_temp.pdb')
            os.remove('_temp.dssp')
        LOGGER.report('DSSP finished in %.1fs.', '_DSSP')
        return ag
    
    def calcDSSP(self, chain='whole'):
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
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain,]
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
        if chain == 'all':
            chain_list = self.chids
        else:
            chain_list = [chain,]
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
    
    def calcAllFeatures(self, chain='all', resid=None, env='chain', 
                        SASA=True, Delta_SASA=True, GNM_PRS=True, 
                        ANM_PRS=True, MBS=True, stiffness=True):
        if resid is not None and chain == 'all':
            raise ValueError('Please select a single chain.')
        assert env in ['chain', 'reduced', 'sliced']
        for k in SASA, Delta_SASA, GNM_PRS, ANM_PRS, MBS, stiffness:
            assert type(k) is bool
        # compute requested features
        self.calcGNMfeatures(chain, env=env, GNM_PRS=GNM_PRS)
        self.calcANMfeatures(chain, env=env, ANM_PRS=ANM_PRS, 
                             MBS=MBS, stiffness=stiffness)
        if Delta_SASA:
            self.calcDeltaSASA(chain)
        elif SASA:
            self.calcSASA(chain)
        # return different outputs depending on options
        _feats = {}
        for c in self.chids:
            d = self.feats[c]
            _feats[c] = {k:v for k,v in d.items() if k in sel_feats}
        if chain == 'all':
            return _feats
        elif resid is None:
            return _feats[chain]
        else:
            d = _feats[chain]
            indices = self._findIndex(chain, resid)
            output = {}
            for k in d:
                if isinstance(d[k], str):
                    output[k] = np.array([np.nan]*len(indices))
                else:
                    output[k] = np.array([d[k][i] for i in indices])
            return output

    def calcSelFeatures(self, chain='all', resid=None, sel_feats=None):
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
            l = [f.replace(s,'') for f in sel_feats if f.endswith(s)]
            if l == []:
                continue
            GNM_PRS, ANM_PRS, MBS, stiffness = (False,)*4
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
            _feats[c] = {k:v for k,v in d.items() if k in sel_feats}
        if chain == 'all':
            return _feats
        elif resid is None:
            return _feats[chain]
        else:
            d = _feats[chain]
            indices = self._findIndex(chain, resid)
            output = {}
            for k in d:
                if isinstance(d[k], str):
                    output[k] = np.array([np.nan]*len(indices))
                else:
                    output[k] = np.array([d[k][i] for i in indices])
            return output


