# -*- coding: utf-8 -*-
"""This module defines a class and relative functions for mapping Uniprot
sequences to PDB and Pfam databases."""

import os
import re
import pickle
import datetime
import time
import numpy as np
import prody as pd
from prody import LOGGER, SETTINGS
from prody.utilities import openURL
from tqdm import tqdm
from Bio.pairwise2 import align as bioalign
from Bio.pairwise2 import format_alignment
from Bio.SubsMat import MatrixInfo as matlist

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

__all__ = ['queryUniprot', 'UniprotMapping', 'mapSAVs2PDB',
           'seqScanning', 'printSAVlist']


def queryUniprot(*args, n_attempts=3, dt=1, **kwargs):
    """
    Redefine prody function to check for no internet connection
    """
    attempt = 0
    while attempt < n_attempts:
        try:
            print(f'attempt {attempt}')
            _ = openURL('http://www.uniprot.org/')
            break
        except:
            attempt += 1
            time.sleep((attempt+1)*dt)
    else:
        _ = openURL('http://www.uniprot.org/')
    return pd.queryUniprot(*args, **kwargs)


class UniprotMapping:

    def __init__(self, acc, recover_pickle=False, **kwargs):
        self.acc = self._checkAccessionNumber(acc)
        self.uniq_acc = None
        self.fullRecord = None
        self.sequence = None
        self.PDBrecords = None
        self.PDBmappings = None
        self.customPDBmappings = None
        self._align_algo_args = None
        self._align_algo_kwargs = None
        self.timestamp = None
        self.Pfam = None
        assert type(recover_pickle) is bool
        if recover_pickle:
            try:
                self.recoverPickle(**kwargs)
            except Exception as e:
                LOGGER.warn(f'Unable to recover pickle: {e}')
                self.refresh()
        else:
            self.refresh()

    def refresh(self):
        """Refresh imported Uniprot records and mappings, and
        delete precomputed alignments.
        """
        # import Uniprot record and official accession number
        self.fullRecord = queryUniprot(self.acc)
        self.uniq_acc = self.fullRecord['accession   0']
        # import main sequence and PDB records
        rec = self.fullRecord
        self.sequence = rec['sequence   0'].replace("\n", "")
        self.PDBrecords = [rec[key] for key in rec.keys()
                           if key.startswith('dbRef') and 'PDB' in rec[key]]
        # parse PDB records into PDB mappings, easier to access
        self._initiatePDBmappings()
        # set remaining attributes
        self.customPDBmappings = []
        self._align_algo_args = ['localxs', -0.5, -0.1]
        self._align_algo_kwargs = {'one_alignment_only': True}
        self.timestamp = str(datetime.datetime.utcnow())
        self.Pfam = None
        return

    def getFullRecord(self):
        """Returns the output from :func:`.queryUniprot`"""
        return self.fullRecord

    def getPDBrecords(self):
        """Returns a dictionary containing only the 'dbReference' records
        relative to PDB, extracted from the full Uniprot record.
        """
        return self.PDBrecords

    def getPDBmappings(self, PDBID=None):
        """Returns a list of dictionaries, with mappings of the Uniprot
        sequence onto single PDB chains. For each PDB chain, the residue
        intervals retrieved from the Uniprot database are parsed into a list
        of tuples ('chain_sel') corresponding to endpoints of individual
        segments. NB: '@' stands for 'all chains', following Uniprot naming
        convention.
        """
        if PDBID is None:
            return self.PDBmappings
        # retrieve record for given PDBID
        PDBID = PDBID.upper()
        recs = [d for d in self.PDBmappings if d['PDB'] == PDBID]
        # there should be only one record for a given PDBID
        if len(recs) == 0:
            raise ValueError(f'PDBID {PDBID} not found in Uniprot record.')
        if len(recs) > 1:
            m = f"Multiple entries in Uniprot record for PDBID {PDBID}. "
            m += "Only the first one will be considered."
            LOGGER.warn(m)
        return recs[0]

    def alignSinglePDB(self, PDBID, chain='longest'):
        """Aligns the Uniprot sequence with the sequence from the given
        PDB entry.
        """
        PDBrecord = self.getPDBmappings(PDBID)
        if PDBrecord['chain_seq'] is None:
            raise RuntimeError("Unable to parse PDB.")
        # retrieve chain mappings. Format: {'A': [(1, 10), (15, 100)]}
        mappings = PDBrecord['chain_sel']
        # retrieve list of chains from Uniprot record for given PDBID
        all_chains = set(mappings.keys())
        if '@' in all_chains:
            all_chains = PDBrecord['chain_seq'].keys()
        # select chains to be aligned
        chains_to_align = []
        if chain == 'longest':
            # align only the longest chain in the PDB file
            nCA_max = 0
            for c in sorted(all_chains):
                nCA = len(PDBrecord['chain_res'][c])
                if nCA > nCA_max:
                    nCA_max = nCA
                    chains_to_align = [c]
        elif chain == 'all' or chain == '@':
            # align all chains
            chains_to_align = list(all_chains)
        elif chain in all_chains:
            # align only the requested chain
            chains_to_align = [chain]
        else:
            raise ValueError(f'chain {chain} not found in Uniprot record.')
        # align selected chains with BioPython module pairwise2
        self._calcAlignments(PDBID, chains_to_align)
        # return alignments and maps of selected chains
        rec = [d for d in self.PDBmappings if d['PDB'] == PDBID][0]
        sel_alignms = {c: rec['alignments'][c] for c in chains_to_align}
        sel_maps = {c: rec['maps'][c] for c in chains_to_align}
        return sel_alignms, sel_maps

    def alignCustomPDB(self, PDB, chain='all', title=None, recover=False):
        """Aligns the Uniprot sequence with the sequence from the given PDB.
        """
        assert isinstance(PDB, (str, pd.Atomic)), \
            'PDB must be a PDBID or an Atomic instance (e.g. AtomGroup).'
        assert isinstance(chain, str) or all(
            isinstance(s, str) for s in chain), \
            "'chain' must be a string or a list of strings."
        assert isinstance(title, str) or title is None
        # parse/import pdb and assign title
        if isinstance(PDB, str):
            pdb = pd.parsePDB(PDB, subset='calpha')
            if title is None:
                title = os.path.basename(PDB.strip())
                title = title.replace(' ', '_')
        else:
            pdb = PDB.ca
            if title is None:
                title = PDB.getTitle()
        # check if a record is already present
        rec = [d for d in self.customPDBmappings if d['PDB'] == title]
        if recover and len(rec) > 1:
            raise RuntimeError('Multiple records found with same ID.')
        elif recover and len(rec) == 1:
            customPDBrecord = rec[0]
        else:
            # create record for custom PDB
            customPDBrecord = {
                'PDB': title,
                'chain_res': {},
                'chain_seq': {},
                'warnings': []
            }
            self.customPDBmappings.append(customPDBrecord)
        # check given chain list
        all_chains = set(pdb.getChids())
        if chain == 'all' or chain == '@':
            chains_to_align = list(all_chains)
        elif type(chain) is list:
            chains_to_align = chain
        else:
            chains_to_align = [chain, ]
        invalid_chIDs = [c for c in chains_to_align if c not in all_chains]
        if invalid_chIDs != []:
            raise ValueError('Invalid chain: {}.'.format(invalid_chIDs))
        # store resids and sequence of selected chains
        for c in chains_to_align:
            if c in customPDBrecord['chain_res']:
                continue

            customPDBrecord['chain_res'][c] = pdb[c].getResnums()
            customPDBrecord['chain_seq'][c] = pdb[c].getSequence()
        # align selected chains with BioPython module pairwise2
        self._calcCustomAlignments(title, chains_to_align)
        return customPDBrecord

    def alignAllPDBs(self, chain='longest'):
        """Aligns the Uniprot sequence with the sequences of all PDBs in the
        Uniprot record.
        """
        assert chain in ['longest', 'all']
        PDBIDs_list = [d['PDB'] for d in self.PDBmappings]
        for PDBID in PDBIDs_list:
            try:
                _ = self.alignSinglePDB(PDBID, chain=chain)
            except:
                continue
        return self.PDBmappings

    def mapSingleResidue(self, resid, check_aa=False, depth='best'):
        """Map a single amino acid in a Uniprot sequence to PDBs.
        If 'check_aa' is True, it will return only PDB residues with the
        wild-type amino acid.
        If 'depth' is 'matching', it will use info from Uniprot record to
        determine which PDBs contain the given residue, and if 'depth' is 'best'
        only the longest chain will be considered and printed, to save time.
        If 'depth' is all, it will perform a thorough search among all PDBs (slow).
        The matching PDB residues will be sorted, in descending order, according
        to the identity of the relative chain with the Uniprot sequence.
        """
        assert 1 <= resid <= len(self.sequence), \
            'Index out of range: sequence length is {}.'.format(len(self.sequence))
        assert type(check_aa) is bool
        if check_aa:
            aa = self.sequence[resid-1]
        else:
            aa = None
        assert depth in ['best', 'matching', 'all']
        matches = []
        if depth in ['best', 'matching']:
            # trust Uniprot database and find PDBs containing the given resid
            # according to Uniprot records
            for PDBrecord in self.PDBmappings:
                PDBID = PDBrecord['PDB']
                chain_sel = PDBrecord['chain_sel']
                # e.g. 'chain_sel': {'A': [(1, 9), (15, 20)]}
                if chain_sel is None:
                    # add all chains anyway, if possible
                    if PDBrecord['chain_seq'] is not None:
                        chainIDs = PDBrecord['chain_seq'].keys()
                    else:
                        chainIDs = []
                    for chainID in chainIDs:
                        matches.append((PDBID, chainID, -999))
                else:
                    for chainID, intervals in chain_sel.items():
                        if None in intervals:
                            # range is undefined, add it anyway
                            matches.append((PDBID, chainID, -999))
                        elif np.any([i[0] <= resid <= i[1] for i in intervals]):
                            length = sum([i[1]-i[0]+1 for i in intervals])
                            matches.append((PDBID, chainID, length))
                # sort first by length, then by PDBID and chainID
                matches.sort(key=lambda x: (-x[2], x[0], x[1]))
        else:
            # don't trust Uniprot record: select all PDBs for
            # alignment to find those containing the given resid
            for PDBrecord in self.PDBmappings:
                PDBID = PDBrecord['PDB']
                for chainID in PDBrecord['chain_sel']:
                    matches.append((PDBID, chainID, -999))
        # now align selected chains to find actual hits
        hits = []
        for PDBID, chainID, _ in matches:
            try:
                als, maps = self.alignSinglePDB(PDBID, chain=chainID)
            except:
                continue
            if chainID == '@':
                c_list = sorted(maps.keys())
            else:
                c_list = [chainID]
            for c in c_list:
                hit = maps[c].get(resid)
                if hit is None:
                    # resid is not found in the chain
                    continue
                elif aa is not None and hit[1] != aa:
                    # resid is in the chain but has wrong aa type
                    continue
                else:
                    identity = sum([1 for a1, a2 in zip(als[c][0], als[c][1])
                                    if a1 == a2])
                    hits.append((PDBID, c, hit[0], hit[1], identity))
            if depth == 'best' and len(hits) > 0:
                # stop after finding first hit
                break
        # sort hits first by identity, then by PDBID and chainID
        hits.sort(key=lambda x: (-x[4], x[0], x[1]))
        if depth == 'best':
            hits = hits[:1]
        return hits

    def mapSingleRes2CustomPDBs(self, resid, check_aa=False):
        """Map an amino acid in the Uniprot sequence to aligned custom PDBs.
        If 'check_aa' is True, it will return only PDB residues with the
        wild-type amino acid.
        """
        assert 1 <= resid <= len(self.sequence), \
            'Index out of range: sequence length is {}.'.format(len(self.sequence))
        assert type(check_aa) is bool
        if check_aa:
            aa = self.sequence[resid-1]
        else:
            aa = None
        hits = []
        for rec in self.customPDBmappings:
            title = rec['PDB']
            als = rec['alignments']
            maps = rec['maps']
            for c in maps.keys():
                hit = maps[c].get(resid)
                if hit is None:
                    # resid is not found in the chain
                    continue
                elif aa is not None and hit[1] != aa:
                    # resid is in the chain but has wrong aa type
                    msg = 'Residue was found in chain {} '.format(c)
                    msg += 'of PDB {} but has wrong aa ({})'.format(title, hit[1])
                    LOGGER.info(msg)
                    continue
                else:
                    identity = sum([1 for a1, a2 in zip(als[c][0], als[c][1])
                                    if a1 == a2])
                    hits.append((title, c, hit[0], hit[1], identity))
        # sort hits first by identity, then by title and chainID
        hits.sort(key=lambda x: (-x[4], x[0], x[1]))
        return hits

    def setAlignAlgorithm(self, align_algorithm=1,
                          gap_open_penalty=-0.5, gap_ext_penalty=-0.1,
                          refresh=True):
        """Set the Biopython alignment algorithm used for aligning
        Uniprot sequence to PDB sequences. All precomputed alignments
        will be deleted.
        """
        assert align_algorithm in [0, 1, 2]
        # delete old alignments
        if refresh:
            self.refresh()
        # set new alignment parameters
        if align_algorithm == 0:
            # use fastest alignment algorithm (gaps are not penalized)
            self._align_algo_args = ['localxx']
        elif align_algorithm == 1:
            # gaps are penalized when opened and extended
            self._align_algo_args = ['localxs',
                                     gap_open_penalty, gap_open_penalty]
        else:
            # slow, high quality alignment, with scoring of mismatching chars
            # based on BLOSUM62 matrix and penalized opened/extended gaps
            self._align_algo_args = ['localds', matlist.blosum62,
                                     gap_open_penalty, gap_open_penalty]
        return

    def savePickle(self, filename=None, folder=None, store_custom_PDBs=False):
        if folder is None:
            folder = SETTINGS.get('rhapsody_local_folder')
            if folder is None:
                folder = '.'
            else:
                folder = os.path.join(folder, 'pickles')
        if filename is None:
            filename = 'UniprotMap-' + self.uniq_acc + '.pkl'
        pickle_path = os.path.join(folder, filename)
        cache = self.customPDBmappings
        if store_custom_PDBs is not True:
            # do not store alignments of custom PDBs
            self.customPDBmappings = []
        # save pickle
        pickle.dump(self, open(pickle_path, "wb"))
        self.customPDBmappings = cache
        LOGGER.info("Pickle '{}' saved.".format(filename))
        return pickle_path

    def recoverPickle(self, filename=None, folder=None, days=30, **kwargs):
        acc = self.uniq_acc
        if acc is None:
            # assume acc is equal to uniq_acc
            acc = self.acc
        if folder is None:
            folder = SETTINGS.get('rhapsody_local_folder')
            if folder is None:
                folder = '.'
            else:
                folder = os.path.join(folder, 'pickles')
        if filename is None:
            # assume acc is equal to uniq_acc
            acc = self.acc
            filename = 'UniprotMap-' + acc + '.pkl'
            pickle_path = os.path.join(folder, filename)
            if not os.path.isfile(pickle_path):
                # import unique accession number
                acc = queryUniprot(self.acc)['accession   0']
                filename = 'UniprotMap-' + acc + '.pkl'
                pickle_path = os.path.join(folder, filename)
        else:
            pickle_path = os.path.join(folder, filename)
        # check if pickle exists
        if not os.path.isfile(pickle_path):
            raise IOError("File '{}' not found".format(filename))
        # load pickle
        recovered_self = pickle.load(open(pickle_path, "rb"))
        if acc not in [recovered_self.acc, recovered_self.uniq_acc]:
            raise ValueError('Accession number in recovered pickle (%s) '
                             % recovered_self.uniq_acc + 'does not match.')
        # check timestamp and ignore pickles that are too old
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        t_old = datetime.datetime.strptime(recovered_self.timestamp,
                                           date_format)
        t_now = datetime.datetime.utcnow()
        Delta_t = datetime.timedelta(days=days)
        if t_old + Delta_t < t_now:
            raise RuntimeError(
                'Pickle {} was too old and was ignored.'.format(filename))
        self.fullRecord = recovered_self.fullRecord
        self.uniq_acc = recovered_self.uniq_acc
        self.sequence = recovered_self.sequence
        self.PDBrecords = recovered_self.PDBrecords
        self.PDBmappings = recovered_self.PDBmappings
        self.customPDBmappings = recovered_self.customPDBmappings
        self._align_algo_args = recovered_self._align_algo_args
        self._align_algo_kwargs = recovered_self._align_algo_kwargs
        self.timestamp = recovered_self.timestamp
        self.Pfam = recovered_self.Pfam
        LOGGER.info("Pickle '{}' recovered.".format(filename))
        return

    def resetTimestamp(self):
        self.timestamp = str(datetime.datetime.utcnow())

    def _checkAccessionNumber(self, acc):
        if '-' in acc:
            acc = acc.split('-')[0]
            message = 'Isoforms are not allowed, the main sequence for ' + \
                      acc + ' will be used instead.'
            LOGGER.warn(message)
        return acc

    def _parseSelString(self, sel_str):
        # example: "A/B/C=15-100, D=30-200"
        # or: "@=10-200"
        parsedSelStr = {}
        for segment in sel_str.replace(' ', '').split(','):
            fields = segment.split('=')
            chains = fields[0].split('/')
            resids = fields[1].split('-')
            try:
                resids = tuple([int(s) for s in resids])
            except Exception:
                # sometimes the interval is undefined,
                # e.g. "A=-"
                resids = None
            for chain in chains:
                parsedSelStr.setdefault(chain, []).append(resids)
        return parsedSelStr

    def _initiatePDBmappings(self):
        illegal_chars = r"[^A-Za-z0-9-@=/,\s]"
        PDBmappings = []
        for singlePDBrecord in self.PDBrecords:
            PDBID = singlePDBrecord.get('PDB').upper()
            mapping = {'PDB': PDBID,
                       'chain_sel': None,
                       'chain_res': None,
                       'chain_seq': None,
                       'warnings': []}
            # import selection string
            sel_str = singlePDBrecord.get('chains')
            if sel_str is None:
                mapping['warnings'].append('Empty selection string.')
            else:
                # check for illegal characters in selection string
                match = re.search(illegal_chars, sel_str)
                if match:
                    chars = re.findall(illegal_chars, sel_str)
                    message = "Illegal characters found in 'chains' " \
                              + 'selection string: ' + ' '.join(chars)
                    mapping['warnings'].append(message)
                else:
                    parsed_sel_str = self._parseSelString(sel_str)
                    mapping['chain_sel'] = parsed_sel_str
            # store resids and sequence of PDB chains
            try:
                pdb = pd.parsePDB(PDBID, subset='calpha')
                mapping['chain_res'] = {}
                mapping['chain_seq'] = {}
                for c in set(pdb.getChids()):
                    mapping['chain_res'][c] = pdb[c].getResnums()
                    mapping['chain_seq'][c] = pdb[c].getSequence()
            except Exception as e:
                mapping['chain_res'] = None
                mapping['chain_seq'] = None
                msg = "Error while parsing PDB: {}".format(e)
                mapping['warnings'].append(msg)
                LOGGER.warn(msg)
            PDBmappings.append(mapping)
        self.PDBmappings = PDBmappings
        if PDBmappings == []:
            LOGGER.warn('No PDB entries have been found '
                        'that map to given sequence.')
        return

    def _align(self, seqU, seqC, PDBresids, print_info=False):
        algo = self._align_algo_args[0]
        args = self._align_algo_args[1:]
        kwargs = self._align_algo_kwargs
        # align Uniprot and PDB sequences
        al = None
        if algo == 'localxx':
            al = bioalign.localxx(seqU, seqC, *args, **kwargs)
        elif algo == 'localxs':
            al = bioalign.localxs(seqU, seqC, *args, **kwargs)
        else:
            al = bioalign.localds(seqU, seqC, *args, **kwargs)
        if print_info is True:
            info = format_alignment(*al[0])
            LOGGER.info(info[:-1])
            idnt = sum([1 for a1, a2 in zip(al[0][0], al[0][1]) if a1 == a2])
            frac = idnt/len(seqC)
            m = "{} out of {} ({:.1%}) residues".format(idnt, len(seqC), frac)
            m += " in the chain are identical to Uniprot amino acids."
            LOGGER.info(m)
        # compute mapping between Uniprot and PDB chain resids
        aligned_seqU = al[0][0]
        aligned_seqC = al[0][1]
        mp = {}
        resid_U = 0
        resindx_PDB = 0
        for i in range(len(aligned_seqU)):
            aaU = aligned_seqU[i]
            aaC = aligned_seqC[i]
            if aaU != '-':
                resid_U += 1
                if aaC != '-':
                    mp[resid_U] = (PDBresids[resindx_PDB], aaC)
            if aaC != '-':
                resindx_PDB += 1
        return al[0][:2], mp

    def _quickAlign(self, seqU, seqC, PDBresids):
        '''Works only if PDB sequence and resids perfectly match
        those found in Uniprot.'''
        s = ['-'] * len(seqU)
        mp = {}
        for resid, aaC in zip(PDBresids, seqC):
            indx = resid-1
            try:
                aaU = seqU[indx]
            except:
                raise RuntimeError('Invalid resid in PDB.')
            if resid in mp:
                raise RuntimeError('Duplicate resid in PDB.')
            elif aaC != aaU:
                raise RuntimeError('Non-WT aa in PDB sequence.')
            else:
                mp[resid] = (resid, aaC)
                s[indx] = aaC
        aligned_seqC = "".join(s)
        return (seqU, aligned_seqC), mp

    def _calcAlignments(self, PDBID, chains_to_align):
        seqUniprot = self.sequence
        PDBrecord = self.getPDBmappings(PDBID)
        alignments = PDBrecord.setdefault('alignments', {})
        maps = PDBrecord.setdefault('maps', {})
        for c in chains_to_align:
            # check for precomputed alignments and maps
            if c in alignments:
                continue
            # otherwise, align and map to PDB resids
            PDBresids = PDBrecord['chain_res'][c]
            seqChain = PDBrecord['chain_seq'][c]
            LOGGER.timeit('_align')
            try:
                a, m = self._quickAlign(seqUniprot, seqChain, PDBresids)
                msg = "Chain {} in {} was quick-aligned".format(c, PDBID)
            except:
                a, m = self._align(seqUniprot, seqChain, PDBresids)
                msg = "Chain {} in {} was aligned".format(c, PDBID)
            LOGGER.report(msg + ' in %.1fs.', '_align')
            # store alignments and maps into PDBmappings
            alignments[c] = a
            maps[c] = m
        return

    def _calcCustomAlignments(self, title, chains_to_align):
        seqUniprot = self.sequence
        PDBrecord = [d for d in self.customPDBmappings
                     if d['PDB'] == title][0]
        alignments = PDBrecord.setdefault('alignments', {})
        maps = PDBrecord.setdefault('maps', {})
        for c in chains_to_align:
            # check for precomputed alignments and maps
            if c in alignments:
                continue
            # otherwise, align and map to PDB resids
            PDBresids = PDBrecord['chain_res'][c]
            seqChain = PDBrecord['chain_seq'][c]
            LOGGER.timeit('_align')
            try:
                a, m = self._quickAlign(seqUniprot, seqChain, PDBresids)
                msg = f"Chain {c} was quick-aligned"
            except:
                LOGGER.info(f"Aligning chain {c} of custom PDB {title}...")
                a, m = self._align(seqUniprot, seqChain, PDBresids,
                                   print_info=True)
                msg = f"Chain {c} was aligned"
            LOGGER.report(msg + ' in %.1fs.', '_align')
            # store alignments and maps into PDBmappings
            alignments[c] = a
            maps[c] = m
        return

    # PFAM methods

    def _searchPfam(self, refresh=False, **kwargs):
        assert type(refresh) is bool
        if refresh is True or self.Pfam is None:
            try:
                self.Pfam = pd.searchPfam(self.uniq_acc, **kwargs)
            except:
                self.Pfam = {}
                raise
        return self.Pfam

    def _sliceMSA(self, msa):
        acc_name = self.fullRecord['name   0']
        # find sequences in MSA related to the given Uniprot name
        indexes = msa.getIndex(acc_name)
        if indexes is None:
            raise RuntimeError('No sequence found in MSA for {}'.format(acc_name))
        elif type(indexes) is not list:
            indexes = [indexes]
        # slice MSA to include only columns from selected sequences
        cols = np.array([], dtype=int)
        arr = msa._getArray()
        for i in indexes:
            cols = np.append(cols, np.char.isalpha(arr[i]).nonzero()[0])
        cols = np.unique(cols)
        arr = arr.take(cols, 1)
        sliced_msa = pd.MSA(arr, title='refined', labels=msa._labels)
        LOGGER.info('Number of columns in MSA reduced to {}.'.format(
            sliced_msa.numResidues()))
        return sliced_msa, indexes

    def _mapUniprot2Pfam(self, PF_ID, msa, indexes):
        def compareSeqs(s1, s2, tol=0.01):
            if len(s1) != len(s2):
                return None
            seqid = sum(np.array(list(s1)) == np.array(list(s2)))
            seqid = seqid/len(s1)
            if (1 - seqid) > tol:
                return None
            return seqid
        # fetch sequences from Pfam (all locations)
        m = [None]*len(self.sequence)
        sP_list = []
        for i in indexes:
            arr = msa[i].getArray()
            cols = np.char.isalpha(arr).nonzero()[0]
            sP = str(arr[cols], 'utf-8').upper()
            sP_list.append((sP, cols))
        # NB: it's not known which msa index corresponds
        # to each location
        for l in self.Pfam[PF_ID]['locations']:
            r_i = int(l['start']) - 1
            r_f = int(l['end']) - 1
            sU = self.sequence[r_i:r_f+1]
            max_seqid = 0.
            for sP, cols in sP_list:
                seqid = compareSeqs(sU, sP)
                if seqid is None:
                    continue
                if seqid > max_seqid:
                    max_seqid = seqid
                    m[r_i:r_f+1] = cols
                if np.allclose(seqid, 1):
                    break
        return {k: v for k, v in enumerate(m) if v is not None}

    def calcEvolProperties(self, resid='all', refresh=False, folder=None,
                           max_cols=None, max_seqs=25000, **kwargs):
        ''' Computes Evol properties, i.e. Shannon entropy, Mutual
        Information and Direct Information, from Pfam Multiple
        Sequence Alignments, for a given residue.
        '''
        assert type(refresh) is bool
        # recover Pfam mapping (if not found already)
        self._searchPfam(refresh=refresh)
        if resid == 'all':
            PF_list = self.Pfam.keys()
        else:
            # get list of Pfam domains containing resid
            PF_list = [k for k in self.Pfam if any([
                    resid >= int(segment['start']) and
                    resid <= int(segment['end'])
                    for segment in self.Pfam[k]['locations']
                ])
            ]
            if len(PF_list) == 0:
                raise RuntimeError(f'No Pfam domain for resid {resid}.')
            if len(PF_list) > 1:
                LOGGER.warn(f'Residue {resid} is found in multiple '
                            '({}) Pfam domains.'.format(len(PF_list)))
        if folder is None:
            folder = SETTINGS.get('rhapsody_local_folder')
            if folder is None:
                folder = '.'
            else:
                folder = os.path.join(folder, 'pickles')
        # iterate over Pfam families
        for PF in PF_list:
            d = self.Pfam[PF]
            # skip if properties are pre-computed
            if not refresh and d.get('mapping') is not None:
                continue
            d['mapping'] = None
            d['ref_MSA'] = None
            d['entropy'] = np.nan
            d['MutInfo'] = np.nan
            d['DirInfo'] = np.nan
            try:
                LOGGER.info('Processing {}...'.format(PF))
                # fetch & parse MSA
#               fname = PF + '_full.sth'
#               fullname = os.path.join(folder, fname)
#               if not os.path.isfile(fullname):
#                   f = fetchPfamMSA(PF)
#                   shutil.move(f, folder)
#               msa = parseMSA(fullname, **kwargs)
                # fetch & parse MSA without saving downloaded MSA
                f = pd.fetchPfamMSA(PF)
                msa = pd.parseMSA(f, **kwargs)
                os.remove(f)
                # slice MSA to match all segments of the Uniprot sequence
                sliced_msa, indexes = self._sliceMSA(msa)
#               if max_cols is not None and sliced_msa.numResidues() > max_cols:
#                   raise Exception('Unable to compute DI: MSA has ' +\
#                                   'too many columns (max: {}).'.format(max_cols))
                # get mapping between Uniprot sequence and Pfam domain
                d['mapping'] = self._mapUniprot2Pfam(PF, sliced_msa, indexes)
            except Exception as e:
                LOGGER.warn('{}: {}'.format(PF, e))
                d['mapping'] = str(e)
                continue
            try:
                # refine MSA ('seqid' param. is set as in PolyPhen-2)
                rowocc = 0.6
                while True:
                    sliced_msa = pd.refineMSA(sliced_msa, rowocc=rowocc)
                    rowocc += 0.02
                    if sliced_msa.numSequences() <= max_seqs or rowocc >= 1:
                        break
                ref_msa = pd.refineMSA(sliced_msa, seqid=0.94, **kwargs)
                d['ref_MSA'] = ref_msa
                # compute evolutionary properties
                d['entropy'] = pd.calcShannonEntropy(ref_msa)
                d['MutInfo'] = pd.buildMutinfoMatrix(ref_msa)
                # d['DirInfo'] = buildDirectInfoMatrix(ref_msa)
            except Exception as e:
                LOGGER.warn('{}: {}'.format(PF, e))
        return {k: self.Pfam[k] for k in PF_list}


def mapSAVs2PDB(SAV_coords, custom_PDB=None, refresh=False,
                status_file=None, status_prefix=None):
    LOGGER.info('Mapping SAVs to PDB structures...')
    LOGGER.timeit('_map2PDB')
    # sort SAVs, so to group together those
    # with identical accession number
    accs = [s.split()[0] for s in SAV_coords]
    sorting_map = np.argsort(accs)
    # define a structured array
    PDBmap_dtype = np.dtype([
        ('orig. SAV coords', 'U25'),
        ('unique SAV coords', 'U25'),
        ('PDB SAV coords', 'U100'),
        ('PDB size', 'i')])
    nSAVs = len(SAV_coords)
    mapped_SAVs = np.zeros(nSAVs, dtype=PDBmap_dtype)
    # define how to report progress
    if status_prefix is None:
        status_prefix = ''
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    if status_file is not None:
        status_file = open(status_file, 'w')
        progress_bar = tqdm(
            [(i, SAV_coords[i]) for i in sorting_map], file=status_file,
            bar_format=bar_format+'\n')
    else:
        progress_bar = tqdm(
            [(i, SAV_coords[i]) for i in sorting_map], bar_format=bar_format)
    # map to PDB using Uniprot class
    cache = {'acc': None, 'obj': None}
    count = 0
    for indx, SAV in progress_bar:
        count += 1
        acc, pos, aa1, aa2 = SAV.split()
        pos = int(pos)
        # report progress
        progress_msg = f"{status_prefix}Mapping SAV '{SAV}' to PDB"
        # LOGGER.info(f"[{count}/{nSAVs}] {progress_msg}...")
        progress_bar.set_description(progress_msg)
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
    if status_file:
        os.remove(status_file.name)
    return mapped_SAVs


def seqScanning(Uniprot_coord, sequence=None):
    '''Returns a list of SAVs. If the string 'Uniprot_coord' is just a
    Uniprot ID, the list will contain all possible amino acid substitutions
    at all positions in the sequence. If 'Uniprot_coord' also includes a
    specific position, the list will only contain all possible amino acid
    variants at that position. If 'sequence' is 'None' (default), the
    sequence will be downloaded from Uniprot.
    '''
    assert isinstance(Uniprot_coord, str), "Must be a string."
    coord = Uniprot_coord.upper().strip().split()
    assert len(coord) < 3, "Invalid format. Examples: 'Q9BW27' or 'Q9BW27 10'."
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    if sequence is None:
        Uniprot_record = queryUniprot(coord[0])
        sequence = Uniprot_record['sequence   0'].replace("\n", "")
    else:
        assert isinstance(sequence, str), "Must be a string."
        sequence = sequence.upper()
        assert set(sequence).issubset(aa_list), "Invalid list of amino acids."
    if len(coord) == 1:
        # user asks for full-sequence scanning
        positions = range(len(sequence))
    else:
        # user asks for single-site scanning
        site = int(coord[1])
        positions = [site - 1]
        # if user provides only one amino acid as 'sequence', interpret it
        # as the amino acid at the specified position
        if len(sequence) == 1:
            sequence = sequence*site
        else:
            assert len(sequence) >= site, ("Requested position is not found "
                                           "in input sequence.")
    SAV_list = []
    acc = coord[0]
    for i in positions:
        wt_aa = sequence[i]
        for aa in aa_list:
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
            print(line.upper(), file=f)
    return filename
