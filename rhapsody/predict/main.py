# -*- coding: utf-8 -*-
"""This module defines the standard interface for running Rhapsody
prediction algorithm."""

from prody import LOGGER
from ..utils.settings import getDefaultClassifiers
from .core import Rhapsody

__all__ = ['rhapsody']


def rhapsody(query, query_type='SAVs',
             main_classifier=None, aux_classifier=None,
             custom_PDB=None, force_env=None,
             refresh=False, log=True, **kwargs):
    """Obtain Rhapsody pathogenicity predictions on a list of human missense
    variants ([ref]_)

    :arg query: Single Amino Acid Variants (SAVs) in Uniprot coordinates

      - if *query_type* = ``'SAVs'`` (default), it should be a filename, a
        string or a list/tuple of strings, containing Uniprot SAV coordinates,
        with the format ``'P17516 135 G E'``. The string could also be just
        a single Uniprot sequence identifier (e.g. ``'P17516'``), or the
        coordinate of a specific site in a sequence (e.g. ``'P17516 135'``), in
        which case all possible 19 amino acid substitutions at the specified
        positions will be analyzed.
      - if *query_type* = ``'PolyPhen2'``, it should be a filename containing
        the output from PolyPhen-2, usually named :file:`pph2-full.txt`
    :type query: str, list

    :arg query_type: ``'SAVs'`` or ``'PolyPhen2'``
    :type query_type: str

    :arg main_classifier: main classifier's filename. If **None**, the default
      *full* Rhapsody classifier will be used
    :type main_classifier: str

    :arg aux_classifier: auxiliary classifier's filename. If both
      *main_classifier* and *aux_classifier* are **None**, the default
      *reduced* Rhapsody classifier will be used
    :type aux_classifier: str

    :arg custom_PDB: a PDBID, a filename or an :class:`Atomic` to be used
      for computing structural and dynamical features, instead of the PDB
      structure automatically selected by the program
    :type custom_PDB: str, :class:`AtomGroup`

    :arg force_env: force a specific environment model for GNM/ANM
      calculations, among ``'chain'``, ``'reduced'`` and ``'PolyPhen2'``.
      If **None** (default), the model of individual dynamical features will
      match that found in the classifier's feature set
    :type force_env: str

    :arg refresh: if **True**, precomputed features and PDB mappings found in
      the working directory will be ignored and computed again
    :type refresh: str

    :arg log: if **True**, log messages will be saved in
      :file:`rhapsody-log.txt`
    :type log: str

    .. [ref] Ponzoni L, Bahar I. Structural dynamics is a determinant of
      the functional significance of missense variants. *PNAS* **2018**
      115 (16) 4164-4169.
    """

    assert query_type in ['SAVs', 'PolyPhen2'], 'Invalid query type.'

    if log:
        LOGGER.start('rhapsody-log.txt')

    # select classifiers
    if main_classifier is None:
        main_classifier = getDefaultClassifiers()['full']
        if aux_classifier is None:
            aux_classifier = getDefaultClassifiers()['reduced']

    # initialize object that will contain all results and predictions
    r = Rhapsody(**kwargs)

    # import classifiers and feature set from pickle
    r.importClassifiers(main_classifier, aux_classifier, force_env=force_env)

    # import custom PDB structure
    if custom_PDB is not None:
        r.setCustomPDB(custom_PDB)

    # obtain or import PolyPhen-2 results
    if query_type == 'SAVs':
        r.queryPolyPhen2(query)
    elif query_type == 'PolyPhen2':
        r.importPolyPhen2output(query)

    # compute predictions
    r.getPredictions(refresh=refresh)

    # print predictions to file
    r.printPredictions()
    if aux_classifier is not None:
        # print both 'full' and 'reduced' predictions in a more detailed format
        r.printPredictions(
            classifier="both", PolyPhen2=False, EVmutation=False,
            filename='rhapsody-predictions-full_vs_reduced.txt')

    # save pickle
    r.savePickle()

    if log:
        LOGGER.close('rhapsody-log.txt')

    return r
