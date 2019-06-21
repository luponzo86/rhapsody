# -*- coding: utf-8 -*-
"""This module defines standard interfaces and functions for running Rhapsody
prediction algorithm."""

from prody import LOGGER
from .settings import getDefaultClassifiers
from .rhapsody import Rhapsody

__all__ = ['rhapsody']


def rhapsody(input_obj, input_type='SAVs', custom_PDB=None, force_env=None,
             main_classifier=None, aux_classifier=None, log=True):
    """Obtain Rhapsody pathogenicity predictions on a list of human missense
    variants ([ref]_)

    :arg input_obj: Single Amino Acid Variants (SAVs) Uniprot coordinates

      - if *input_type* = ``'SAVs'`` (default), it should be a filename, a
        string or a list/tuple of strings, containing Uniprot SAV coordinates,
        with the format ``'P17516 135 G E'``
      - if *input_type* = ``'scanning'``, it should be a string identifying a
        Uniprot sequence (e.g. ``'P17516'``) or a specific site in a sequence
        (e.g. ``'P17516 135'``). All possible 19 amino acid substitutions at
        the specified positions on the sequence will be analyzed
      - if *input_type* = ``'PP2'``, it should be a filename containing the
        output from PolyPhen-2, usually named :file:`pph2-full.txt`
    :type input_obj: str, list

    :arg input_type: ``'SAVs'``, ``'scanning'`` or ``'PP2'``
    :type input_type: str

    :arg custom_PDB: a PDBID, a filename or an :class:`Atomic` to be used
      for computing structural and dynamical features, instead of the PDB
      structure automatically selected by the program
    :type custom_PDB: str, :class:`AtomGroup`

    :arg input_type: force a specific environment model for GNM/ANM
      calculations, among ``'chain'``, ``'reduced'`` and ``'PP2'``. If **None**
      (default), the model of individual dynamical features will match that
      found in the classifier's feature set
    :type input_type: str

    :arg main_classifier: main classifier's filename. If **None**, the default
      *full* Rhapsody classifier will be used
    :type main_classifier: str

    :arg aux_classifier: auxiliary classifier's filename. If both
      *main_classifier* and *aux_classifier* are **None**, the default
      *reduced* Rhapsody classifier will be used
    :type aux_classifier: str

    :arg log: if **True**, log messages will be saved in
      :file:`rhapsody-log.txt`
    :type log: str

    .. [ref] Ponzoni L, Bahar I. Structural dynamics is a determinant of
      the functional significance of missense variants. *PNAS* **2018**
      115 (16) 4164-4169.
    """
    assert input_type in ('SAVs', 'scanning', 'PP2')

    if log:
        LOGGER.start('rhapsody-log.txt')

    # select classifiers
    if main_classifier is None:
        main_classifier = getDefaultClassifiers()['full']
        if aux_classifier is None:
            aux_classifier = getDefaultClassifiers()['reduced']

    # initialize object that will contain all results and predictions
    r = Rhapsody()

    # import classifier and feature set from pickle
    r.importClassifier(main_classifier, force_env=force_env)

    # import custom PDB structure
    if custom_PDB is not None:
        r.setCustomPDB(custom_PDB)

    # obtain or import PolyPhen-2 results
    if input_type == 'SAVs':
        # 'input_obj' is a filename, list, tuple or string
        # containing SAV coordinates
        r.queryPolyPhen2(input_obj)
    elif input_type == 'scanning':
        # 'input_obj' is a Uniprot accession number identifying a sequence,
        # with or without a specified position
        r.queryPolyPhen2(input_obj, scanning=True)
    elif input_type == 'PP2':
        # 'input_obj' is a filename containing PolyPhen-2's output
        r.importPolyPhen2output(input_obj)

    # compute needed features
    r.calcFeatures()

    # compute predictions
    r.calcPredictions()
    if aux_classifier is not None:
        # compute additional predictions from a subset of features
        try:
            r.calcAuxPredictions(aux_classifier, force_env=force_env)
            r.printPredictions(format="both",
                               filename='rhapsody-predictions-full.txt')
        except Exception as e:
            LOGGER.warn(f'Unable to compute auxiliary predictions: {e}')

    # print final predictions
    r.printPredictions(filename='rhapsody-predictions.txt')

    # save pickle
    r.savePickle()

    if log:
        LOGGER.close('rhapsody-log.txt')

    return r
