from prody import LOGGER
from .rhapsody import *

__all__ = ['rhapsody']

def rhapsody(input_obj, classifier, input_type='SAVs', 
             custom_PDB=None, aux_classifier=None):
    """'input_obj' can be:
    * a filename, a list/tuple of strings or a single string, containing SAV 
      coordinates, with the format "P17516 135 G E" (input_type='SAVs', default)
    * a filename of the output from PolyPhen-2, usually named "pph2-full.txt" 
      (input_type='PP2')
    * a string of Uniprot coordinates with unspecified variant, for performing
      simulated mutagenesis experiment (input_type='scanning'). Possible formats
      are: 'P17516 135' for a single site scanning, and 'P17516' for a complete
      sequence scanning.
    """
    assert input_type in ('SAVs', 'scanning', 'PP2')


    # initialize object that will contain all results and predictions
    r = Rhapsody()

    # import classifier and feature set from pickle
    r.importClassifier(classifier)

    # import custom PDB structure
    if custom_PDB is not None:
        r.setCustomPDB(custom_PDB)

    # obtain or import PolyPhen-2 results
    if input_type == 'SAVs':
        # 'input_obj' is a filename, list, tuple or string 
        # providing SAV coordinates
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

    # if available, compute additional predictions from a subset of features
    if aux_classifier is not None:
        try:
            r.calcAuxPredictions(aux_classifier)
        except Exception as e:
            LOGGER.warn('Unable to compute auxiliary predictions: {}'.format(e))

    # save pickle
    r.savePickle()

    return r
