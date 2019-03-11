import numpy as np
import sys

sys.path.append('../../')

# check if rhapsody can be imported correctly
from rhapsody import *

# set folders
pathRhapsodyFolder('./data')
pathEVmutationFolder('./data')

# test cases
test_SAVs = [
  'O00294 496 A T', # "good" SAV where all features are well-defined
  'O00238 31 R H'   # "bad" SAV with no PDB structure (but does have Pfam domain)
]

# initialize a rhapsody object
rh = Rhapsody()

# import precomputed PolyPhen-2 output file
rh.importPolyPhen2output('data/pph2-full.txt')

# we would like to compute all features
rh.setFeatSet('all')

# true labels must be set prior to exporting training data
true_labels = {
  'O00294 496 A T': 1,
  'O00238 31 R H': 0
}
rh.setTrueLabels(true_labels)

# compute all features
m = rh.exportTrainingData()

# final check
precomp_feats = np.load('data/precomputed_features.npy')
errors = []
for i in range(len(m)):
    for field in m.dtype.names:
        v_test = m[i][field]
        v_true = precomp_feats[i][field]
        err = f"Discordant '{field}': '{v_test}' instead of '{v_true}'"
        if field in ['SAV_coords', 'Uniprot2PDB']:
            if v_test != v_true:
                errors.append(err)
        elif not np.allclose(v_test, v_true, equal_nan=True):
            errors.append(err)
assert not errors, '\n'.join(errors)
