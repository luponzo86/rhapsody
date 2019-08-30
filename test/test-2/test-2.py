import sys
import os
import shutil
import prody as pd

# check if rhapsody can be imported correctly
sys.path.append('../../')
import rhapsody as rd

# temporarily switch to new set of folders
if not os.path.isdir('workspace'):
    os.mkdir('workspace')
old_rhaps_dir = pd.SETTINGS.get('rhapsody_local_folder')
pd.SETTINGS['rhapsody_local_folder'] = os.path.abspath('./workspace')

# train classifiers
rd.initialSetup(download_EVmutation=False)
shutil.copyfile('./data/UniprotMap-P01112.pkl',
                './workspace/pickles/UniprotMap-P01112.pkl')

# let's run a saturation mutagenesis test with a custom PDB structure
os.chdir('workspace')
rh = rd.rhapsody('../data/pph2-full.txt', query_type='PolyPhen2',
                 main_classifier=rd.getDefaultClassifiers()['reduced'],
                 custom_PDB='../data/RAS_customPDB.pdb')

# print figure
rd.print_sat_mutagen_figure('rhapsody-figure.png', rh, html=True,
                            EVmutation=False)

# restore previous settings
if old_rhaps_dir is not None:
    pd.SETTINGS['rhapsody_local_folder'] = old_rhaps_dir
    pd.SETTINGS.save()

# final check
assert os.path.isfile('rhapsody-figure.png')
