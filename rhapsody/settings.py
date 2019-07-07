# -*- coding: utf-8 -*-
"""This module defines a few default configuration parameters and
a function for the initial setup and training of Rhapsody."""

import os
import glob
import tarfile
import urllib.request
import shutil
import sklearn
import numpy as np
import prody as pd
import rhapsody as rd

__all__ = ['DEFAULT_FEATSETS', 'initialSetup', 'getDefaultClassifiers',
           'getSettings']

USERHOME = os.getenv('USERPROFILE') or os.getenv('HOME') or './'
DEFAULT_WORKING_DIR = os.path.join(USERHOME, 'rhapsody')
DEFAULT_EVMUT_DIR = os.path.join(DEFAULT_WORKING_DIR,
                                 'EVmutation_mutation_effects')
EVMUT_URL = 'https://marks.hms.harvard.edu/evmutation/data/effects.tar.gz'
PACKAGE_DATA = os.path.join(rd.__path__[0], 'data.tar.gz')
TRAINING_DATASET = 'precomputed_features-ID_opt.npy'
DEFAULT_CLSF_DIR = f'default_classifiers-sklearn_v{sklearn.__version__}'
DEFAULT_FEATSETS = {
  'full':    ['wt_PSIC', 'Delta_PSIC', 'SASA', 'ANM_MSF-chain',
              'ANM_effectiveness-chain', 'ANM_sensitivity-chain',
              'stiffness-chain', 'entropy', 'ranked_MI', 'BLOSUM'],
  'reduced': ['wt_PSIC', 'Delta_PSIC', 'SASA', 'ANM_MSF-chain',
              'ANM_effectiveness-chain', 'ANM_sensitivity-chain',
              'stiffness-chain', 'BLOSUM'],
  'EVmut':   ['wt_PSIC', 'Delta_PSIC', 'SASA', 'ANM_MSF-chain',
              'ANM_effectiveness-chain', 'ANM_sensitivity-chain',
              'stiffness-chain', 'entropy', 'ranked_MI', 'BLOSUM',
              'EVmut-DeltaE_epist'],
}


def initialSetup(working_dir=None, refresh=False, download_EVmutation=True):
    """Function to be run right after installation for setting up the
    environment and main parameters and for training the default classifiers.
    By default, a working directory  will be created in the user home directory
    (:file:`~/rhapsody/`). Previous configuration data will be recovered.
    Additional data from EVmutation website will be automatically downloaded
    (~1.4GB).

    :arg working_dir: path to a local folder
    :type working_dir: str

    :arg refresh: if **True**, previous trained classifiers will be deleted,
        if found
    :type refresh: bool

    :arg download_EVmutation: if **True**, precomputed EVmutation scores will
        be downloaded (recommended)
    :type download_EVmutation: bool
    """

    pd.LOGGER.info(f'You are running Rhapsody v{rd.__version__}')

    # set working directory
    if working_dir is None:
        # check pre-existing configuration
        old_dir = pd.SETTINGS.get('rhapsody_local_folder')
        if type(old_dir) is str and os.path.isdir(old_dir):
            working_dir = old_dir
            pd.LOGGER.info('Pre-existing working directory detected: '
                           f'{working_dir}')
        else:
            # use default location and create folder if needed
            working_dir = DEFAULT_WORKING_DIR
            if os.path.isdir(working_dir):
                pd.LOGGER.info('Pre-existing working directory detected: '
                               f'{working_dir}')
            else:
                os.mkdir(working_dir)
                pd.LOGGER.info(f'Default working directory set: {working_dir}')
    else:
        working_dir = os.path.abspath(working_dir)
        if os.path.isdir(working_dir):
            pd.LOGGER.info(f'Working directory set: {working_dir}')
        else:
            raise EnvironmentError(f'Invalid working directory: {working_dir}')
    pd.SETTINGS['rhapsody_local_folder'] = working_dir

    # check for pre-existing folder containing trained classifiers
    folder = os.path.join(working_dir, DEFAULT_CLSF_DIR)
    training_dataset = None
    if os.path.isdir(folder) and not refresh:
        pd.LOGGER.info(f'Pre-existing classifiers found: {folder}')
        # check for missing classifiers
        for featset in DEFAULT_FEATSETS:
            fname = os.path.join(folder, featset, 'trained_classifier.pkl')
            if not os.path.isfile(fname):
                raise IOError(f"Missing classifier: '{featset}'. Please "
                              f'delete folder {folder} and rerun setup.')
    else:
        # delete old classifiers and train new ones
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        pd.LOGGER.info(f'Classifiers folder created: {folder}')
        # delete EVmutation metrics as well, that must be updated
        pd.SETTINGS.pop('EVmutation_metrics')
        # import training dataset included with package
        tar = tarfile.open(PACKAGE_DATA, "r:gz")
        tar.extractall(path=working_dir)
        tar.close()
        fname = os.path.join(working_dir, TRAINING_DATASET)
        training_dataset = np.load(fname)
        os.remove(fname)
        # train new default classifiers
        pd.LOGGER.info('')
        for name, featset in DEFAULT_FEATSETS.items():
            clsf_folder = os.path.join(folder, name)
            os.mkdir(clsf_folder)
            logfile = os.path.join(clsf_folder, 'RF_training.log')
            # run training procedure
            pd.LOGGER.info(f'Training {name} classifier...')
            pd.LOGGER.start(logfile)
            fields = ['SAV_coords', 'true_label'] + featset
            rd.trainRFclassifier(training_dataset[fields])
            # move trained classifier and figures into folder
            output_files = ['predictions_distribution.png',
                            'pathogenicity_prob.png',
                            'ROC.png',
                            'feat_importances.png',
                            'trained_classifier.pkl', ]
            for file in output_files:
                shutil.move(file, clsf_folder)
            pd.LOGGER.close(logfile)
        pd.LOGGER.info('')

    # check EVmutation metrics
    metrics = pd.SETTINGS.get('EVmutation_metrics', default={})
    if 'AUROC' in metrics:
        pd.LOGGER.info(f'Pre-existing EVmutation metrics found.')
    else:
        # compute EVmutation metrics from included training dataset
        if training_dataset is None:
            tar = tarfile.open(PACKAGE_DATA, "r:gz")
            tar.extractall(path=working_dir)
            tar.close()
            fname = os.path.join(working_dir, TRAINING_DATASET)
            training_dataset = np.load(fname)
            os.remove(fname)
        if 'EVmut-DeltaE_epist' not in training_dataset.dtype.names:
            pd.SETTINGS['EVmutation_metrics'] = np.nan
            pd.LOGGER.warn('Unable to compute EVmutation cutoff: '
                           'precomputed scores not found.')
        else:
            sel = ~np.isnan(training_dataset['EVmut-DeltaE_epist'])
            # NB: EVmutation score and pathogenicity are anti-correlated
            true_labels = training_dataset['true_label'][sel]
            EVmut_predictor = -training_dataset['EVmut-DeltaE_epist'][sel]
            metrics = rd.calcMetrics(true_labels, EVmut_predictor)
            pd.SETTINGS['EVmutation_metrics'] = metrics
            pd.LOGGER.info(f'EVmutation metrics computed.')

    # fetch EVmutation precomputed data, if needed
    folder = pd.SETTINGS.get('EVmutation_local_folder')
    if type(folder) is str and os.path.isdir(folder):
        pd.LOGGER.info(f'EVmutation folder found: {folder}')
    else:
        folder = DEFAULT_EVMUT_DIR
        if os.path.isdir(DEFAULT_EVMUT_DIR):
            pd.LOGGER.info(f'EVmutation folder found: {folder}')
        elif download_EVmutation:
            pd.LOGGER.info(f'Downloading EVmutation data...')
            # download tar.gz file and save it locally
            tgz = os.path.join(working_dir, 'effects.tar.gz')
            with urllib.request.urlopen(EVMUT_URL) as r, open(tgz, 'wb') as f:
                shutil.copyfileobj(r, f)
            # extract archive
            tar = tarfile.open(tgz, "r:gz")
            tar.extractall(path=folder)
            tar.close()
            os.remove(tgz)
            pd.LOGGER.info(f'EVmutation folder set: {folder}')
        else:
            folder = None
            msg = ('For full functionality, please consider downloading '
                   f'EVmutation data from {EVMUT_URL} and then set up the '
                   'relative path in the configuration file.')
            pd.LOGGER.warn(msg)
    pd.SETTINGS['EVmutation_local_folder'] = folder

    # check if DSSP is installed
    which = pd.utilities.which
    if which('dssp') is None and which('mkdssp') is None:
        msg = ('For full functionality, please consider installing DSSP, '
               'for instance by typing in a Linux terminal: '
               "'sudo apt install dssp'")
        pd.LOGGER.warn(msg)
    else:
        pd.LOGGER.info('DSSP is installed on the system.')

    pd.SETTINGS.save()
    pd.LOGGER.info('Setup complete.')

    return getSettings(print=False)


def getDefaultClassifiers():
    """Returns a dictionary with the paths to the three default classifiers
    (``'full'``, ``'reduced'`` and ``'EVmut'``)
    """
    working_dir = pd.SETTINGS.get('rhapsody_local_folder')
    clsf_folder = os.path.join(working_dir, DEFAULT_CLSF_DIR)

    def_clsfs = {fs: os.path.join(clsf_folder, fs, 'trained_classifier.pkl')
                 for fs in DEFAULT_FEATSETS}

    if any([not os.path.isfile(c) for c in def_clsfs.values()]):
        raise IOError('One or more default classifiers are missing. '
                      'Please rerun initialSetup()')
    else:
        return def_clsfs


def getSettings(print=True):
    """Returns and prints essential information about the current Rhapsody
    configuration, such as the location of working directory and default
    classifiers
    """

    config_dict = {}

    for entry in ['rhapsody_local_folder', 'EVmutation_local_folder']:
        config_dict[entry] = pd.SETTINGS.get(entry)

    EVmut_metrics = pd.SETTINGS.get('EVmutation_metrics', default={})
    config_dict['EVmutation_metrics'] = EVmut_metrics

    def_clsfs = getDefaultClassifiers()
    for fs, path in def_clsfs.items():
        fs += ' classifier'
        config_dict[fs] = path

    if print:
        entries = ['rhapsody_local_folder', 'EVmutation_local_folder'] \
                  + [f'{c} classifier' for c in def_clsfs]
        for entry in entries:
            pd.LOGGER.info(f'{entry:24}: {config_dict[entry]}')
        if 'AUROC' in EVmut_metrics:
            pd.LOGGER.info('EVmutation_metrics      : <computed>')
        else:
            pd.LOGGER.info('EVmutation_metrics      : <missing>')

    return config_dict
