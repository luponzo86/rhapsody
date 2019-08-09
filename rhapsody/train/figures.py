# -*- coding: utf-8 -*-
"""This module defines functions for generating figures summarizing
results from the training process."""

import os
import numpy as np
from prody import LOGGER

__all__ = ['print_pred_distrib_figure', 'print_path_prob_figure',
           'print_ROC_figure', 'print_feat_imp_figure']


def _try_import_matplotlib():
    try:
        import matplotlib as plt
        plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})
    except ImportError:
        LOGGER.warn('matplotlib is required for generating figures')
        return None
    return plt


def print_pred_distrib_figure(filename, bins, histo, dx, J_opt):
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    matplotlib = _try_import_matplotlib()
    if matplotlib is None:
        return
    else:
        from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(7, 7))
    plt.bar(bins[:-1], histo[0], width=dx, align='edge',
            color='blue', alpha=0.7, label='neutral')
    plt.bar(bins[:-1], histo[1], width=dx, align='edge',
            color='red',  alpha=0.7, label='deleterious')
    plt.axvline(x=J_opt, color='k', ls='--', lw=1)
    plt.ylabel('distribution')
    plt.xlabel('predicted score')
    plt.legend()
    figure.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    LOGGER.info(f'Predictions distribution saved to {filename}')


def print_path_prob_figure(filename, bins, histo, dx, path_prob,
                           smooth_plot=None, cutoff=200):
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    matplotlib = _try_import_matplotlib()
    if matplotlib is None:
        return
    else:
        from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(7, 7))
    s = np.sum(histo, axis=0)
    v1 = np.where(s >= cutoff, path_prob, 0)
    v2 = np.where(s < cutoff, path_prob, 0)
    plt.bar(bins[:-1], v1, width=dx, align='edge', color='red', alpha=1,
            label='fraction of positives')
    plt.bar(bins[:-1], v2, width=dx, align='edge', color='red', alpha=0.7)
    if smooth_plot is not None:
        plt.plot(smooth_plot[0], smooth_plot[1], color='orange',
                 label='smoothed path. prob.')
    plt.ylabel('')
    plt.xlabel('predicted score')
    plt.ylim((0, 1))
    plt.legend()
    figure.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    LOGGER.info(f'Pathogenicity plot saved to {filename}')


def print_ROC_figure(filename, fpr, tpr, mean_auc):
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    matplotlib = _try_import_matplotlib()
    if matplotlib is None:
        return
    else:
        from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
    plt.plot(fpr, tpr,       linestyle='-',  lw=2, color='r',
             label=f'Mean AUROC = {mean_auc:.3f}')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    LOGGER.info(f'ROC plot saved to {filename}')


def print_feat_imp_figure(filename, feat_imp, featset):
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    matplotlib = _try_import_matplotlib()
    if matplotlib is None:
        return
    else:
        from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(7, 7))
    n = len(feat_imp)
    plt.bar(range(n), feat_imp, align='center', tick_label=featset)
    plt.xticks(rotation='vertical')
    plt.ylabel('feat. importance')
    fig.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    LOGGER.info(f'Feat. importance plot saved to {filename}')
