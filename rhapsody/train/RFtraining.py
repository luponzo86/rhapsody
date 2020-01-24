# -*- coding: utf-8 -*-
"""This module defines functions for training Random Forest classifiers
implementing Rhapsody's classification schemes."""

import pickle
import numpy as np
import numpy.lib.recfunctions as rfn
from collections import Counter
from prody import LOGGER
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from sklearn.utils import resample
from ..utils.settings import DEFAULT_FEATSETS, getDefaultTrainingDataset
from .figures import print_pred_distrib_figure, print_path_prob_figure
from .figures import print_ROC_figure, print_feat_imp_figure

__author__ = "Luca Ponzoni"
__date__ = "December 2019"
__maintainer__ = "Luca Ponzoni"
__email__ = "lponzoni@pitt.edu"
__status__ = "Production"

__all__ = ['calcScoreMetrics', 'calcClassMetrics', 'calcPathogenicityProbs',
           'RandomForestCV', 'trainRFclassifier',
           'extendDefaultTrainingDataset']


def calcScoreMetrics(y_test, y_pred, bootstrap=0, **resample_kwargs):
    '''Compute accuracy metrics of continuous values (optionally bootstrapped)
    '''
    def _calcScoreMetrics(y_test, y_pred):
        # compute ROC and AUROC
        fpr, tpr, roc_thr = roc_curve(y_test, y_pred)
        roc = {'FPR': fpr, 'TPR': tpr, 'thresholds': roc_thr}
        auroc = roc_auc_score(y_test, y_pred)
        # compute optimal cutoff J (argmax of Youden's index)
        diff = np.array([y-x for x, y in zip(fpr, tpr)])
        Jopt = roc_thr[(-diff).argsort()][0]
        # compute Precision-Recall curve and AUPRC
        pre, rec, prc_thr = precision_recall_curve(y_test, y_pred)
        prc = {'precision': pre, 'recall': rec, 'thresholds': prc_thr}
        auprc = average_precision_score(y_test, y_pred)
        return {
            'ROC': roc,
            'AUROC': auroc,
            'optimal cutoff': Jopt,
            'PRC': prc,
            'AUPRC': auprc
        }

    if bootstrap < 2:
        output = _calcScoreMetrics(y_test, y_pred)
    else:
        # apply bootstrap
        outputs = []
        for i in range(bootstrap):
            yy_test, yy_pred = resample(y_test, y_pred, **resample_kwargs)
            outputs.append(_calcScoreMetrics(yy_test, yy_pred))
        # compute mean and standard deviation of metrics
        output = {}
        for metric in ['AUROC', 'optimal cutoff', 'AUPRC']:
            v = [d[metric] for d in outputs]
            output[f'mean {metric}'] = np.mean(v)
            output[f'{metric} std'] = np.std(v)
        # compute average ROC
        mean_fpr = np.linspace(0, 1, len(y_pred))
        mean_tpr = 0.0
        for d in outputs:
            mean_tpr += np.interp(mean_fpr, d['ROC']['FPR'], d['ROC']['TPR'])
        mean_tpr /= bootstrap
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        output['mean ROC'] = {'FPR': mean_fpr, 'TPR': mean_tpr}

    return output


def calcClassMetrics(y_test, y_pred, bootstrap=0, **resample_kwargs):
    '''Compute accuracy metrics of binary labels (optionally bootstrapped)
    '''
    def _calcClassMetrics(y_test, y_pred):
        mcc = matthews_corrcoef(y_test, y_pred)
        pre, rec, f1s, sup = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1])
        avg_pre, avg_rec, avg_f1s, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        return {
            'MCC': mcc,
            'precision (0)': pre[0],
            'recall (0)': rec[0],
            'F1 score (0)': f1s[0],
            'precision (1)': pre[1],
            'recall (1)': rec[1],
            'F1 score (1)': f1s[1],
            'precision': avg_pre,
            'recall': avg_rec,
            'F1 score': avg_f1s
        }

    if bootstrap < 2:
        output = _calcClassMetrics(y_test, y_pred)
    else:
        # apply bootstrap
        outputs = []
        for i in range(bootstrap):
            yy_test, yy_pred = resample(y_test, y_pred, **resample_kwargs)
            outputs.append(_calcClassMetrics(yy_test, yy_pred))
        # compute mean and standard deviation of metrics
        output = {}
        for metric in outputs[0].keys():
            v = [d[metric] for d in outputs]
            output[f'mean {metric}'] = np.mean(v)
            output[f'{metric} std'] = np.std(v)

    return output


def calcPathogenicityProbs(CV_info, num_bins=15,
                           ppred_reliability_cutoff=200,
                           pred_distrib_fig='predictions_distribution.png',
                           path_prob_fig='pathogenicity_prob.png', **kwargs):
    '''Compute pathogenicity probabilities,
    from predictions on CV test sets
    '''

    mean_Jopt = np.mean(CV_info['optimal cutoff'])
    preds = [np.array(CV_info['predictions_0']),
             np.array(CV_info['predictions_1'])]

    # compute (normalized) histograms
    dx = 1./num_bins
    bins = np.arange(0, 1+dx, dx)
    histo = np.empty((2, len(bins)-1))
    norm_histo = np.empty_like(histo)
    for i in [0, 1]:
        h, _ = np.histogram(preds[i], bins, range=(0, 1))
        histo[i] = h
        norm_histo[i] = h/len(preds[i])

    # print predictions distribution figure
    if pred_distrib_fig is not None:
        print_pred_distrib_figure(pred_distrib_fig, bins, norm_histo,
                                  dx, mean_Jopt)

    # compute pathogenicity probability
    s = np.sum(norm_histo, axis=0)
    path_prob = np.divide(norm_histo[1], s, out=np.zeros_like(s),
                          where=(s != 0))

    # smooth path. probability profile and extend it to [0, 1] interval
    _smooth = _running_average(path_prob)
    # _smooth = _calcSmoothCurve(path_prob, 5)
    y_ext = np.concatenate([[0], _smooth, [1]])
    x_ext = np.concatenate([[0], bins[:-1]+dx/2, [1]])
    smooth_path_prob = np.array([x_ext, y_ext])

    # print pathogenicity probability figure
    if path_prob_fig is not None:
        print_path_prob_figure(
            path_prob_fig, bins, histo, dx, path_prob,
            smooth_plot=smooth_path_prob, cutoff=ppred_reliability_cutoff)

    return np.array(smooth_path_prob)


def _running_average(curve):
    ext_pprob = np.concatenate([[0], curve, [1]])
    return np.convolve(ext_pprob, np.ones((3,))/3, mode='valid')


def _calcSmoothCurve(curve, smooth_window):
    # smooth pathogenicity probability profile
    n = len(curve)
    smooth_curve = np.zeros_like(curve)
    for i in range(n):
        p = curve[i]
        sw = 0
        for k in range(1, smooth_window + 1):
            if (i-k < 0) or (i+k >= n):
                break
            else:
                sw = k
                p += curve[i-k] + curve[i+k]
        smooth_curve[i] = p / (1 + sw*2)
    return smooth_curve


def _performCV(X, y, sel_SAVs, n_estimators=1000, max_features='auto',
               n_splits=10, ROC_fig='ROC.png', feature_names=None,
               CVseed=666, stratification=None, **kwargs):

    assert stratification in [None, 'protein', 'residue']

    # set classifier
    classifier = RandomForestClassifier(
        n_estimators=n_estimators, max_features=max_features,
        oob_score=True, n_jobs=-1, class_weight='balanced')

    # define folds
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CVseed)
    CV_folds = []
    for train, test in cv.split(X, y):
        CV_folds.append([train, test])

    # protein-stratification: a same protein should not be found in
    # both training and test sets
    if stratification is not None:
        # for each fold, count occurrences of each protein/residue
        occurrences = {}
        if stratification == 'protein':
            # e.g. 'P01112'
            accs = np.array([s.split()[0] for s in sel_SAVs['SAV_coords']])
        else:
            # e.g. P01112 99
            accs = np.array([' '.join(s.split()[:2])
                             for s in sel_SAVs['SAV_coords']])
        for k, (train, test) in enumerate(CV_folds):
            counts = Counter(accs[test])
            for acc, count in counts.items():
                occurrences.setdefault(acc, np.zeros(n_splits, dtype=int))
                occurrences[acc][k] = count
        # for each acc. number, find fold with largest occurrences
        best_fold = {a: np.argmax(c) for a, c in occurrences.items()}
        new_folds = np.array([best_fold[a] for a in accs])
        # update folds
        for k in range(n_splits):
            CV_folds[k][0] = np.where(new_folds != k)[0]
            CV_folds[k][1] = np.where(new_folds == k)[0]

    # cross-validation loop
    CV_info = {k: [] for k in [
        'AUROC', 'AUPRC', 'OOB score', 'optimal cutoff', 'MCC',
        'precision (0)', 'recall (0)', 'F1 score (0)',
        'precision (1)', 'recall (1)', 'F1 score (1)',
        'precision', 'recall', 'F1 score',
        'feat. importances', 'predictions_0', 'predictions_1']}
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 20)
    i = 0
    for train, test in CV_folds:
        # create training and test datasets
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        # train Random Forest classifier
        classifier.fit(X_train, y_train)
        # calculate probabilities over decision trees
        y_pred = classifier.predict_proba(X_test)[:, 1]

        # compute ROC, AUROC, optimal cutoff (argmax of Youden's index), etc.
        sm = calcScoreMetrics(y_test, y_pred)
        for stat in ['AUROC', 'AUPRC', 'optimal cutoff']:
            CV_info[stat].append(sm[stat])
        # compute Matthews corr. coeff., precision/recall, etc. on classes
        y_pred_binary = np.where(y_pred > sm['optimal cutoff'], 1, 0)
        cm = calcClassMetrics(y_test, y_pred_binary)
        for stat in cm.keys():
            CV_info[stat].append(cm[stat])
        # other info
        mean_tpr += np.interp(mean_fpr, sm['ROC']['FPR'], sm['ROC']['TPR'])
        CV_info['OOB score'].append(classifier.oob_score_)
        CV_info['feat. importances'].append(
            np.array(classifier.feature_importances_))
        CV_info['predictions_0'].extend(y_pred[y_test == 0])
        CV_info['predictions_1'].extend(y_pred[y_test == 1])
        # print log
        i += 1
        LOGGER.info('CV iteration #{:2d}:   '.format(i) +
                    'AUROC = {:.3f}   '.format(sm['AUROC']) +
                    'AUPRC = {:.3f}   '.format(sm['AUPRC']) +
                    'OOB score = {:.3f}'.format(classifier.oob_score_))

    # compute average ROC curves
    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    # compute average ROC, optimal cutoff and other stats
    stats = {}
    for s in CV_info.keys():
        if s in ['predictions_0', 'predictions_1']:
            continue
        stats[s] = (np.mean(CV_info[s], axis=0), np.std(CV_info[s], axis=0))

    LOGGER.info('-'*60)
    LOGGER.info('Cross-validation summary:')
    LOGGER.info(f'training dataset size:   {len(y):<d}')
    LOGGER.info(f'fraction of positives:   {sum(y)/len(y):.3f}')
    for s in ['AUROC', 'AUPRC', 'OOB score', 'optimal cutoff']:
        if s == 'optimal cutoff':
            fields = ('optimal cutoff*:', stats[s][0], stats[s][1])
        else:
            fields = (f'mean {s}:', stats[s][0], stats[s][1])
        LOGGER.info('{:24} {:.3f} +/- {:.3f}'.format(*fields))
    LOGGER.info("(* argmax of Youden's index)")

    n_feats = len(stats['feat. importances'][0])
    if feature_names is None:
        feature_names = [f'feature {i}' for i in range(n_feats)]
    LOGGER.info('feature importances:')
    for i, feat_name in enumerate(feature_names):
        LOGGER.info('{:>23s}: {:.3f}'.format(
            feat_name, stats['feat. importances'][0][i]))
    LOGGER.info('-'*60)

    path_prob = calcPathogenicityProbs(CV_info, **kwargs)
    CV_summary = {
        'dataset size': len(y),
        'dataset bias': sum(y)/len(y),
        'mean ROC': list(zip(mean_fpr, mean_tpr)),
        'optimal cutoff': stats['optimal cutoff'],
        'feat. importances': stats['feat. importances'],
        'path. probability': path_prob,
        'training dataset': sel_SAVs,
        'folds': CV_folds
    }
    for s in ['AUROC', 'AUPRC', 'OOB score', 'MCC',
              'precision (0)', 'recall (0)', 'F1 score (0)',
              'precision (1)', 'recall (1)', 'F1 score (1)',
              'precision', 'recall', 'F1 score']:
        CV_summary['mean ' + s] = stats[s]

    # plot average ROC
    if ROC_fig is not None:
        print_ROC_figure(ROC_fig, mean_fpr, mean_tpr, stats['AUROC'])

    return CV_summary


def _importFeatMatrix(fm):
    assert fm.dtype.names is not None, \
        "feat. matrix must be a NumPy structured array."
    assert 'true_label' in fm.dtype.names, \
        "feat. matrix must have a 'true_label' field."
    assert 'SAV_coords' in fm.dtype.names, \
        "feat. matrix must have a 'SAV_coords' field."
    assert set(fm['true_label']) == {0, 1}, \
        'Invalid true labels in feat. matrix.'

    # check for ambiguous cases in training dataset
    del_SAVs = set(fm[fm['true_label'] == 1]['SAV_coords'])
    neu_SAVs = set(fm[fm['true_label'] == 0]['SAV_coords'])
    amb_SAVs = del_SAVs.intersection(neu_SAVs)
    if amb_SAVs:
        raise RuntimeError('Ambiguous cases found in training dataset: {}'
                           .format(amb_SAVs))

    # eliminate rows containing NaN values from feature matrix
    featset = [f for f in fm.dtype.names
               if f not in ['true_label', 'SAV_coords']]
    sel = [~np.isnan(np.sum([x for x in r])) for r in fm[featset]]
    fms = fm[sel]
    n_i = len(fm)
    n_f = len(fms)
    dn = n_i - n_f
    if dn:
        LOGGER.info(f'{dn} out of {n_i} cases ignored with missing features.')

    # split into feature array and true label array
    X = np.array([[np.float32(x) for x in v] for v in fms[featset]])
    y = fms['true_label']
    sel_SAVs = fms[['SAV_coords', 'true_label']]

    return X, y, sel_SAVs, featset


def RandomForestCV(feat_matrix, n_estimators=1500, max_features=2, **kwargs):

    X, y, sel_SAVs, featset = _importFeatMatrix(feat_matrix)
    CV_summary = _performCV(
        X, y, sel_SAVs, n_estimators=n_estimators,
        max_features=max_features, feature_names=featset, **kwargs)
    return CV_summary


def trainRFclassifier(feat_matrix, n_estimators=1500, max_features=2,
                      pickle_name='trained_classifier.pkl',
                      feat_imp_fig='feat_importances.png', **kwargs):

    X, y, sel_SAVs, featset = _importFeatMatrix(feat_matrix)

    # calculate optimal Youden cutoff through CV
    CV_summary = _performCV(
        X, y, sel_SAVs, n_estimators=n_estimators,
        max_features=max_features, feature_names=featset, **kwargs)

    # train a classifier on the whole dataset
    clsf = RandomForestClassifier(
        n_estimators=n_estimators, max_features=max_features,
        oob_score=True, class_weight='balanced', n_jobs=-1)
    clsf.fit(X, y)

    fimp = clsf.feature_importances_
    LOGGER.info('-'*60)
    LOGGER.info('Classifier training summary:')
    LOGGER.info(f'mean OOB score:          {clsf.oob_score_:.3f}')
    LOGGER.info('feature importances:')
    for feat_name, importance in zip(featset, fimp):
        LOGGER.info(f'{feat_name:>23s}: {importance:.3f}')
    LOGGER.info('-'*60)

    if feat_imp_fig is not None:
        print_feat_imp_figure(feat_imp_fig, fimp, featset)

    clsf_dict = {
        'trained RF': clsf,
        'features': featset,
        'CV summary': CV_summary,
    }

    # save pickle with trained classifier and other info
    if pickle_name is not None:
        pickle.dump(clsf_dict, open(pickle_name, 'wb'))

    return clsf_dict


def extendDefaultTrainingDataset(names, arrays, base_default_featset='full'):
    """base : array
    Input array to extend.

    names : string, sequence
    String or sequence of strings corresponding to the names of the new fields.

    data : array or sequence of arrays
    Array or sequence of arrays storing the fields to add to the base.
    """

    training_dataset = getDefaultTrainingDataset()

    # select features from integrated dataset
    if base_default_featset is None:
        base_featset = []
    if base_default_featset in DEFAULT_FEATSETS:
        base_featset = DEFAULT_FEATSETS[base_default_featset]
    else:
        base_featset = list(base_default_featset)
    featset = ['SAV_coords', 'true_label'] + base_featset
    base_dataset = training_dataset[featset]

    # extend base training dataset
    fm = rfn.rec_append_fields(base_dataset, names, arrays, dtypes='float32')

    return fm
