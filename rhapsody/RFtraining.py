import numpy as np
import pickle
from prody import LOGGER
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from .figures import *

__all__ = ['calcROC', 'calcPathogenicityProbs',
           'RandomForestCV', 'trainRFclassifier']


def calcROC(y_test, y_pred):
    # compute ROC and AUC-ROC
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # compute optimal cutoff J (argmax of Youden's index)
    diff = np.array([y-x for x,y in zip(fpr, tpr)])
    J_opt = thr[(-diff).argsort()][0]
    return {'FPR': fpr, 'TPR': tpr, 'thresholds': thr,
            'AUROC': roc_auc, 'optimal cutoff': J_opt}


def calcPathogenicityProbs(CV_info, bin_width=0.04, smooth_window=5,
                           ppred_reliability_cutoff=200,
                           pred_distrib_fig='predictions_distribution.png',
                           path_prob_fig='pathogenicity_prob.png',
                           **kwargs):
    '''Compute pathogenicity probabilities,
    from predictions on CV test sets
    '''

    avg_J_opt = np.mean(CV_info['Youden_cutoff'])
    preds = [np.array(CV_info['predictions_0']),
             np.array(CV_info['predictions_1'])]

    # compute (normalized) histograms
    dx = bin_width
    bins = np.arange(0, 1+dx, dx)
    n_bins = len(bins)-1
    histo = np.empty((2, n_bins))
    norm_histo = np.empty_like(histo)
    for i in [0, 1]:
        h, _ = np.histogram(preds[i], bins, range=(0,1))
        histo[i] = h
        norm_histo[i] = h/len(preds[i])

    # print predictions distribution figure
    if pred_distrib_fig is not None:
        print_pred_distrib_figure(pred_distrib_fig, bins, norm_histo,
                                  dx, avg_J_opt)

    # compute pathogenicity probability
    s = np.sum(norm_histo, axis=0)
    path_prob = np.divide(norm_histo[1], s, out=np.zeros_like(s), where=s!=0)

    # smooth pathogenicity probability profile
    smooth_path_prob = np.zeros_like(path_prob)
    for i in range(n_bins) :
        p = path_prob[i]
        sw = 0
        for k in range(1, smooth_window+1) :
            if (i-k < 0) or (i+k >= n_bins) :
                break
            else :
                sw = k
                p += path_prob[i-k] + path_prob[i+k]
        smooth_path_prob[i] = p / (1 + sw*2)

    # print pathogenicity probability figure
    if path_prob_fig is not None:
        print_path_prob_figure(path_prob_fig, bins, histo, dx, path_prob,
                         smooth_path_prob, cutoff=ppred_reliability_cutoff)

    return np.array((bins[:-1], path_prob, smooth_path_prob))


def RandomForestCV(X, y, n_estimators=1000, max_features='auto', n_splits=10,
                   ROC_fig='ROC.png', feature_names=None, **kwargs):

    # set classifier
    classifier = RandomForestClassifier(n_estimators=n_estimators,
                 max_features=max_features, oob_score=True,
                 class_weight='balanced', n_jobs=-1)

    # set cross-validation procedure
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=666)

    # cross-validation loop
    CV_info = {'ROC-AUC'        : [],
               'feat_importance': [],
               'OOB_score'      : [],
               'Youden_cutoff'  : [],
               'predictions_0'  : [],
               'predictions_1'  : []}
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        # create training and test datasets
        X_train = X[train]
        X_test  = X[test]
        y_train = y[train]
        y_test  = y[test]
        # train Random Forest classifier
        classifier.fit(X_train, y_train)
        # calculate probabilities over decision trees
        y_pred = classifier.predict_proba(X_test)
        # compute ROC, AUROC and optimal cutoff (argmax of Youden's index)
        d = calcROC(y_test, y_pred[:, 1])
        roc_auc = d['AUROC']
        J_opt   = d['optimal cutoff']
        # store other info and metrics for each iteration
        mean_tpr += np.interp(mean_fpr, d['FPR'], d['TPR'])
        CV_info['ROC-AUC'].append(roc_auc)
        CV_info['feat_importance'].append(classifier.feature_importances_)
        CV_info['OOB_score'].append(classifier.oob_score_)
        CV_info['Youden_cutoff'].append(J_opt)
        CV_info['predictions_0'].extend(y_pred[np.where(y_test==0), 1][0])
        CV_info['predictions_1'].extend(y_pred[np.where(y_test==1), 1][0])
        # print log
        i += 1
        LOGGER.info(f'CV iteration #{i:2d}:    ROC-AUC = {roc_auc:.3f}' + \
                    f'   OOB score = {classifier.oob_score:.3f}')

    # compute average ROC, optimal cutoff and other stats
    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[ 0] = 0.0
    mean_tpr[-1] = 1.0
    mean_auc  = auc(mean_fpr, mean_tpr)
    mean_oob  = np.mean(CV_info['OOB_score'])
    avg_J_opt = np.mean(CV_info['Youden_cutoff'])
    std_J_opt = np.std( CV_info['Youden_cutoff'])
    avg_feat_imp = np.mean(np.array(CV_info['feat_importance']), axis=0)
    LOGGER.info('-'*60)
    LOGGER.info('Cross-validation summary:')
    LOGGER.info(f'mean ROC-AUC:        {mean_auc:.3f}')
    LOGGER.info(f'mean OOB score:      {mean_oob:.3f}')
    LOGGER.info(f'optimal cutoff*:     {avg_J_opt:.3f} +/- {std_J_opt:.3f}')
    LOGGER.info("(* argmax of Youden's index)")
    LOGGER.info('feature importances:')
    if feature_names is None:
        feature_names = [' ']*len(avg_feat_imp)
    for feat_name, importance in zip(feature_names, avg_feat_imp):
        LOGGER.info(f'{feat_name:>19s}: {importance:.3f}')
    LOGGER.info('-'*60)
    path_prob = calcPathogenicityProbs(CV_info, **kwargs)
    CV_summary = {'mean ROC-AUC'     : mean_auc,
                  'mean OOB score'   : mean_oob,
                  'mean ROC'         : list(zip(mean_fpr, mean_tpr)),
                  'optimal cutoff'   : (avg_J_opt, std_J_opt),
                  'feat. importance' : avg_feat_imp,
                  'path. probability': path_prob}

    # plot average ROC
    if ROC_fig is not None:
        print_ROC_figure(ROC_fig, mean_fpr, mean_tpr)

    return CV_summary


def trainRFclassifier(feat_matrix, n_estimators=1500, max_features=2,
                      pickle_name='trained_classifier.pkl',
                      feat_imp_fig='feat_importances.png', **kwargs):

    assert feat_matrix.dtype.names is not None, \
           "'feat_matrix' must be a NumPy structured array."
    assert 'true_label' in feat_matrix.dtype.names,  \
           "'feat_matrix' must have a 'true_label' field."
    assert 'SAV_coords' in feat_matrix.dtype.names,  \
           "'feat_matrix' must have a 'SAV_coords' field."
    assert 'Uniprot2PDB' in feat_matrix.dtype.names,  \
           "'feat_matrix' must have a 'Uniprot2PDB' field."

    # eliminate rows containing NaN values from feature matrix
    featset = [f for f in feat_matrix.dtype.names if f not in
               ('true_label', 'SAV_coords', 'Uniprot2PDB')]
    sel = [~np.isnan(np.sum([x for x in r])) for r in feat_matrix[featset]]
    fm = feat_matrix[sel]
    n_i = len(feat_matrix)
    n_f = len(fm)
    n = n_i-n_f
    if n: LOGGER.info(f'{n} out of {n_i} cases ignored with missing features.')

    # split into feature array and true label array
    X = fm[featset].copy()
    X = X.view((np.float32, len(featset)))
    y = fm['true_label']

    # calculate optimal Youden cutoff through CV
    CV_summary = RandomForestCV(X, y, n_estimators=n_estimators,
                 max_features=max_features, feature_names=featset, **kwargs)

    # train a classifier on the whole dataset
    clsf = RandomForestClassifier(n_estimators=n_estimators,
                                  max_features=max_features, oob_score=True,
                                  class_weight='balanced', n_jobs=-1)
    clsf.fit(X, y)

    fimp = clsf.feature_importances_
    LOGGER.info('-'*60)
    LOGGER.info('Classifier training summary:')
    LOGGER.info(f'mean OOB score:      {clsf.oob_score_:.3f}')
    LOGGER.info('feature importances:')
    for feat_name, importance in zip(featset, fimp):
        LOGGER.info(f'{feat_name:>19s}: {importance:.3f}')
    LOGGER.info('-'*60)

    if feat_imp_fig is not None:
        print_feat_imp_figure(feat_imp_fig, fimp, featset)

    clsf_dict = {'trained RF': clsf,
                 'features'  : featset,
                 'CV summary': CV_summary,
                 'training dataset': fm}

    # save pickle with trained classifier and other info
    if pickle_name is not None:
        pickle.dump(clsf_dict, open(pickle_name, 'wb'))

    return clsf_dict
