import numpy as np
import pickle
from prody import LOGGER
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


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
                           distrib_figure='predictions_distribution.pdf',
                           path_prob_figure='pathogenicity_prob.pdf',
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
        norm_histo[i] =  h/len(preds[i])
    if distrib_figure is not None:
        figure = plt.figure(figsize=(5, 4))
        plt.bar(bins[:-1], norm_histo[0], width=dx, align='edge', 
                color='blue', alpha=0.7, label='neutral'    )
        plt.bar(bins[:-1], norm_histo[1], width=dx, align='edge', 
                color='red',  alpha=0.7, label='deleterious')
        plt.axvline(x=avg_J_opt, color='k', ls='--', lw=1)
        plt.ylabel('distribution')
        plt.xlabel('predicted score')
        plt.legend()
        figure.savefig(distrib_figure, bbox_inches='tight')
        plt.close()
        LOGGER.info('Predictions distribution saved to {}'
                    .format(distrib_figure))

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
    if path_prob_figure is not None:
        figure = plt.figure(figsize=(5, 4))
        s = np.sum(histo, axis=0)
        c = ppred_reliability_cutoff
        v1 = np.where(s>=c, path_prob, 0)
        v2 = np.where(s< c, path_prob, 0)
        v3 = np.where(s>=c, smooth_path_prob, 0.)
        plt.bar(bins[:-1], v1, width=dx, align='edge', color='red', alpha=1  )
        plt.bar(bins[:-1], v2, width=dx, align='edge', color='red', alpha=0.7)
        plt.plot(bins[:-1]+dx/2, v3, color='orange')
        plt.ylabel('pathogenicity prob.')
        plt.xlabel('predicted score')
        plt.ylim((0, 1))
        figure.savefig(path_prob_figure, bbox_inches='tight')
        plt.close()
        LOGGER.info('Pathogenicity plot saved to {}'.format(path_prob_figure))
    
    return np.array((bins[:-1], path_prob, smooth_path_prob))


def RandomForestCV(X, y, n_estimators=1000, max_features='auto', n_splits=10, 
                   print_ROC='ROC.pdf', feature_names=None, **kwargs):

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
        LOGGER.info('CV iteration #{:2d}:    '.format(i) + \
                    'ROC-AUC = {:.3f}   OOB score = {:.3f}'
                    .format(roc_auc, classifier.oob_score_) )

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
    LOGGER.info('mean ROC-AUC:        {:.3f}'.format(mean_auc))
    LOGGER.info('mean OOB score:      {:.3f}'.format(mean_oob))
    LOGGER.info("optimal cutoff*:    {:.3f} +/- {:.3f}"
                .format(avg_J_opt, std_J_opt))
    LOGGER.info("(* argmax of Youden's index)")
    LOGGER.info('feature importances:')
    if feature_names is None:
        feature_names = [' ']*len(avg_feat_imp)
    for feat_name, importance in zip(feature_names, avg_feat_imp):
        LOGGER.info('{:>19s}: {:.3f}'.format(feat_name, importance))
    LOGGER.info('-'*60)
    path_prob = calcPathogenicityProbs(CV_info, **kwargs)
    CV_summary = {'mean ROC-AUC'     : mean_auc,
                  'mean OOB score'   : mean_oob,
                  'mean ROC'         : list(zip(mean_fpr, mean_tpr)),
                  'optimal cutoff'   : (avg_J_opt, std_J_opt),
                  'feat. importance' : avg_feat_imp,
                  'path. probability': path_prob}

    # plot average ROC
    if print_ROC is not None:
        fig = plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1],     linestyle='--', lw=1, color='k')
        plt.plot(mean_fpr, mean_tpr, linestyle='-',  lw=2, color='r',
                 label='Mean ROC (AUC = {:.3f})'.format(mean_auc))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        fig.savefig(print_ROC, format='pdf')
        plt.close()
        LOGGER.info('ROC plot saved to {}'.format(print_ROC))
        
    return CV_summary


def trainRFclassifier(feat_matrix, n_estimators=1500, max_features=2,
                      pickle_name='trained_classifier.pkl', 
                      print_feat_import='feat_importances.pdf', **kwargs):

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
           max_features=max_features, oob_score=True, class_weight='balanced', n_jobs=-1)
    clsf.fit(X, y)
    LOGGER.info('-'*60)
    LOGGER.info('Classifier training summary:')
    LOGGER.info('mean OOB score:      {:.3f}'.format(clsf.oob_score_))
    LOGGER.info('feature importances:')
    for feat_name, importance in zip(featset, clsf.feature_importances_):
        LOGGER.info('{:>19s}: {:.3f}'.format(feat_name, importance))
    LOGGER.info('-'*60)

    if print_feat_import is not None:
        # print feature importance figure 
        f = clsf.feature_importances_
        fig = plt.figure(figsize=(5, 5))
        plt.bar(range(len(f)), f, align='center', tick_label=featset)
        plt.xticks(rotation='vertical')
        plt.ylabel('feat. importance')
        fig.savefig(print_feat_import, format='pdf', bbox_inches = 'tight')
        plt.close()
        LOGGER.info(f'Feat. importance plot saved to {print_feat_import}')

    clsf_dict = {'trained RF': clsf,
                 'features'  : featset,
                 'CV summary': CV_summary,
                 'training dataset': fm}

    # save pickle with trained classifier and other info
    if pickle_name is not None:
        pickle.dump(clsf_dict, open(pickle_name, 'wb'))

    return clsf_dict


