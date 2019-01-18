from prody import LOGGER

__all__ = ['print_pred_distrib_figure', 'print_path_prob_figure',
           'print_ROC_figure', 'print_feat_imp_figure']


def try_plt_import():
    try:
        import matplotlib as plt
    except ImportError:
        LOGGER.warn('Please install matplotlib in order to generate figures')
        return False
    return True


def print_pred_distrib_figure(filename, bins, histo, dx, J_opt):
    assert isinstance(filename, str), 'filename must be a string'

    if not try_plt_import():
        return

    figure = plt.figure(figsize=(5, 4))
    plt.bar(bins[:-1], histo[0], width=dx, align='edge',
            color='blue', alpha=0.7, label='neutral'    )
    plt.bar(bins[:-1], histo[1], width=dx, align='edge',
            color='red',  alpha=0.7, label='deleterious')
    plt.axvline(x=J_opt, color='k', ls='--', lw=1)
    plt.ylabel('distribution')
    plt.xlabel('predicted score')
    plt.legend()
    figure.savefig(filename, bbox_inches='tight')
    plt.close()
    LOGGER.info(f'Predictions distribution saved to {filename}')


def print_path_prob_figure(filename, bins, histo, dx, path_prob,
                     smooth_path_prob, cutoff=200):
    assert isinstance(filename, str), 'filename must be a string'

    if not try_plt_import():
        return

    figure = plt.figure(figsize=(5, 4))
    s = np.sum(histo, axis=0)
    v1 = np.where(s>=cutoff, path_prob, 0)
    v2 = np.where(s< cutoff, path_prob, 0)
    v3 = np.where(s>=cutoff, smooth_path_prob, 0.)
    plt.bar(bins[:-1], v1, width=dx, align='edge', color='red', alpha=1  )
    plt.bar(bins[:-1], v2, width=dx, align='edge', color='red', alpha=0.7)
    plt.plot(bins[:-1]+dx/2, v3, color='orange')
    plt.ylabel('pathogenicity prob.')
    plt.xlabel('predicted score')
    plt.ylim((0, 1))
    figure.savefig(filename, bbox_inches='tight')
    plt.close()
    LOGGER.info(f'Pathogenicity plot saved to {filename}')


def print_ROC_figure(filename, fpr, tpr):
    assert isinstance(filename, str), 'filename must be a string'

    if not try_plt_import():
        return

    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
    plt.plot(fpr, tpr,       linestyle='-',  lw=2, color='r',
             label = f'Mean ROC (AUC = {mean_auc:.3f})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(filename, format='pdf')
    plt.close()
    LOGGER.info(f'ROC plot saved to {print_ROC}')


def print_feat_imp_figure(filename, feat_imp, featset):
    assert isinstance(filename, str), 'filename must be a string'

    if not try_plt_import():
        return

    fig = plt.figure(figsize=(5, 5))
    n = len(feat_imp)
    plt.bar(range(n), feat_imp, align='center', tick_label=featset)
    plt.xticks(rotation='vertical')
    plt.ylabel('feat. importance')
    fig.savefig(filename, format='pdf', bbox_inches = 'tight')
    plt.close()
    LOGGER.info(f'Feat. importance plot saved to {filename}')



