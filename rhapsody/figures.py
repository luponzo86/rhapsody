import numpy as np
import warnings
from prody import LOGGER
from .rhapsody import Rhapsody

__all__ = ['print_pred_distrib_figure', 'print_path_prob_figure',
           'print_ROC_figure', 'print_feat_imp_figure',
           'print_sat_mutagen_figure']


def try_import_matplotlib():
    try:
        import matplotlib as plt
        plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})
    except ImportError:
        LOGGER.warn('matplotlib is required for generating figures')
        return None
    return plt


def print_pred_distrib_figure(filename, bins, histo, dx, J_opt):
    assert isinstance(filename, str), 'filename must be a string'

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return
    plt = matplotlib.pyplot

    figure = plt.figure(figsize=(5, 4))
    plt.bar(bins[:-1], histo[0], width=dx, align='edge',
            color='blue', alpha=0.7, label='neutral'    )
    plt.bar(bins[:-1], histo[1], width=dx, align='edge',
            color='red',  alpha=0.7, label='deleterious')
    plt.axvline(x=J_opt, color='k', ls='--', lw=1)
    plt.ylabel('distribution')
    plt.xlabel('predicted score')
    plt.legend()
    figure.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    LOGGER.info(f'Predictions distribution saved to {filename}')


def print_path_prob_figure(filename, bins, histo, dx, path_prob,
                     smooth_path_prob, cutoff=200):
    assert isinstance(filename, str), 'filename must be a string'

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return
    plt = matplotlib.pyplot

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
    figure.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    LOGGER.info(f'Pathogenicity plot saved to {filename}')


def print_ROC_figure(filename, fpr, tpr):
    assert isinstance(filename, str), 'filename must be a string'

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return
    plt = matplotlib.pyplot

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
    fig.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    LOGGER.info(f'ROC plot saved to {print_ROC}')


def print_feat_imp_figure(filename, feat_imp, featset):
    assert isinstance(filename, str), 'filename must be a string'

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return
    plt = matplotlib.pyplot

    fig = plt.figure(figsize=(5, 5))
    n = len(feat_imp)
    plt.bar(range(n), feat_imp, align='center', tick_label=featset)
    plt.xticks(rotation='vertical')
    plt.ylabel('feat. importance')
    fig.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
    LOGGER.info(f'Feat. importance plot saved to {filename}')


def adjust_res_interval(res_interval, min_size=10):
    res_i = max(1, res_interval[0])
    res_f = max(1, res_interval[1])
    n = min_size - 1
    while (res_f - res_i) < n:
        if res_i > 1:
            res_i -= 1
        if (res_f - res_i) >= n:
            break
        res_f += 1
    return (res_i, res_f)


def print_sat_mutagen_figure(filename, rhapsody_obj, other_preds=None,
                             res_interval=None, min_interval_size=15):
    assert isinstance(filename, str), 'filename must be a string'
    assert isinstance(rhapsody_obj, Rhapsody), 'not a Rhapsody object'
    assert rhapsody_obj.predictions is not None, 'predictions not found'
    if res_interval is not None:
        assert isinstance(res_interval, tuple) and len(res_interval)==2, \
               'res_interval must be a tuple of 2 values'
        assert res_interval[1] >= res_interval[0], 'invalid res_interval'
    if other_preds is not None:
        assert len(other_preds) == len(rhapsody_obj.predictions), \
               'length of additional predictions array is incorrect'

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return

    # make sure that all variants belong to the same Uniprot sequence
    s = rhapsody_obj.SAVcoords['acc']
    if len(set(s)) == 1:
        acc = s[0]
    else:
        m = 'Only variants from a single Uniprot sequence can be accepted'
        raise ValueError(m)

    if rhapsody_obj.auxPreds is not None:
        aux_preds_found = True
    else:
        aux_preds_found = False
    if other_preds is not None:
        other_preds_found = True
    else:
        other_preds_found = False

    # import pathogenicity probability from Rhapsody object
    p_full = rhapsody_obj.predictions['path. probability']
    p_aux  = None
    p_mix  = None
    if aux_preds_found:
        p_aux = rhapsody_obj.auxPreds[  'path. probability']
        p_mix = rhapsody_obj.mixedPreds['path. probability']

    # select an appropriate interval, based on available predictions
    res_min = np.min(rhapsody_obj.SAVcoords['pos'])
    res_max = np.max(rhapsody_obj.SAVcoords['pos'])
    upper_lim = res_max + min_interval_size

    # create empty (20 x num_res) mutagenesis tables
    table_full = np.zeros((20, upper_lim), dtype=float)
    table_full[:] = 'nan'
    table_mix = table_full.copy()
    if other_preds_found:
        table_other = table_full.copy()

    # fill tables with predicted probability
    #  1:    deleterious
    #  0:    neutral
    # 'nan': no prediction/wt
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_map  = {aa: i for i, aa in enumerate(aa_list)}
    for i, SAV in enumerate(rhapsody_obj.SAVcoords):
        aa_mut = SAV['aa_mut']
        index  = SAV['pos']-1
        table_full[aa_map[aa_mut], index] = p_full[i]
        if aux_preds_found:
            table_mix[aa_map[aa_mut], index] = p_mix[i]
        if other_preds_found:
            table_other[aa_map[aa_mut], index] = other_preds[i]

    # compute average pathogenicity profiles
    # NB: I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_p_full = np.nanmean(table_full, axis=0)
        avg_p_mix  = np.nanmean(table_mix,  axis=0)
        min_p = np.nanmin(table_mix, axis=0)
        max_p = np.nanmax(table_mix, axis=0)
        if other_preds_found:
            avg_p_other = np.nanmean(table_other, axis=0)

    # use upper strip for showing additional info, such as PDB lengths
    upper_strip = np.zeros((1, upper_lim))
    for a, b in zip(rhapsody_obj.SAVcoords, rhapsody_obj.Uniprot2PDBmap):
        index = a['pos'] - 1
        PDB_length = int(b[2][4]) if isinstance(b[2], tuple) else np.nan
        upper_strip[0, index] = PDB_length
    upper_strip[0, :] /= np.nanmax(upper_strip[0, :])

    # PLOT FIGURE

    from matplotlib import pyplot as plt
    from matplotlib import gridspec as gridspec

    # portion of the sequence to display
    if res_interval is None:
        res_interval = (res_min, res_max)
    # adjust interval
    res_i, res_f = adjust_res_interval(res_interval, min_interval_size)
    nres_shown = res_f - res_i + 1

    # figure proportions
    fig_height = 8 # inches
    fig_width  = fig_height/2
    fig_width *= nres_shown/20
    fig, ax = plt.subplots(3, 2, figsize=(fig_width, fig_height))
    wspace = 0.5 # inches
    plt.subplots_adjust(wspace=wspace/fig_width, hspace=0.15)

    # figure structure
    gs = gridspec.GridSpec(3, 2, width_ratios=[nres_shown, 1],
                           height_ratios=[1, 20, 10])
    ax0  = plt.subplot(gs[0, 0]) # secondary structure strip
    ax1  = plt.subplot(gs[1, 0]) # mutagenesis table
    axcb = plt.subplot(gs[1, 1]) # colorbar
    ax2  = plt.subplot(gs[2, 0]) # average profile

    # secondary structure strip
    ax0.imshow(upper_strip[:, res_i-1:res_f], aspect='auto',
               cmap='YlGn', vmin=0, vmax=1)
    ax0.set_ylim((-0.45, .45))
    ax0.set_yticks([])
    ax0.set_xticks([])

    # mutagenesis table (heatmap)
    matplotlib.cm.coolwarm.set_bad(color='white')
    if aux_preds_found:
        table = table_mix
    else:
        table = table_full
    im = ax1.imshow(table[:, res_i-1:res_f], aspect='auto',
                    cmap='coolwarm', vmin=0, vmax=1)
    axcb.figure.colorbar(im, cax=axcb)
    ax1.set_yticks(np.arange(len(aa_list)))
    pad = 0.2/fig_width
    ax1.set_yticklabels(aa_list, ha='center', position=(-pad,0), fontsize=14)
    ax1.set_xticks(np.arange(5-res_i%5, res_f-res_i+1, 5))
    ax1.set_xticklabels([])
    ax1.set_ylabel('pathog. probability', labelpad=10)

    # average pathogenicity profile
    x_resids = np.arange(1, upper_lim+1)
    # cutoff line
    ax2.hlines(0.5, -.5, upper_lim+.5, colors='grey', lw=.8,
               linestyle='dashed')
    # solid line for predictions obtained with full classifier
    ax2.plot(x_resids, avg_p_full, color='red')
    # dotted line for predictions obtained with auxiliary classifier
    _p = np.where(np.isnan(avg_p_full), avg_p_mix, avg_p_full)
    ax2.plot(x_resids, _p, color='red', ls='dotted')
    # shading showing range of values
    ax2.fill_between(x_resids, min_p, max_p, alpha=0.5, edgecolor='white',
                     facecolor='salmon')
    # plot average profile for other predictions, if available
    if other_preds_found:
        ax2.plot(x_resids, avg_p_other, color='blue', lw=.5)
    ax2.set_xlim((res_i-.5, res_f+.5))
    ax2.set_xlabel('residue number')
    ax2.set_ylim((0, 1))
    ax2.set_ylabel('average', rotation=90, labelpad=10)
    ax2.set_yticklabels([])
    ax2r = ax2.twinx()
    ax2r.set_yticks([0, .5, 1])
    ax2r.set_yticklabels(['0', '0.5', '1'])
    ax2r.tick_params(axis='both', which='major', pad=15)

    fig.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    LOGGER.info(f'Saturation mutagenesis figure saved to {filename}')
