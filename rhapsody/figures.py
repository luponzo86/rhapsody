import numpy as np
import os
import warnings
from string import Template
from prody import LOGGER, SETTINGS
from .rhapsody import Rhapsody

__all__ = ['print_pred_distrib_figure', 'print_path_prob_figure',
           'print_ROC_figure', 'print_feat_imp_figure',
           'print_sat_mutagen_figure']


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


def _adjust_res_interval(res_interval, upper_lim, min_size=10):
    res_i = max(1, res_interval[0])
    res_f = max(1, res_interval[1])
    n = min_size - 1
    while (res_f - res_i) < n:
        if res_i > 1:
            res_i -= 1
        if (res_f - res_i) >= n:
            break
        res_f += 1
    res_f = min(upper_lim, res_f)
    return (res_i, res_f)


def print_sat_mutagen_figure(filename, rhapsody_obj, res_interval=None,
                             PolyPhen2=True, EVmutation=True, extra_plot=None,
                             fig_height=8, fig_width=None, dpi=300,
                             min_interval_size=15, html=False):

    # check inputs
    assert isinstance(filename, str), 'filename must be a string'
    assert isinstance(rhapsody_obj, Rhapsody), 'not a Rhapsody object'
    assert rhapsody_obj._isColSet('main score'), 'predictions not found'
    if res_interval is not None:
        assert isinstance(res_interval, tuple) and len(res_interval) == 2, \
               'res_interval must be a tuple of 2 values'
        assert res_interval[1] >= res_interval[0], 'invalid res_interval'
    if extra_plot is not None:
        assert len(extra_plot) == rhapsody_obj.numSAVs, \
               'length of additional predictions array is incorrect'
    assert isinstance(fig_height, (int, float))
    assert isinstance(dpi, int)

    matplotlib = _try_import_matplotlib()
    if matplotlib is None:
        return

    # delete extension from filename
    filename = os.path.splitext(filename)[0]

    # make sure that all variants belong to the same Uniprot sequence
    accs = [s.split()[0] for s in rhapsody_obj.data['SAV coords']]
    if len(set(accs)) != 1:
        m = 'Only variants from a single Uniprot sequence can be accepted'
        raise ValueError(m)

    # select an appropriate interval, based on available predictions
    seq_pos = [int(s.split()[1]) for s in rhapsody_obj.data['SAV coords']]
    res_min = np.min(seq_pos)
    res_max = np.max(seq_pos)
    upper_lim = res_max + min_interval_size

    # create empty (20 x num_res) mutagenesis tables
    table_best = np.zeros((20, upper_lim), dtype=float)
    table_best[:] = 'nan'
    table_main = table_best.copy()
    if extra_plot is not None:
        table_other = table_best.copy()
    if PolyPhen2:
        table_PP2 = table_best.copy()
    if EVmutation:
        table_EVmut = table_best.copy()

    # import pathogenicity probabilities from Rhapsody object
    p_best = rhapsody_obj.getPredictions(classifier='best')['path. prob.']
    p_main = rhapsody_obj.data['main path. prob.']
    if PolyPhen2:
        rhapsody_obj._calcPolyPhen2Predictions()
        p_PP2 = rhapsody_obj.data['PolyPhen-2 score']
    if EVmutation:
        rhapsody_obj._calcEVmutationPredictions()
        EVmut_score = np.array(rhapsody_obj.data['EVmutation score'])
        EVmut_cutoff = SETTINGS.get('EVmutation_metrics')['optimal cutoff']
        p_EVmut = -EVmut_score/EVmut_cutoff*0.5

    # fill tables with predicted probability
    #  1:    deleterious
    #  0:    neutral
    # 'nan': no prediction/wt
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_map = {aa: i for i, aa in enumerate(aa_list)}
    for i, SAV in enumerate(rhapsody_obj.data['SAV coords']):
        aa_mut = SAV.split()[3]
        index = int(SAV.split()[1]) - 1
        table_best[aa_map[aa_mut], index] = p_best[i]
        table_main[aa_map[aa_mut], index] = p_main[i]
        if extra_plot is not None:
            table_other[aa_map[aa_mut], index] = extra_plot[i]
        if PolyPhen2:
            table_PP2[aa_map[aa_mut], index] = p_PP2[i]
        if EVmutation:
            table_EVmut[aa_map[aa_mut], index] = p_EVmut[i]

    # compute average pathogenicity profiles
    # NB: I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_p_best = np.nanmean(table_best, axis=0)
        avg_p_main = np.nanmean(table_main, axis=0)
        min_p = np.nanmin(table_best, axis=0)
        max_p = np.nanmax(table_best, axis=0)
        if extra_plot is not None:
            avg_p_other = np.nanmean(table_other, axis=0)
        if PolyPhen2:
            avg_p_PP2 = np.nanmean(table_PP2, axis=0)
        if EVmutation:
            avg_p_EVmut = np.nanmean(table_EVmut, axis=0)

    # use upper strip for showing additional info, such as PDB lengths
    upper_strip = np.zeros((1, upper_lim))
    upper_strip[:] = 'nan'
    PDB_sizes = np.zeros(upper_lim, dtype=int)
    PDB_coords = ['']*upper_lim
    for s in rhapsody_obj.data:
        index = int(s['SAV coords'].split()[1]) - 1
        if s['PDB size'] != 0:
            PDB_length = int(s['PDB size'])
            PDBID_chain = ':'.join(s['PDB SAV coords'][0].split()[:2])
            upper_strip[0, index] = PDB_length
            PDB_sizes[index] = PDB_length
            PDB_coords[index] = PDBID_chain
    max_PDB_size = max(PDB_sizes)
    if max_PDB_size != 0:
        upper_strip[0, :] /= max_PDB_size

    # PLOT FIGURE

    from matplotlib import pyplot as plt
    from matplotlib import gridspec as gridspec

    # portion of the sequence to display
    if res_interval is None:
        res_interval = (res_min, res_max)
    # adjust interval
    res_i, res_f = _adjust_res_interval(res_interval, upper_lim,
                                        min_interval_size)
    nres_shown = res_f - res_i + 1

    # figure proportions
    if fig_width is None:
        fig_width = fig_height/2  # inches
        fig_width *= nres_shown/20
    fig, ax = plt.subplots(3, 2, figsize=(fig_width, fig_height))
    wspace = 0.5  # inches
    plt.subplots_adjust(wspace=wspace/fig_width, hspace=0.15)

    # figure structure
    gs = gridspec.GridSpec(3, 2, width_ratios=[nres_shown, 1],
                           height_ratios=[1, 20, 10])
    ax0 = plt.subplot(gs[0, 0])  # secondary structure strip
    ax1 = plt.subplot(gs[1, 0])  # mutagenesis table
    axcb = plt.subplot(gs[1, 1])  # colorbar
    ax2 = plt.subplot(gs[2, 0])  # average profile

    # padding for tick labels
    pad = 0.2/fig_width

    # top strip
    matplotlib.cm.YlGn.set_bad(color='antiquewhite')
    ax0.imshow(upper_strip[0:1, res_i-1:res_f], aspect='auto',
               cmap='YlGn', vmin=0, vmax=1)
    ax0.set_ylim((-0.45, .45))
    ax0.set_yticks([])
    ax0.set_ylabel(f'PDB size \n[0-{max_PDB_size} res] ', fontsize=14,
                   ha='right', va='center', rotation=0)
    ax0.set_xticks(np.arange(5-res_i % 5, res_f-res_i+1, 5))
    ax0.set_xticklabels([])
    # add white grid
    ax0.set_xticks(np.arange(-.5, res_f-res_i+1, 1), minor=True)
    ax0.tick_params(axis='both', which='minor', length=0)
    ax0.grid(which='minor', color='w', linestyle='-', linewidth=.5)

    # mutagenesis table (heatmap)
    matplotlib.cm.coolwarm.set_bad(color='antiquewhite')
    im = ax1.imshow(table_best[:, res_i-1:res_f], aspect='auto',
                    cmap='coolwarm', vmin=0, vmax=1)
    axcb.figure.colorbar(im, cax=axcb)
    ax1.set_yticks(np.arange(len(aa_list)))
    ax1.set_yticklabels(aa_list, ha='center', position=(-pad, 0), fontsize=14)
    ax1.set_xticks(np.arange(5-res_i % 5, res_f-res_i+1, 5))
    ax1.set_xticklabels([])
    ax1.set_ylabel('pathog. probability', labelpad=10)
    # add white grid
    ax1.set_xticks(np.arange(-.5, res_f-res_i+1, 1), minor=True)
    ax1.set_yticks(np.arange(-.5, 20, 1), minor=True)
    ax1.tick_params(axis='both', which='minor', length=0)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=.5)

    # average pathogenicity profile
    x_resids = np.arange(1, upper_lim+1)
    # shading showing range of values
    ax2.fill_between(x_resids, min_p, max_p, alpha=0.5,
                     edgecolor='salmon', facecolor='salmon')
    # plot average profile for other predictions, if available
    if extra_plot is not None:
        ax2.plot(x_resids, avg_p_other, color='gray', lw=1)
    if PolyPhen2:
        ax2.plot(x_resids, avg_p_PP2, color='blue', lw=1)
    if EVmutation:
        ax2.plot(x_resids, avg_p_EVmut, color='green', lw=1)
    # solid line for predictions obtained with full classifier
    ax2.plot(x_resids, avg_p_main, 'ro-')
    # dotted line for predictions obtained with auxiliary classifier
    ax2.plot(x_resids, avg_p_best, 'ro-', markerfacecolor='none', ls='dotted')
    # cutoff line
    ax2.axhline(y=0.5, color='grey', lw=.8, linestyle='dashed')

    ax2.set_xlim((res_i-.5, res_f+.5))
    ax2.set_xlabel('residue number')
    ax2.set_ylim((-0.05, 1.05))
    ax2.set_ylabel('average', rotation=90, labelpad=10)
    ax2.set_yticklabels([])
    ax2r = ax2.twinx()
    ax2r.set_ylim((-0.05, 1.05))
    ax2r.set_yticks([0, .5, 1])
    ax2r.set_yticklabels(['0', '0.5', '1'])
    ax2r.tick_params(axis='both', which='major', pad=15)

    tight_padding = 0.1
    fig.savefig(filename+'.png', format='png', bbox_inches='tight',
                pad_inches=tight_padding, dpi=dpi)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    LOGGER.info(f'Saturation mutagenesis figure saved to {filename}.png')

    # write a map in html format, to make figure clickable
    if html:
        all_axis = {'strip': ax0, 'table': ax1, 'bplot': ax2}

        # precompute some useful quantities for html code
        html_data = {}
        # dpi of printed figure
        html_data["dpi"] = dpi
        # figure size *before* tight
        html_data["fig_size"] = fig.get_size_inches()
        # tight bbox as used by fig.savefig()
        html_data["tight_bbox"] = fig.get_tightbbox(fig.canvas.get_renderer())
        # compute new origin and height, based on tight box and padding
        html_data["new_orig"] = html_data["tight_bbox"].min - tight_padding
        html_data["new_height"] = (html_data["tight_bbox"].height
                                   + 2*tight_padding)

        def get_area_coords(ax, d):
            assert ax_type in ("strip", "table", "bplot")
            # get bbox coordinates (x0, y0, x1, y1)
            bbox = ax.get_position().get_points()
            # get bbox coordinates in inches
            b_inch = bbox * d["fig_size"]
            # adjust bbox coordinates based on tight bbox
            b_adj = b_inch - d["new_orig"]
            # use html reference system (y = 1 - y)
            b_html = b_adj*np.array([1, -1]) + np.array([0, d["new_height"]])
            # convert to pixels
            b_px = (d["dpi"]*b_html).astype(int)
            # put in html format
            coords = '{},{},{},{}'.format(*b_px.flatten())
            # output
            return coords

        # html templates
        area_html = Template(
            '<area shape="rect" coords="$coords" '
            'id="{{map_id}}_$areaid" {{area_attrs}}> \n'
        )

        # write html
        with open(filename + '.html', 'w') as f:
            f.write('<div>\n')
            f.write('<map name="{{map_id}}" id="{{map_id}}" {{map_attrs}}>\n')
            for ax_type, ax in all_axis.items():
                fields = {'areaid': ax_type}
                fields['coords'] = get_area_coords(ax, html_data)
                f.write(area_html.substitute(fields))
            f.write('</map>\n')
            f.write('</div>\n')

        # populate info table that will be passed as a javascript variable
        arr_best = rhapsody_obj.getPredictions(classifier='best')
        PDB_coords = rhapsody_obj.getPDBcoords()
        info = {}
        for k in ['strip', 'table', 'bplot']:
            n_cols = 20 if k == 'table' else 1
            info[k] = [['']*nres_shown for i in range(n_cols)]
        for i, SAV in enumerate(rhapsody_obj.data['SAV coords']):
            acc, resid, aa_wt, aa_mut = SAV.split()
            resid = int(resid)
            # consider only residues shown in figure
            if not (res_i <= resid <= res_f):
                continue
            # SAV coordinates
            SAV_code = f'{aa_wt}{resid}{aa_mut}'
            # coordinates on table
            t_i = aa_map[aa_mut]
            t_j = resid - 1
            # coordinates on *shown* table
            ts_i = t_i
            ts_j = resid - res_i
            # predictions and other info
            rh_pred = table_best[t_i, t_j]
            av_rh_pred = avg_p_best[t_j]
            pclass = arr_best['path. class'][i]
            alii = {}
            if extra_plot is not None:
                alii['other'] = (table_other[t_i, t_j], avg_p_other[t_j])
            if PolyPhen2:
                alii['PolyPhen2'] = (table_PP2[t_i, t_j], avg_p_PP2[t_j])
            if EVmutation:
                alii['EVmutation'] = (table_EVmut[t_i, t_j], avg_p_EVmut[t_j])
            # compose message for table
            m = f'{SAV_code}: {rh_pred:4.2f} ({pclass})'
            for k, t in alii.items():
                m += f', {k}={t[0]:<4.2f}'
            info['table'][ts_i][ts_j] = m
            info['table'][aa_map[aa_wt]][ts_j] = f'{SAV_code[:-1]}: wild-type'
            # compose message for upper strip
            PDB_SAV = PDB_coords[i]['PDB SAV coords']
            PDB_size = PDB_coords[i]['PDB size']
            if PDB_size > 0:
                m = f'{PDB_SAV} (size: {PDB_size} res)'
            else:
                m = 'no PDB'
            info['strip'][0][ts_j] = m
            # compose message for bottom plot
            m = f'{SAV_code[:-1]}: {av_rh_pred:4.2f}'
            for k, t in alii.items():
                m += f', {k}={t[1]:<4.2f}'
            info['bplot'][0][ts_j] = m

        def create_info_msg(ax_type, d):
            text = '[ \n'
            for row in d:
                text += '  ['
                for m in row:
                    text += f'"{m}",'
                text += '], \n'
            text += ']'
            return text

        area_js = Template(
            '{{map_data}}["{{map_id}}_$areaid"] = { \n'
            '  "num_rows": $num_rows, \n'
            '  "num_cols": $num_cols, \n'
            '  "info_msg": $info_msg, \n'
            '}; \n'
        )

        # dump info in javascript format
        with open(filename + '.js', 'w') as f:
            f.write('var {{map_data}} = {{map_data}} || {}; \n')
            for ax_type, d in info.items():
                vars = {'areaid': ax_type}
                vars['num_rows'] = 20 if ax_type == 'table' else 1
                vars['num_cols'] = nres_shown
                vars['info_msg'] = create_info_msg(ax_type, d)
                f.write(area_js.substitute(vars))

        return info
    return
