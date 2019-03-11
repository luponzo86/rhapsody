import numpy as np
import os
import warnings
from string import Template
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
    filename = os.path.splitext(filename)[0] + '.png'

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return
    else:
        from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(7, 7))
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
    plt.rcParams.update(plt.rcParamsDefault)
    LOGGER.info(f'Predictions distribution saved to {filename}')


def print_path_prob_figure(filename, bins, histo, dx, path_prob,
                     smooth_path_prob, cutoff=200):
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return
    else:
        from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(7, 7))
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
    plt.rcParams.update(plt.rcParamsDefault)
    LOGGER.info(f'Pathogenicity plot saved to {filename}')


def print_ROC_figure(filename, fpr, tpr, mean_auc):
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return
    else:
        from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
    plt.plot(fpr, tpr,       linestyle='-',  lw=2, color='r',
             label = f'Mean AUROC = {mean_auc:.3f}')
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

    matplotlib = try_import_matplotlib()
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


def print_sat_mutagen_figure(filename, rhapsody_obj,
    res_interval=None, min_interval_size=15,
    other_preds=None, PP2=True, EVmutation=True, EVmut_cutoff=-4.551,
    html=False, fig_height=8, dpi=300):

    # check inputs
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
    assert isinstance(fig_height, (int, float))
    assert isinstance(dpi, int)

    matplotlib = try_import_matplotlib()
    if matplotlib is None:
        return

    # delete extension from filename
    filename = os.path.splitext(filename)[0]

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

    # import pathogenicity probability from Rhapsody object
    p_full = rhapsody_obj.predictions['path. probability']
    p_aux  = None
    p_mix  = None
    if aux_preds_found:
        p_aux = rhapsody_obj.auxPreds['path. probability']
        p_mix = rhapsody_obj.mixPreds['path. probability']

    # select an appropriate interval, based on available predictions
    res_min = np.min(rhapsody_obj.SAVcoords['pos'])
    res_max = np.max(rhapsody_obj.SAVcoords['pos'])
    upper_lim = res_max + min_interval_size

    # create empty (20 x num_res) mutagenesis tables
    table_full = np.zeros((20, upper_lim), dtype=float)
    table_full[:] = 'nan'
    table_mix = table_full.copy()
    if other_preds is not None:
        table_other = table_full.copy()
    if PP2:
        table_PP2   = table_full.copy()
    if EVmutation:
        table_EVmut = table_full.copy()

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
        if other_preds is not None:
            table_other[aa_map[aa_mut], index] = other_preds[i]
        if PP2:
            s = float( rhapsody_obj.PP2output['pph2_prob'][i] )
            table_PP2[  aa_map[aa_mut], index] = s
        if EVmutation:
            s = rhapsody_obj.calcEVmutationFeats()['EVmut-DeltaE_epist'][i]
            table_EVmut[aa_map[aa_mut], index] = s/EVmut_cutoff*0.5

    # compute average pathogenicity profiles
    # NB: I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_p_full = np.nanmean(table_full, axis=0)
        avg_p_mix  = np.nanmean(table_mix,  axis=0)
        min_p = np.nanmin(table_mix, axis=0)
        max_p = np.nanmax(table_mix, axis=0)
        if other_preds is not None:
            avg_p_other = np.nanmean(table_other, axis=0)
        if PP2:
            avg_p_PP2   = np.nanmean(table_PP2,   axis=0)
        if EVmutation:
            avg_p_EVmut = np.nanmean(table_EVmut, axis=0)


    # use upper strip for showing additional info, such as PDB lengths
    upper_strip = np.zeros((1, upper_lim))
    upper_strip[:] = 'nan'
    PDB_sizes = np.zeros(upper_lim, dtype=int)
    PDB_coords = ['']*upper_lim
    for a, b in zip(rhapsody_obj.SAVcoords, rhapsody_obj.Uniprot2PDBmap):
        index = a['pos'] - 1
        if b['PDB size'] != 0:
            PDB_length = int(b['PDB size'])
            PDBID_chain = ':'.join(b['PDB SAV coords'][0].split()[:2])
            upper_strip[0, index] = PDB_length
            PDB_sizes[index] = PDB_length
            PDB_coords[index] = PDBID_chain
    max_PDB_size = np.nanmax(upper_strip[0, :])
    upper_strip[0, :] /= max_PDB_size

    # final data to show on figure
    if aux_preds_found:
        table_final = table_mix
        avg_p_final = avg_p_mix
        pclass_final = rhapsody_obj.mixPreds['path. class']
    else:
        table_final = table_full
        avg_p_final = avg_p_full
        pclass_final = rhapsody_obj.predictions['path. class']
    # avg_p_final = np.where(np.isnan(avg_p_full), avg_p_mix, avg_p_full)

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
    fig_width  = fig_height/2 # inches
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

    # padding for tick labels
    pad = 0.2/fig_width

    # top strip
    ax0.imshow(upper_strip[0:1, res_i-1:res_f], aspect='auto',
               cmap='YlGn', vmin=0, vmax=1)
    ax0.set_ylim((-0.45, .45))
    ax0.set_yticks([])
    ax0.set_ylabel(f'PDB size \n[0-{max_PDB_size} res] ', fontsize=14,
                   ha='right', va='center', rotation=0)
    ax0.set_xticks([])

    # mutagenesis table (heatmap)
    matplotlib.cm.coolwarm.set_bad(color='white')
    im = ax1.imshow(table_final[:, res_i-1:res_f], aspect='auto',
                    cmap='coolwarm', vmin=0, vmax=1)
    axcb.figure.colorbar(im, cax=axcb)
    ax1.set_yticks(np.arange(len(aa_list)))
    ax1.set_yticklabels(aa_list, ha='center', position=(-pad,0), fontsize=14)
    ax1.set_xticks(np.arange(5-res_i%5, res_f-res_i+1, 5))
    ax1.set_xticklabels([])
    ax1.set_ylabel('pathog. probability', labelpad=10)

    # average pathogenicity profile
    x_resids = np.arange(1, upper_lim+1)
    # shading showing range of values
    ax2.fill_between(x_resids, min_p, max_p, alpha=0.5, edgecolor='salmon',
                     facecolor='salmon')
    # plot average profile for other predictions, if available
    if other_preds is not None:
        ax2.plot(x_resids, avg_p_other, color='gray',  lw=1)
    if PP2:
        ax2.plot(x_resids, avg_p_PP2,   color='blue',  lw=1)
    if EVmutation:
        ax2.plot(x_resids, avg_p_EVmut, color='green', lw=1)
    # solid line for predictions obtained with full classifier
    ax2.plot(x_resids, avg_p_full, 'ro-')
    # dotted line for predictions obtained with auxiliary classifier
    ax2.plot(x_resids, avg_p_final, 'ro-', markerfacecolor='none', ls='dotted')
    # cutoff line
    ax2.hlines(0.5, -.5, upper_lim+.5, colors='grey', lw=.8,
               linestyle='dashed')

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
        html_data["new_orig"]   = html_data["tight_bbox"].min - tight_padding
        html_data["new_height"] = html_data["tight_bbox"].height + 2*tight_padding

        def get_area_coords(ax, d):
            assert ax_type in ("strip", "table", "bplot")
            # get bbox coordinates (x0, y0, x1, y1)
            bbox = ax.get_position().get_points()
            # get bbox coordinates in inches
            b_inch = bbox * d["fig_size"]
            # adjust bbox coordinates based on tight bbox
            b_adj = b_inch - d["new_orig"]
            # use html reference system (y = 1 - y)
            b_html = b_adj*np.array([1,-1]) + np.array([0, d["new_height"]])
            # convert to pixels
            b_px = (d["dpi"]*b_html).astype(int)
            # put in html format
            coords = '{},{},{},{}'.format(*b_px.flatten())
            # output
            return coords

        # html templates
        area_html = Template(
        '<area shape="rect" coords="$coords" ' + \
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
        info = {}
        for k in ['strip', 'table', 'bplot']:
            n_cols = 20 if k=='table' else 1
            info[k] = [['']*nres_shown for i in range(n_cols)]
        for i, SAV in enumerate(rhapsody_obj.SAVcoords):
            resid = SAV['pos']
            aa_wt = SAV['aa_wt']
            aa_mut = SAV['aa_mut']
            # consider only residues shown in figure
            if not (res_i <= resid <= res_f):
                continue
            # SAV & PDB coordinates
            SAV_code = f'{aa_wt}{resid}{aa_mut}'
            PDB_code = PDB_coords[resid - 1]
            # coordinates on table
            t_i = aa_map[aa_mut]
            t_j = resid - 1
            # coordinates on *shown* table
            ts_i = t_i
            ts_j = resid - res_i
            # predictions and other info
            rh_pred = table_final[t_i, t_j]
            av_rh_pred = avg_p_final[t_j]
            pclass = pclass_final[i]
            others = {}
            if other_preds is not None:
                others['other'] = (table_other[t_i, t_j], avg_p_other[t_j])
            if PP2:
                others['PP2']   = (table_PP2[t_i, t_j],   avg_p_PP2[t_j])
            if EVmutation:
                others['EVmut'] = (table_EVmut[t_i, t_j], avg_p_EVmut[t_j])
            # compose message for table
            m = f'{SAV_code}: {rh_pred:4.2f} ({pclass})'
            for k,t in others.items():
                m += f', {k}={t[0]:<4.2f}'
            info['table'][ts_i][ts_j] = m
            info['table'][aa_map[aa_wt]][ts_j] = f'{SAV_code[:-1]}: wild-type'
            # compose message for upper strip
            m = f'{PDB_code}'
            if PDB_code != '':
                m += f' (size: {PDB_sizes[t_j]} res)'
            info['strip'][0][ts_j] = m
            # compose message for bottom plot
            m = f'{SAV_code[:-1]}: {av_rh_pred:4.2f}'
            for k,t in others.items():
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
        '{{map_data}}["{{map_id}}_$areaid"] = { \n' + \
        '  "num_rows": $num_rows, \n' + \
        '  "num_cols": $num_cols, \n' + \
        '  "info_msg": $info_msg, \n' + \
        '}; \n'
        )

        # dump info in javascript format
        with open(filename + '.js', 'w') as f:
            f.write('var {{map_data}} = {{map_data}} || {}; \n')
            for ax_type, d in info.items():
                vars = {'areaid': ax_type}
                vars['num_rows'] = 20 if ax_type=='table' else 1
                vars['num_cols'] = nres_shown
                vars['info_msg'] = create_info_msg(ax_type, d)
                f.write(area_js.substitute(vars))

        return info
    return
