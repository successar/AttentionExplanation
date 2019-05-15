from Transparency.common_code.common import *

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import tight_layout

mpl.style.use('seaborn-poster')
sns.set_palette(sns.color_palette(['#43406b', '#d15a00', '#27f77d']))
# sns.palplot(sns.color_palette(['#43406b', '#d15a00', '#27f77d']))
# sns.set_palette('cubehelix')

font = {'size'   : 17}
mpl.rc('font', **font)

histcolor = '#143f7a'

conscmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#142c89", "#142c89"])

def init_gridspec(nrow, ncol, nax) :
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig)
    axes = []
    for i in range(nax) :
        axes.append(plt.subplot(gs[i//ncol, i%ncol]))

    return fig, axes

def adjust_gridspec() :
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

def show_gridspec() :
    plt.show()

def set_square_aspect(axes) :
    x0,x1 = axes.get_xlim()
    y0,y1 = axes.get_ylim()
    axes.set_aspect(abs(x1-x0)/abs(y1-y0))

def save_axis_in_file(fig, ax, dirname, filename):
    ax.set_title("")

    renderer = tight_layout.get_renderer(fig)
    inset_tight_bbox = ax.get_tightbbox(renderer)
    extent = inset_tight_bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(dirname, filename + '.png'), bbox_inches=extent, dpi=1000)

    renderer = tight_layout.get_renderer(fig)
    inset_tight_bbox = ax.get_tightbbox(renderer)
    extent = inset_tight_bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(dirname, filename + '.svg'), bbox_inches=extent)

    if 'sst' not in dirname and 'readmission' not in dirname:
        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        
    renderer = tight_layout.get_renderer(fig)
    inset_tight_bbox = ax.get_tightbbox(renderer)
    extent = inset_tight_bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(os.path.join(dirname, filename + '.pdf'), bbox_inches=extent)

def save_table_in_file(table, dirname, filename) :
    table.to_csv(os.path.join(dirname, filename + '.csv'), index=True)

def annotate(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, legend='upper left') :
    if xlabel is not None : ax.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None : ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(labelsize=20)
    if title is not None : ax.set_title(title)
    if xlim is not None : ax.set_xlim(*xlim)
    if ylim is not None : ax.set_ylim(*ylim)

    set_square_aspect(ax)
    sns.despine(ax=ax)
    if legend is None and ax.get_legend() is not None : ax.get_legend().remove()

###########################################################################################################################

def plot_SP_histogram_by_class(ax, spcorr, yhat, bins=30) :
    sprho = np.array([x[0] for x in spcorr])
    sppval = np.array([x[1] for x in spcorr])

    measures = {"pval_sig" : {}, "mean" : {}, "std" : {}}

    measures['pval_sig']["Overall"] = "{:.2f}".format((sppval <= 0.05).sum() / len(sppval))
    measures['mean']["Overall"] = np.mean(sprho)
    measures['std']["Overall"] = np.std(sprho)

    unique_y = None
    if len(yhat.shape) == 1 or yhat.shape[1] == 1:
        yhat = yhat.flatten()
        yhat = np.round(yhat)
        unique_y = np.sort(np.unique(yhat))

    if unique_y is not None and len(unique_y) < 4:
        for y in unique_y :
            rho = sprho[yhat == y]
            pval = sppval[yhat == y]
            measures['pval_sig'][str(int(y))] = "{:.2f}".format((pval <= 0.05).sum() / len(pval))
            measures['mean'][str(int(y))] = np.mean(rho)
            measures['std'][str(int(y))] = np.std(rho)
            ax.hist(rho, bins=bins, range=(-1.0, 1.0), alpha=0.6, linewidth=0.5, edgecolor='k', weights=np.ones(len(rho))/len(rho))
    else :
        ax.hist(sprho, bins=bins, range=(-1.0, 1.0), alpha=0.6, linewidth=0.5, edgecolor='k', weights=np.ones(len(sprho))/len(sprho))

    return pd.DataFrame(measures)

def plot_SP_density_by_class(ax, spcorr, yhat, linestyle='-') :
    sprho = np.array([x[0] for x in spcorr])
    sppval = np.array([x[1] for x in spcorr])

    measures = {"pval_sig" : {}, "mean" : {}, "std" : {}}

    unique_y = None
    if len(yhat.shape) == 1 or yhat.shape[1] == 1:
        yhat = yhat.flatten()
        yhat = np.round(yhat)
        unique_y = np.sort(np.unique(yhat))

    if False : #unique_y is not None and len(unique_y) < 4:
        for y in unique_y :
            rho = sprho[yhat == y]
            pval = sppval[yhat == y]
            measures['pval_sig'][str(int(y))] = "{:.2f}".format((pval <= 0.05).sum() / len(pval))
            measures['mean'][str(int(y))] = np.mean(rho)
            measures['std'][str(int(y))] = np.std(rho)
            sns.kdeplot(rho, linewidth=2, linestyle=linestyle, ax=ax)
    else :
        measures['pval_sig']["Overall"] = "{:.2f}".format((sppval <= 0.05).sum() / len(sppval))
        measures['mean']["Overall"] = np.mean(sprho)
        measures['std']["Overall"] = np.std(sprho)
        sns.kdeplot(sprho, linewidth=2, linestyle=linestyle, color='k', ax=ax)

    return pd.DataFrame(measures)

def plot_histogram_by_class(ax, values, yhat, bins=40, hist_lims=None) :
    values = np.array(values)

    unique_y = None
    if len(yhat.shape) == 1 or yhat.shape[1] == 1:
        yhat = yhat.flatten()
        yhat = np.round(yhat)
        unique_y = np.sort(np.unique(yhat))

    if unique_y is not None and len(unique_y) < 4:
        for y in unique_y :
            rho = values[yhat == y]
            ax.hist(rho, bins=bins, range=hist_lims, alpha=0.6, linewidth=0.5, edgecolor='k', weights=np.ones(len(rho))/len(rho))
    else :
        ax.hist(values, bins=bins, range=hist_lims, alpha=0.6, linewidth=0.5, edgecolor='k', weights=np.ones(len(values))/len(values))

def plot_violin_by_class(ax, X_vals, Y_vals, yhat, xlim, bins=4) :
    bins = xlim[0] + np.arange(bins+1) / bins * (xlim[1]+1e-4 - xlim[0])
    xbins = np.digitize(X_vals, bins[1:])
    order = ["[" + "{:.2f}".format(bins[p]) + ',\n' + "{:.2f}".format(bins[p+1]) + ")" for p in np.arange(len(bins)-1)]

    xnames = []
    for p in xbins :
        xnames.append("[" + "{:.2f}".format(bins[p]) + ',\n' + "{:.2f}".format(bins[p+1]) + ")")

    classes = np.zeros((len(yhat,)))
    if len(yhat.shape) == 1 or yhat.shape[1] == 1:
        yhat = yhat.flatten()
        yhat = np.round(yhat)
        unique_y = np.sort(np.unique(yhat))
        if len(unique_y) < 4 :
            classes = yhat

    df = pd.DataFrame({'bin' : xnames, 'val' : Y_vals, 'class' : classes})
    sns.violinplot(data=df, y="bin", x="val", hue="class", ax=ax, linewidth=1.0, order=order, cut=0.02, inner='quartiles', dodge=True)

    ax.get_legend().remove()

def plot_scatter_by_class(ax, X_vals, Y_vals, yhat) :
    classes = np.zeros((len(yhat,)))
    if len(yhat.shape) == 1 or yhat.shape[1] == 1:
        yhat = yhat.flatten()
        yhat = np.round(yhat)
        unique_y = np.sort(np.unique(yhat))
        if len(unique_y) < 4 :
            classes = yhat

    unique_y = np.sort(np.unique(classes))

    df = pd.DataFrame({'bin' : X_vals, 'val' : Y_vals, 'class' : classes})
    if len(unique_y) <= 1 :
        sns.scatterplot(x='bin', y='val', data=df, ax=ax, alpha=1.0, s=10, linewidth=0)
    else :
        sns.scatterplot(x='bin', y='val', hue='class', style='class', data=df, ax=ax, alpha=0.7, s=10, linewidth=0)
        ax.get_legend().remove()