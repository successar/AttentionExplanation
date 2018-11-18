from common import *

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import tight_layout

font = {'size'   : 12}
mpl.rc('font', **font)
mpl.style.use('seaborn-poster')
sns.set_palette(sns.color_palette(['#7570b3', '#d95f02', '#1b9e77']))
sns.palplot(sns.color_palette(['#7570b3', '#d95f02', '#1b9e77']))

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
    plt.savefig(dirname + '/' + filename + '.pdf', bbox_inches=extent)

def save_table_in_file(table, dirname, filename) :
    table.to_csv(dirname + '/' + filename + '.csv', index=True)

def annotate(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, legend='upper left', left=False) :
    if xlabel is not None : ax.set_xlabel(xlabel)
    if ylabel is not None : ax.set_ylabel(ylabel)
    if title is not None : ax.set_title(title)
    if xlim is not None : ax.set_xlim(*xlim)
    if ylim is not None : ax.set_ylim(*ylim)

    ax.legend(loc=legend, frameon=False)
    set_square_aspect(ax)
    sns.despine(ax=ax, left=left)

###########################################################################################################################

def plot_SP_histogram_by_class(ax, spcorr, yhat, bins=30, by_class=False) :
    sprho = np.array([x[0] for x in spcorr])
    sppval = np.array([x[1] for x in spcorr])

    measures = {"pval_sig" : {}, "mean" : {}, "std" : {}}

    yhat = np.round(yhat)
    unique_y = np.sort(np.unique(yhat))

    if by_class or len(unique_y) < 4:
        for y in unique_y :
            rho = sprho[yhat == y]
            pval = sppval[yhat == y]
            measures['pval_sig'][str(int(y))] = "{:.2f}".format((pval <= 0.05).sum() / len(pval))
            measures['mean'][str(int(y))] = np.mean(rho)
            measures['std'][str(int(y))] = np.std(rho)

            sns.distplot(rho, bins=bins, norm_hist=True, kde=False, 
                            hist_kws={"range" : (-1.0, 1.0), "alpha" : 0.6, "linewidth" : 0.5, "edgecolor":"k"}, ax=ax)
    else :
        measures['pval_sig']["Overall"] = "{:.2f}".format((sppval <= 0.05).sum() / len(sppval))
        measures['mean']["Overall"] = np.mean(sprho)
        measures['std']["Overall"] = np.std(sprho)
        sns.distplot(sprho, bins=bins, norm_hist=True, kde=False, 
                            hist_kws={"range" : (-1.0, 1.0), "alpha" : 0.6, "linewidth" : 0.5, "edgecolor":"k"}, ax=ax)

    ax.set_xlabel("Kendall $\\tau$")
    ax.set_yticks([])

    return pd.DataFrame(measures)

def plot_histogram_by_class(ax, values, yhat, bins=40, hist_lims=None, pval=None, pvallabel=None, by_class=False) :
    values = np.array(values)
    yhat = np.round(yhat)
    unique_y = np.sort(np.unique(yhat))

    if by_class or len(unique_y) < 4:
        for y in unique_y :
            rho = values[yhat == y]
            sns.distplot(rho, bins=bins, norm_hist=True, kde=False, 
                        hist_kws={"range" : hist_lims, "alpha" : 0.6, "linewidth" : 0.5, "edgecolor":"k"}, ax=ax)
    else :
        sns.distplot(values, bins=bins, norm_hist=True, kde=False, 
                    hist_kws={"range" : hist_lims, "alpha" : 0.6, "linewidth" : 0.5, "edgecolor":"k"}, ax=ax)

    ax.set_yticks([])

def plot_scatter_by_class(ax, X_vals, Y_vals, yhat, by_class=False) :
    yhat = np.round(yhat)
    unique_y = np.sort(np.unique(yhat))

    if by_class or len(unique_y) < 4:
        alpha = 0.5
        for y in unique_y :
            ax.scatter(X_vals[yhat == y], Y_vals[yhat == y], s=10, alpha=alpha)
            alpha -= 0.1
    else :
        ax.scatter(X_vals, Y_vals, s=10, alpha=0.7)