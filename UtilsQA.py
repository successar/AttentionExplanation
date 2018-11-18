from sklearn.metrics import accuracy_score
from common import *
from plotting import *

from scipy.stats import kendalltau

#################### Preprocessing Begin ############################################

def evaluate_and_print(model, data) :
    P, Q, E, A = data.P, data.Q, data.E, data.A
    yhat, attn = model.evaluate(P, Q, E)
    yhat = np.array(yhat)
    rep = accuracy_score(A, yhat)
    print("Accuracy Score : ", rep)

    return yhat, attn
            
def print_example(vec, data, predict, attn, n) :
    print(" ".join(vec.map2words(data.Q[n])))
    print("Truth : " , vec.idx2entity[data.A[n]], " | Predicted : ", vec.idx2entity[predict[n]])
    print_attn(sentence=vec.map2words(data.P[n]), attention=attn[n])

###########################################################################################################################

from scipy.stats import mode
from sklearn.metrics import classification_report 

def plot_permutations(permutations, data, yhat, attn, by_class=False, dirname='') :
    ad_y, ad_diffs = permutations
    ad_diffs = 0.5*np.array(ad_diffs)

    median_diff = np.median(ad_diffs, 1)
    
    perms_perc = np.array([(ad_y[i] == yhat[i]).mean() for i in range(len(ad_y))])
    diffs_perc = [(ad_diffs[i] < 5e-2).mean() for i in range(len(ad_diffs))]
    X = data.P

    max_attn = calc_max_attn(X, attn)

    fig, ax = init_gridspec(3, 3, 2)

    plot_scatter_by_class(ax[0], max_attn, perms_perc, yhat, by_class=by_class)
    annotate(ax[0], xlabel=('Max attention'), ylabel=(''))

    plot_scatter_by_class(ax[1], max_attn, median_diff, yhat, by_class=by_class)
    annotate(ax[1], xlabel=('Max attention'), ylabel='$\\Delta\\hat{y}$', xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

    adjust_gridspec()

    save_axis_in_file(fig, ax[1], dirname, "Permutation_MAvDY")

    show_gridspec()

##########################################################################################################

def plot_multi_adversarial(data, yhat, attn, adversarial_outputs, epsilon=0.01, by_class=False, dirname='') :
    fig, ax = init_gridspec(3, 3, 3)

    X = data.P

    ad_y, ad_attn, ad_diffs = adversarial_outputs
    ad_y = np.array(ad_y)
    ad_diffs = 0.5 * np.array(ad_diffs)[:, :, 0]

    ad_attn = [np.vstack([ad_attn[i], attn[i]]) for i in range(len(attn))]
    ad_diffs = np.array([np.append(ad_diffs[i], 0) for i in range(len(attn))])
    ad_y = np.array([np.append(ad_y[i], yhat[i]) for i in range(len(attn))])

    ad_sims = 1 - ad_diffs
    ad_probs = ad_sims / ad_sims.sum(1)[:, None]

    mean_attn = [(ad_attn[i] * ad_probs[i][:, None]).sum(0) for i in range(len(attn))]

    K = ad_attn[0].shape[0]

    for k in range(K) :
        print("Accuracy Score : " , accuracy_score(yhat, ad_y[:, k]))

    jds = []
    emax_jds, emax_adv_attn, emax_ad_y = [], [], []

    for i in range(len(X)) :
        L = len(X[i])
        multi_jds_from_mean = np.array([jsd(mean_attn[i][1:L-1], ad_attn[i][k][1:L-1]) for k in range(K)])
        jds.append(multi_jds_from_mean)

        multi_jds_from_true = np.array([jsd(attn[i][1:L-1], ad_attn[i][k][1:L-1]) for k in range(K)])
        multi_jds_from_true[ad_diffs[i] > epsilon] = -1
        pos = np.argmax(multi_jds_from_true)
        emax_jds.append(multi_jds_from_true[pos])
        emax_adv_attn.append(ad_attn[i][pos])
        emax_ad_y.append(ad_y[i][pos])

    jds = np.array(jds)
    emax_jds = np.array(emax_jds)
    mean_jds = (ad_probs * jds).sum(-1)

    plot_histogram_by_class(ax[0], mean_jds, yhat, hist_lims=(0, 0.7), pval=g30, pvallabel="JSD > 0.34", by_class=by_class)
    annotate(ax[0], xlabel=("Weighted Mean JS Divergence"), left=True, xlim=(-0.02, 0.72))

    plot_histogram_by_class(ax[1], emax_jds, yhat, hist_lims=(0, 0.7), pval=g30, pvallabel="JSD > 0.34", by_class=by_class)
    annotate(ax[1], xlabel=("Max JS Divergence within $\\epsilon$"), left=True, xlim=(-0.02, 0.72))

    max_attn = calc_max_attn(X, attn)
    plot_scatter_by_class(ax[2], max_attn, emax_jds, yhat, by_class=by_class)
    annotate(ax[2], xlim=(-0.05, 1.05), ylim=(-0.05, 0.7), xlabel=("Max Attention"), ylabel=("Max JS Divergence within $\\epsilon$"), legend='lower left')

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "WeightedMeanJDS_Hist")
    save_axis_in_file(fig, ax[1], dirname, "eMaxJDS_Hist")
    save_axis_in_file(fig, ax[2], dirname, "eMaxJDS_Scatter")
    show_gridspec()

    return emax_jds, emax_adv_attn, emax_ad_y

def print_adversarial_example(sentence, attn, attn_new, latex=False) :
    L = len(sentence)
    print_attn(sentence[1:L-1], attn[1:L-1], latex=latex)
    print('-'*20)
    print_attn(sentence[1:L-1], attn_new[1:L-1], latex=latex)

###########################################################################################################################

def process_grads(grads) :
    for k in grads :
        xxe = grads[k]
        for i in range(len(xxe)) :
            xxe[i] = np.abs(xxe[i])
            xxe[i] = xxe[i] / xxe[i].sum()

def plot_grads(data, yhat, attn, grads, by_class=False, dirname='') :
    X = data.P
    fig, ax = init_gridspec(3, 3, len(grads)*3)

    colnum = 0

    max_attn = np.array([max(attn[i][1:len(X[i])-1]) for i in range(len(attn))])

    pval_tables = {}

    for k in grads :
        xxe = grads[k]    
        a, x = [], []
        spcorrs = []
        
        for i in range(len(xxe)) :
            L = len(X[i])
            a += list(attn[i][1:L-1])
            x += list(xxe[i][1:L-1])
            spcorr = kendalltau(list(attn[i][1:L-1]), list(xxe[i][1:L-1]))
            spcorrs.append(spcorr)
            
        coef = np.corrcoef(a, x)[0, 1]
        axes = ax[colnum]
        colnum += 1

        axes.hexbin(a, x, mincnt=1, gridsize=50, cmap=conscmap, extent=(0, 1, 0, 1))
        annotate(axes, xlabel="Attention", ylabel="Normalised Gradient", title=k + ' rho=' + str(coef), xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

        axes = ax[colnum]
        colnum += 1

        pval_tables[k] = plot_SP_histogram_by_class(axes, spcorrs, yhat, by_class=by_class)
        annotate(axes, left=True)

        axes = ax[colnum]
        colnum += 1

        prcorrs = np.array([x[0] for x in spcorrs])
        plot_scatter_by_class(axes, max_attn, prcorrs, yhat, by_class=by_class)
        annotate(axes, xlabel="Max Attention", ylabel="Kendall $\\tau$", title=k + ' rho=' + str(coef), xlim=(-0.05, 1.05), ylim=(-1.05, 1.05))

        if k == 'XxE[X]' :
            spcorrs_xxex = spcorrs

    adjust_gridspec()

    save_axis_in_file(fig, ax[4], dirname, 'GradientXHist')
    save_axis_in_file(fig, ax[5], dirname, 'GradientXScatter')
    save_table_in_file(pval_tables['XxE[X]'], dirname, 'GradientPval')

    show_gridspec()
    return spcorrs_xxex

###########################################################################################################################

def plot_y_diff(data, yhat, attn, y_diff, title="Attention vs change in output", 
                    xlabel="Attention", ylabel="$\\bigtriangleup y$", save_name=None, dirname='') :
    X = data.P

    a = []
    b = []
    spcorrs = []

    for i in range(len(attn)) :
        L = len(X[i])
        a += list(attn[i][1:L-1])
        f = np.abs(y_diff[i][1:L-1])
        f = f / f.sum()
        b += list(f)
        spcorrs.append(kendalltau(attn[i][1:L-1], np.abs(y_diff[i][1:L-1])))

    fig, ax = init_gridspec(3, 3, 3)

    plot_SP_histogram_by_class(ax[0], spcorrs, yhat)
    annotate(ax[0], left=True)

    max_attn = calc_max_attn(X, attn)
    prcorrs = np.array([x[0] for x in spcorrs])
    plot_scatter_by_class(ax[2], max_attn, prcorrs, yhat)
    annotate(ax[2], xlabel="Max Attention", ylabel="Kendall $\\tau$", xlim=(-0.05, 1.05), ylim=(-1.05, 1.05))

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_SP')
        save_axis_in_file(fig, ax[2], dirname, save_name + '_Scatter')

    show_gridspec()
    return spcorrs

def plot_attn_diff(X, attn, attn_new, yhat, title="Old vs New Attention", xlabel="Old Attention", ylabel="New Attention", save_name=None, dirname='') :
    a = []
    b = []
    spcorrs = []

    for i in range(len(attn)) :
        L = len(X[i])
        a += list(attn[i][1:L-1])
        b += list(attn_new[i][1:L-1])
        spcorrs.append(kendalltau(attn[i][1:L-1], attn_new[i][1:L-1]))

    fig, ax = init_gridspec(3, 3, 3)

    plot_SP_histogram_by_class(ax[0], spcorrs, yhat)
    annotate(ax[0], left=True)

    # ax[1].hexbin(a, b, mincnt=1, gridsize=100, cmap=conscmap)
    # annotate(ax[1], xlabel=xlabel, ylabel=ylabel, title=title)

    max_attn = calc_max_attn(X, attn)
    prcorrs = np.array([x[0] for x in spcorrs])
    plot_scatter_by_class(ax[2], max_attn, prcorrs, yhat)
    annotate(ax[2], xlabel="Max Attention", ylabel="Kendall $\\tau$", xlim=(-0.05, 1.05), ylim=(-1.05, 1.05))

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_SP')
        # save_axis_in_file(fig, ax[1], dirname, save_name + '_Hex')
        save_axis_in_file(fig, ax[2], dirname, save_name + '_Scatter')

    show_gridspec()
    return spcorrs

###########################################################################################################################################

def generate_medians_from_sampling_top(yhat, attn, output, dirname='') :
    perts_attn, words_sampled, best_attn_idxs, perts_diffs = output

    n_top = perts_attn[0].shape[2]
    top_attns = []
    med_attns = []
    mean_out_diffs = []
    spcorrs = []

    for i in range(len(perts_attn)) :
        attn_ex = []
        med_out_ex = []
        for j in range(n_top) :
            actual_attn = attn[i][best_attn_idxs[i][j]]
            median_attn = np.median(perts_attn[i][best_attn_idxs[i][j], :, j])

            top_attns.append(actual_attn)
            med_attns.append(median_attn)

            mean_out_diff = 0.5*perts_diffs[i][:, j].mean()
            mean_out_diffs.append(mean_out_diff)

            attn_ex.append(actual_attn)
            med_out_ex.append(mean_out_diff)

        spcorrs.append(kendalltau(attn_ex, med_out_ex))

    fig, ax = init_gridspec(3, 3, 3)

    ax[0].hexbin(top_attns, med_attns, mincnt=1, gridsize=100, cmap=conscmap, extent=(0, 1, 0, 1))
    annotate(ax[0], xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Attention", ylabel="Median Attention")

    ax[1].hexbin(top_attns, mean_out_diffs, mincnt=1, gridsize=100, cmap=conscmap)
    annotate(ax[1], xlabel="Attention", ylabel="$\\bigtriangleup \\hat{y}$", title="Attention vs output with substitutions")

    plot_SP_histogram_by_class(ax[2], spcorrs, yhat, bins=10)
    annotate(ax[2], left=True)

    adjust_gridspec()

    save_axis_in_file(fig, ax[0], dirname, 'substitute_median')
    save_axis_in_file(fig, ax[2], dirname, 'Substitute_y_SP')

    show_gridspec()

######################################################################################################################################################

def generate_graphs(dataset, model, test_data) :
    print(dataset.name)

    # gradients = model.gradient_mem(test_data.P, test_data.Q, test_data.E)
    # process_grads(gradients)

    # grad_corrs = plot_grads(test_data, test_data.predict, test_data.attn_hat, gradients, 
    #                                    by_class=dataset.by_class, dirname=model.dirname)

    # try :
    #     print("Adversarial")
    #     multi_adversarial_outputs = pload(model, 'multi_adversarial')
    #     _ = plot_multi_adversarial(test_data, test_data.predict, test_data.attn_hat, 
    #                                                         multi_adversarial_outputs, 
    #                                                         epsilon=0.05,
    #                                                         by_class=dataset.by_class, dirname=model.dirname)
    # except FileNotFoundError :
    #     print("Multi Adversarial Outputs doesn't exist")

    # try :
    #     print("Permutation")
    #     perms = pload(model, 'permutations')
    #     plot_permutations(perms, test_data, test_data.predict, test_data.attn_hat, 
    #                             by_class=dataset.by_class, dirname=model.dirname)
    # except FileNotFoundError :
    #     print("Permutatiosn Outputs doesn't exist")

    print("Zero Runs")
    zero_outputs, zero_O_diff, zero_H_diff = pload(model, 'zero_output')
    _ = plot_y_diff(test_data, test_data.predict, test_data.attn_hat, zero_H_diff,
                            xlabel='Attention', ylabel="H(x|c) - H(x)", 
                            title="Attention vs change in hidden state", save_name="hxc-hx", dirname=model.dirname)

    _ = plot_y_diff(test_data, test_data.predict, test_data.attn_hat, zero_O_diff, 
                        xlabel="Attention", ylabel="p(y|x, c) - p(y|x)", 
                        title="Attention vs change in output", save_name="pyxc-pyx", dirname=model.dirname)

    try :
        print("Remove and Run")
        remove_outputs = pload(model, 'remove_and_run')
        _ = plot_y_diff(test_data, test_data.predict, test_data.attn_hat, remove_outputs,
                        xlabel="Attention", ylabel="p(y|x, c) - p(y|c)", 
                        title="Attention vs change in output", save_name="pyxc-pyc", dirname=model.dirname)
    except FileNotFoundError:
        print("Remove Outputs doesn't exist")

    print("Pushing to Directory")
    push_graphs_to_main_directory(model, dataset.name)

    print("="*30)
    print("="*30)