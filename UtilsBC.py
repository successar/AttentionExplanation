from scipy.stats import kendalltau
from sklearn.metrics import classification_report

from common import *
from plotting import *

#################### Preprocessing Begin ############################################

def evaluate_and_print(model, data) :
    X, y = data.X, data.y
    yhat, attn = model.evaluate(X)
    yhat = np.array(yhat)[:, 0]
    rep = classification_report(y, (yhat > 0.5))
    print(rep)

    return yhat, attn

##################################################################################################################

def plot_diff(sentence_1, idx, new_word, old_attn, new_attn) :
    # need vec in environment
    L = len(sentence_1)
    print_attn(sentence_1[1:L-1], old_attn[1:L-1], idx-1)
    sentence_1 = [x for x in sentence_1]
    sentence_1[idx] = new_word
    print("-"*20)
    print_attn(sentence_1[1:L-1], new_attn[1:L-1], idx-1)

#######################################################################################################################

def process_grads(grads) :
    for k in grads :
        xxe = grads[k]
        for i in range(len(xxe)) :
            xxe[i] = np.abs(xxe[i])
            xxe[i] = xxe[i] / xxe[i].sum()

def plot_grads(X, attn, grads, yhat, dirname='') :
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

        pval_tables[k] = plot_SP_histogram_by_class(axes, spcorrs, yhat)
        annotate(axes, left=True)

        axes = ax[colnum]
        colnum += 1

        prcorrs = np.array([x[0] for x in spcorrs])
        plot_scatter_by_class(axes, max_attn, prcorrs, yhat)
        annotate(axes, xlabel="Max Attention", ylabel="Kendall $\\tau$", title=k + ' rho=' + str(coef), xlim=(-0.05, 1.05), ylim=(-1.05, 1.05))

        if k == 'XxE[X]' :
            spcorrs_xxex = spcorrs

    adjust_gridspec()
    save_axis_in_file(fig, ax[4], dirname, 'GradientXHist')
    save_axis_in_file(fig, ax[5], dirname, 'GradientXScatter')
    save_table_in_file(pval_tables['XxE[X]'], dirname, 'GradientPval')

    show_gridspec()
    return spcorrs_xxex

##########################################################################################################################

def generate_medians_from_sampling_top(output, attn, yhat, dirname='') :
    perts_attn, words_sampled, best_attn_idxs, perts_output = output

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
            median_attn = np.median(perts_attn[best_attn_idxs[i][j], j])
            top_attns.append(actual_attn)
            med_attns.append(median_attn)

            mean_out_diff = np.abs(perts_output[i][:, j] - yhat[i]).mean()
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

def get_distractors(sampled_output, attn_hat) :
    perts_attn, words_sampled, best_attn_idxs, perts_output = sampled_output
    perts_attn_med = [np.median(perts_attn[i], 1) for i in range(len(perts_attn))]
    
    n_top = perts_attn[0].shape[2]
    median_attention = []
    for i in range(len(perts_attn_med)) :
        med = perts_attn_med[i]
        med_at_idx = []
        for j in range(n_top) :
            med_at_idx.append(med[best_attn_idxs[i][j], j])

        median_attention.append(med_at_idx)
        
    num_high_med_high_attn = 0
    num_high_attn = 0
    
    distractors = []
    for i, x in enumerate(best_attn_idxs) :
        idxs = best_attn_idxs[i]
        for j, idx in enumerate(idxs) :
            if attn_hat[i][idx] > 0.5 : 
                num_high_attn += 1
                if median_attention[i][j] > 0.5 :
                    num_high_med_high_attn += 1
                    distractors.append((i, j, idx))
                    
    print(num_high_med_high_attn / num_high_attn * 100, num_high_med_high_attn, num_high_attn)
    return distractors

def print_few_distractors(vec, X, attn_hat, sampled_output, distractors) :
    perts_attn, words_sampled, best_attn_idxs, perts_outputs = sampled_output

    for (i, j, idx) in distractors :
        sentence = vec.map2words(X[i])
        attn = attn_hat[i]
        pos = j
        w = np.argsort(perts_attn[i][idx, :, j])[len(perts_attn[i][idx, :, j])//2]

        new_word = vec.idx2word[int(words_sampled[i][w])]

        plot_diff(sentence, idx, new_word, attn, perts_attn[i][:, w, pos])
        print("*"*40)

###########################################################################################################################

def plot_permutations(permutations, X, yhat, attn, dirname='') :
    med_diff = np.abs(np.array(permutations) - yhat[:, None])
    med_diff = np.median(med_diff, 1)

    max_attn = np.array([max(attn[i][1:len(X[i])-1]) for i in range(len(attn))])
    fig, ax = init_gridspec(3, 3, 2)

    ax[0].scatter(yhat, med_diff, s=1)
    annotate(ax[0], xlabel="$\\hat{y}$", ylabel="$\\bigtriangleup \\hat{y}$")

    plot_scatter_by_class(ax[1], max_attn, med_diff, yhat)
    annotate(ax[1], xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel="Max attention", ylabel="$\\bigtriangleup \\hat{y}$")

    adjust_gridspec()
    save_axis_in_file(fig, ax[1], dirname, "Permutation_MAvDY")

    show_gridspec()
    return med_diff

############################################################################################################

def plot_multi_adversarial(X, yhat, attn, adversarial_outputs, dirname='') :
    ad_y, ad_attn = adversarial_outputs
    ad_y = np.array(ad_y)
    ad_diffs = np.abs((ad_y - yhat[:, None]))

    ad_attn = [np.vstack([ad_attn[i], attn[i]]) for i in range(len(attn))]
    ad_diffs = np.array([np.append(ad_diffs[i], 0) for i in range(len(attn))])
    ad_y = np.array([np.append(ad_y[i], yhat[i]) for i in range(len(attn))])

    ad_sims = 1 - ad_diffs
    ad_probs = ad_sims / ad_sims.sum(1)[:, None]

    mean_attn = [(ad_attn[i] * ad_probs[i][:, None]).sum(0) for i in range(len(attn))]

    K = ad_attn[0].shape[0]

    jds = []
    emax_jds, emax_adv_attn, emax_ad_y = [], [], []

    for i in range(len(X)) :
        L = len(X[i])
        multi_jds_from_mean = np.array([jsd(mean_attn[i][1:L-1], ad_attn[i][k][1:L-1]) for k in range(K)])
        jds.append(multi_jds_from_mean)

        multi_jds_from_true = np.array([jsd(attn[i][1:L-1], ad_attn[i][k][1:L-1]) for k in range(K)])
        multi_jds_from_true[ad_diffs[i] > 1e-2] = -1
        pos = np.argmax(multi_jds_from_true)
        emax_jds.append(multi_jds_from_true[pos])
        emax_adv_attn.append(ad_attn[i][pos])
        emax_ad_y.append(ad_y[i][pos])

    jds = np.array(jds)
    emax_jds = np.array(emax_jds)
    mean_jds = (ad_probs * jds).sum(-1)

    fig, ax = init_gridspec(3, 3, 3)

    plot_histogram_by_class(ax[0], mean_jds, yhat, hist_lims=(0, 0.7), pval=g30, pvallabel="JSD > 0.34")
    annotate(ax[0], xlabel="Weighted Mean JS Divergence", left=True)

    plot_histogram_by_class(ax[1], emax_jds, yhat, hist_lims=(0, 0.7), pval=g30, pvallabel="JSD > 0.34")
    annotate(ax[1], xlabel="Max JS Divergence within $\\epsilon$", left=True)

    max_attn = calc_max_attn(X, attn)
    plot_scatter_by_class(ax[2], max_attn, emax_jds, yhat)
    annotate(ax[2], xlim=(-0.05, 1.05), ylim=(-0.05, 0.7), xlabel="Max Attention", ylabel="Max JS Divergence within $\\epsilon$", legend="lower left")

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

############################################################################################################

def plot_y_diff(X, attn, yhat, ynew_list, title="Attention vs change in output", 
                    xlabel="Attention", ylabel="$\\bigtriangleup y$", save_name=None, dirname='') :
    y_diff = [ynew_list[i] - yhat[i] for i in range(len(yhat))]
    
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
    
    ax[1].hexbin(a, b, mincnt=1, gridsize=100, cmap=conscmap, extent=(0, 1, 0, 1))
    annotate(ax[1], xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel=xlabel, ylabel=ylabel, title=title)

    max_attn = calc_max_attn(X, attn)
    prcorrs = np.array([x[0] for x in spcorrs])
    plot_scatter_by_class(ax[2], max_attn, prcorrs, yhat)
    annotate(ax[2], xlabel="Max Attention", ylabel="Kendall $\\tau$", xlim=(-0.05, 1.05), ylim=(-1.05, 1.05))

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_SP')
        save_axis_in_file(fig, ax[1], dirname, save_name + '_Hex')
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

    ax[1].hexbin(a, b, mincnt=1, gridsize=100, cmap=conscmap)
    annotate(ax[1], xlabel=xlabel, ylabel=ylabel, title=title)

    max_attn = calc_max_attn(X, attn)
    prcorrs = np.array([x[0] for x in spcorrs])
    plot_scatter_by_class(ax[2], max_attn, prcorrs, yhat)
    annotate(ax[2], xlabel="Max Attention", ylabel="Kendall $\\tau$", xlim=(-0.05, 1.05), ylim=(-1.05, 1.05))

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_SP')
        save_axis_in_file(fig, ax[1], dirname, save_name + '_Hex')
        save_axis_in_file(fig, ax[2], dirname, save_name + '_Scatter')

    show_gridspec()
    return spcorrs

######################################################################################################

def plot_pertub_embedding(X, attn, yhat, perturb_E_outputs, dirname='') :
    y_diff = [np.median(np.abs(perturb_E_outputs[i] - yhat[i]), 1) for i in range(len(X))]

    a = []
    b = []
    spcorrs = []

    for i in range(len(attn)) :
        L = len(X[i])
        a += list(attn[i][1:L-1])
        f = y_diff[i][1:L-1]
        f = f / f.sum()
        b += list(f)
        spcorrs.append(kendalltau(attn[i][1:L-1], y_diff[i][1:L-1]))

    fig, ax = init_gridspec(3, 3, 2)

    plot_SP_histogram_by_class(ax[0], spcorrs, yhat)
    annotate(ax[0], left=True)

    ax[1].hexbin(a, b, mincnt=1, gridsize=100, cmap=conscmap)
    annotate(ax[1], xlabel="Attention", ylabel="$p(y|E+e) - p(y|E)$")

    save_axis_in_file(fig, ax[0], dirname, 'Embedding_SP')
    save_axis_in_file(fig, ax[1], dirname, 'Embedding_Hex')

    show_gridspec()

##########################################################################################################

def generate_graphs(dataset, model, test_data) :
    print(dataset.name)
    try :
        print("Sampled Outputs")
        sampled_output = pload(model, 'sampled')
        generate_medians_from_sampling_top(sampled_output, test_data.attn_hat, test_data.yt_hat, dirname=model.dirname)
        distractors = get_distractors(sampled_output, test_data.attn_hat)
        print_few_distractors(dataset.vec, test_data.X, test_data.attn_hat, sampled_output, distractors)
    except FileNotFoundError:
        print("Sampling Outputs doesn't exist")

    print("Gradients")
    grads = model.gradient_mem(test_data.X)
    process_grads(grads)
    _ = plot_grads(test_data.X, test_data.attn_hat, grads, test_data.yt_hat, dirname=model.dirname)

    try :
        print("Permutations")
        perms = pload(model, 'permutations')
        _ = plot_permutations(perms, test_data.X, test_data.yt_hat, test_data.attn_hat, dirname=model.dirname)
    except FileNotFoundError:
        print("Permutation Outputs doesn't exist")

    try :
        print("Adversarial")
        multi_adversarial_outputs = pload(model, 'multi_adversarial')
        _ = plot_multi_adversarial(test_data.X, test_data.yt_hat, 
                                                                test_data.attn_hat, multi_adversarial_outputs, dirname=model.dirname)
    except FileNotFoundError :
        print("Multi Adversarial Outputs doesn't exist")

    print("Zero Runs")
    zero_runs = model.zero_H_run(test_data.X)
    zero_outputs, zero_H_diff = zero_runs
    _ = plot_attn_diff(test_data.X, test_data.attn_hat, zero_H_diff, test_data.yt_hat, 
                            xlabel='Attention', ylabel="H(x|c) - H(x)", 
                            title="Attention vs change in hidden state", save_name="hxc-hx", dirname=model.dirname)

    _ = plot_y_diff(test_data.X, test_data.attn_hat, test_data.yt_hat, zero_outputs, 
                        xlabel="Attention", ylabel="p(y|x, c) - p(y|x)", 
                        title="Attention vs change in output", save_name="pyxc-pyx", dirname=model.dirname)

    try :
        print("Remove and Run")
        remove_outputs = pload(model, 'remove_and_run')
        _ = plot_y_diff(test_data.X, test_data.attn_hat, test_data.yt_hat, remove_outputs, 
                        xlabel="Attention", ylabel="p(y|x, c) - p(y|c)", 
                        title="Attention vs change in output", save_name="pyxc-pyc", dirname=model.dirname)
    except FileNotFoundError:
        print("Remove Outputs doesn't exist")

    print("Pushing to Directory")
    push_graphs_to_main_directory(model, dataset.name)

    print("="*30)
    print("="*30)