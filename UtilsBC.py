from scipy.stats import kendalltau
from sklearn.metrics import classification_report

from common import *
from plotting import *

import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

#################### Preprocessing Begin ############################################

def evaluate_and_print(model, data, log=True) :
    X, y = data.X, data.y
    yhat, attn = model.evaluate(X)
    yhat = np.array(yhat)[:, 0]
    rep = classification_report(y, (yhat > 0.5))
    if log : print(rep)

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
    fig, ax = init_gridspec(3, 3, len(grads))
    yhat = yhat.flatten()
    colnum = 0

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

        axes = ax[colnum]
        colnum += 1

        pval_tables[k] = plot_SP_histogram_by_class(axes, spcorrs, yhat)
        annotate(axes, left=True)

    adjust_gridspec()
    save_axis_in_file(fig, ax[1], dirname, 'GradientXHist')
    save_table_in_file(pval_tables['XxE[X]'], dirname, 'GradientPval')

    show_gridspec()

###########################################################################################################################

def plot_permutations(permutations, X, yhat, attn, dirname='') :
    yhat = yhat.flatten()
    med_diff = np.abs(np.array(permutations) - yhat[:, None])
    med_diff = np.median(med_diff, 1)

    max_attn = np.array([max(attn[i][1:len(X[i])-1]) for i in range(len(attn))])
    fig, ax = init_gridspec(3, 3, 1)

    plot_violin_by_class(ax[0], max_attn, med_diff, yhat, xlim=(0, 1.0))
    annotate(ax[0], xlim=(-0.05, 1.05), ylabel="Max attention", xlabel="Median Output Difference", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "Permutation_MAvDY")

    show_gridspec()

############################################################################################################

def plot_multi_adversarial(X, yhat, attn, adversarial_outputs, dirname='') :
    yhat = yhat.flatten()
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

    fig, ax = init_gridspec(3, 3, 2)

    plot_histogram_by_class(ax[0], emax_jds, yhat, hist_lims=(0, 0.7), pval=g30, pvallabel="JSD > 0.34")
    annotate(ax[0], xlabel="Max JS Divergence within $\\epsilon$", left=True)

    max_attn = calc_max_attn(X, attn)
    plot_violin_by_class(ax[1], max_attn, emax_jds, yhat, xlim=(0, 1.0))
    annotate(ax[1], xlim=(-0.05, 0.7), ylabel=("Max Attention"), xlabel=("Max JS Divergence within $\\epsilon$"), legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "eMaxJDS_Hist")
    save_axis_in_file(fig, ax[1], dirname, "eMaxJDS_Scatter")

    show_gridspec()

    return emax_jds, emax_adv_attn, emax_ad_y

def print_adversarial_examples(dataset, X, yt_hat, attn, jds, adv_attn, ad_y, dirname='') :
    maxjds = np.argsort(jds)[-5:]
    df = []
    for n in maxjds :
        s1, s2 = print_adversarial_example(dataset.vec.map2words(X[n]), attn[n], adv_attn[n], latex=True)
        print(yt_hat[n], ad_y[n])
        df.append({'sentence' : " ".join(dataset.vec.map2words(X[n])), 'orig' : s1, 'adv' : s2, 'ydiff' : abs(yt_hat[n] - ad_y[n])})

    save_table_in_file(pd.DataFrame(df), dirname, 'adv_examples')

def print_adversarial_example(sentence, attn, attn_new, latex=False) :
    L = len(sentence)
    s1 = print_attn(sentence[1:L-1], attn[1:L-1], latex=latex)
    print('-'*20)
    s2 = print_attn(sentence[1:L-1], attn_new[1:L-1], latex=latex)
    return s1, s2

############################################################################################################

def plot_y_diff(X, attn, yhat, ynew_list, title="Attention vs change in output", 
                    xlabel="Attention", ylabel="$\\bigtriangleup y$", save_name=None, dirname='') :
    yhat = yhat.flatten()
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

    fig, ax = init_gridspec(3, 3, 1)

    pval_table = plot_SP_histogram_by_class(ax[0], spcorrs, yhat)
    annotate(ax[0], left=True)

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_SP')
        save_table_in_file(pval_table, dirname, save_name + '_pval')

    show_gridspec()

##########################################################################################################

def compare_gradient_and_loo(test_data, model) :
    logging.info("Gradients")
    grads = model.gradient_mem(test_data.X)
    process_grads(grads)
    grad_corrs = plot_grads(test_data.X, test_data.attn_hat, grads, test_data.yt_hat, dirname=model.dirname)

    logging.info("Remove and Run")
    remove_outputs = pload(model, 'remove_and_run')
    loo_corrs = plot_y_diff(test_data.X, test_data.attn_hat, test_data.yt_hat, remove_outputs, 
                    xlabel="Attention", ylabel="p(y|x, c) - p(y|c)", 
                    title="Attention vs change in output", save_name="pyxc-pyc", dirname=model.dirname)

    plot_scatter_by_class(plt.gca(), np.array(grad_corrs)[:, 0], np.array(loo_corrs)[:, 0], test_data.yt_hat)
    annotate(plt.gca(), xlim=(-1, 1), ylim=(-1, 1))
    plt.show()
    logging.info(np.corrcoef(np.array(grad_corrs)[:, 0], np.array(loo_corrs)[:, 0]))

    print("="*30)
    display(HTML("<hr/>"))

def generate_graphs(dataset, exp_name, model, test_data) :
    logging.info("Generating graph for %s", model.dirname)

    logging.info("Generating Gradients Graph ...")
    grads = pload(model, 'gradients')
    process_grads(grads)
    _ = plot_grads(test_data.X, test_data.attn_hat, grads, test_data.yt_hat, dirname=model.dirname)

    try :
        logging.info("Generating Permutations Graph ...")
        perms = pload(model, 'permutations')
        _ = plot_permutations(perms, test_data.X, test_data.yt_hat, test_data.attn_hat, dirname=model.dirname)
    except FileNotFoundError:
        logging.warning("Permutation Outputs doesn't exist")

    try :
        logging.info("Generating Multi Adversarial Graph ...")
        multi_adversarial_outputs = pload(model, 'multi_adversarial')
        _ = plot_multi_adversarial(test_data.X, test_data.yt_hat, test_data.attn_hat, multi_adversarial_outputs, dirname=model.dirname)
    except FileNotFoundError :
        logging.warning("Multi Adversarial Output doesn't exists ...")

    try :
        logging.info("Generating Remove and Run Graph ...")
        remove_outputs = pload(model, 'remove_and_run')
        _ = plot_y_diff(test_data.X, test_data.attn_hat, test_data.yt_hat, remove_outputs, 
                        xlabel="Attention", ylabel="p(y|x, c) - p(y|c)", 
                        title="Attention vs change in output", save_name="pyxc-pyc", dirname=model.dirname)
    except FileNotFoundError:
        logging.warning("Remove Outputs doesn't exist")

    logging.info("Pushing Graphs to Directory for %s", model.dirname)
    push_graphs_to_main_directory(model.dirname, dataset.name+'+'+exp_name)

    print("="*300)