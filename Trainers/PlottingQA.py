from Transparency.common_code.common import *
from Transparency.common_code.plotting import *

from scipy.stats import kendalltau

import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)
            
def print_example(vec, data, predict, attn, n) :
    print(" ".join(vec.map2words(data.Q[n])))
    print("Truth : " , vec.idx2entity[data.A[n]], " | Predicted : ", vec.idx2entity[predict[n]])
    print_attn(sentence=vec.map2words(data.P[n]), attention=attn[n])

###########################################################################################################################

def process_grads(grads) :
    for k in grads :
        xxe = grads[k]
        for i in range(len(xxe)) :
            xxe[i] = np.abs(xxe[i])
            xxe[i] = xxe[i] / xxe[i].sum()

def plot_grads(test_data, gradients, dirname='') :
    X, attn, yhat = test_data.P, test_data.attn_hat, test_data.yt_hat
    fig, ax = init_gridspec(3, 3, len(gradients)-1)

    pval_tables = {}

    for col, k in enumerate(gradients.keys()) :
        if k == 'H' : continue
        gradlist = gradients[k]    
        spcorrs = []
        
        for i in range(len(X)) :
            L = len(X[i])
            spcorr = kendalltau(list(attn[i][1:L-1]), list(gradlist[i][1:L-1]))
            spcorrs.append(spcorr)

        axes = ax[col]
        pval_tables[k] = plot_SP_histogram_by_class(axes, spcorrs, yhat)
        annotate(axes)

    adjust_gridspec()
    save_axis_in_file(fig, ax[1], dirname, 'GradientXHist')
    save_table_in_file(pval_tables['XxE[X]'], dirname, 'GradientPval')
    show_gridspec()

def plot_correlation_between_grad_and_loo(test_data, gradients, ydiffs) :
    X, yhat, attn = test_data.P, test_data.yt_hat, test_data.attn_hat
    fig, ax = init_gridspec(3, 3, 1)
    pval_tables = {}

    gradlist = gradients['XxE[X]']
    spcorrs_gl = []
    spcorrs_ag = []
    spcorrs_al = []
    
    for i in range(len(X)) :
        L = len(X[i])
        ydiff = ydiffs[i]
        spcorr_gl = kendalltau(list(ydiff[1:L-1]), list(gradlist[i][1:L-1]))
        spcorrs_gl.append(spcorr_gl)

        spcorr_ag = kendalltau(list(ydiff[1:L-1]), list(attn[i][1:L-1]))
        spcorrs_ag.append(spcorr_ag)

        spcorr_al = kendalltau(list(attn[i][1:L-1]), list(gradlist[i][1:L-1]))
        spcorrs_al.append(spcorr_al)


    axes = ax[0]
    pval_tables['XxE[X]'] = plot_SP_density_by_class(axes, spcorrs_ag, yhat)
    pval_tables['XxE[X]'] = plot_SP_density_by_class(axes, spcorrs_al, yhat, linestyle='--')
    pval_tables['XxE[X]'] = plot_SP_density_by_class(axes, spcorrs_gl, yhat, linestyle=':')
    annotate(axes)

    adjust_gridspec()
    # save_axis_in_file(fig, ax[1], dirname, 'GradientXHist')
    # save_table_in_file(pval_tables['XxE[X]'], dirname, 'GradientPval')
    show_gridspec()

###########################################################################################################################

def plot_permutations(test_data, permutations, dirname='') :
    X, attn, yhat = test_data.P, test_data.attn_hat, test_data.yt_hat
    ad_y, ad_diffs = permutations
    ad_diffs = 0.5*np.array(ad_diffs)

    med_diff = np.median(ad_diffs, 1)
    max_attn = calc_max_attn(X, attn)
    fig, ax = init_gridspec(3, 3, 1)

    plot_violin_by_class(ax[0], max_attn, med_diff, yhat, xlim=(0, 1.0))
    annotate(ax[0], xlim=(-0.05, 1.05), ylabel="Max attention", xlabel="Median Output Difference", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "Permutation_MAvDY")
    show_gridspec()

##########################################################################################################

def plot_multi_adversarial(test_data, adversarial_outputs, epsilon=0.05, dirname='') :
    X, attn, yhat = test_data.P, test_data.attn_hat, test_data.yt_hat
    fig, ax = init_gridspec(3, 3, 2)

    ad_y, ad_attn, ad_diffs = adversarial_outputs
    ad_y = np.array(ad_y)
    ad_diffs = 0.5 * np.array(ad_diffs)[:, :, 0]

    ad_attn = [np.vstack([ad_attn[i], attn[i]]) for i in range(len(attn))]
    ad_diffs = np.array([np.append(ad_diffs[i], 0) for i in range(len(attn))])
    ad_y = np.array([np.append(ad_y[i], yhat[i]) for i in range(len(attn))])

    K = ad_attn[0].shape[0]

    emax_jds, emax_adv_attn, emax_ad_y, emax_diff = [], [], [], []

    for i in range(len(X)) :
        L = len(X[i])
        multi_jds_from_true = np.array([jsd(attn[i][1:L-1], ad_attn[i][k][1:L-1]) for k in range(K)])
        multi_jds_from_true[ad_diffs[i] > epsilon] = -1
        pos = np.argmax(multi_jds_from_true)
        emax_jds.append(multi_jds_from_true[pos])
        emax_adv_attn.append(ad_attn[i][pos])
        emax_ad_y.append(ad_y[i][pos])
        emax_diff.append(ad_diffs[i][pos])

    emax_jds = np.array(emax_jds)
    emax_diff = np.array(emax_diff)

    plot_histogram_by_class(ax[0], emax_jds, yhat, hist_lims=(0, 0.7))
    annotate(ax[0], xlabel=("Max JS Divergence within $\\epsilon$"))

    max_attn = calc_max_attn(X, attn)
    plot_violin_by_class(ax[1], max_attn, emax_jds, yhat, xlim=(0, 1.0))
    annotate(ax[1], xlim=(-0.05, 0.7), xlabel=("Max Attention"), ylabel=("Max JS Divergence within $\\epsilon$"), legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "eMaxJDS_Hist")
    save_axis_in_file(fig, ax[1], dirname, "eMaxJDS_Scatter")
    show_gridspec()

    return emax_jds, emax_adv_attn, emax_ad_y, emax_diff

def print_adversarial_examples(dataset, test_data, yt_hat, attn, jds, adv_attn, ad_y, ad_diffs, dirname='') :
    maxjds = np.argsort(jds)[-5:]
    X = test_data.P
    Q = test_data.Q
    df = []
    for n in maxjds :
        print("Question", " ".join(dataset.vec.map2words(Q[n])))
        s1, s2 = print_adversarial_example(dataset.vec.map2words(X[n]), attn[n], adv_attn[n], latex=True)
        print(dataset.vec.idx2entity[yt_hat[n]], dataset.vec.idx2entity[ad_y[n]], ad_diffs[n])
        df.append({
            'sentence' : " ".join(dataset.vec.map2words(X[n])), 
            'question' : " ".join(dataset.vec.map2words(Q[n])),
            'orig' : s1, 
            'adv' : s2, 
            'ydiff' : ad_diffs[n],
            'yold' : dataset.vec.idx2entity[yt_hat[n]],
            'ynew' : dataset.vec.idx2entity[ad_y[n]]
        })

    save_table_in_file(pd.DataFrame(df), dirname, 'adv_examples')

def print_adversarial_example(sentence, attn, attn_new, latex=False) :
    L = len(sentence)
    s1 = print_attn(sentence[1:L-1], attn[1:L-1], latex=latex)
    print('-'*20)
    s2 = print_attn(sentence[1:L-1], attn_new[1:L-1], latex=latex)
    return s1, s2

###########################################################################################################################

def plot_y_diff(test_data, y_diff, save_name=None, dirname='') :
    X, attn, yhat = test_data.P, test_data.attn_hat, test_data.yt_hat

    spcorrs = []

    for i in range(len(attn)) :
        L = len(X[i])
        spcorrs.append(kendalltau(attn[i][1:L-1], y_diff[i][1:L-1]))

    fig, ax = init_gridspec(3, 3, 1)
    pval_table = plot_SP_histogram_by_class(ax[0], spcorrs, yhat)
    annotate(ax[0])

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_SP')
        save_table_in_file(pval_table, dirname, save_name + '_pval')

    show_gridspec()

######################################################################################################################################################

def generate_graphs(dataset, exp_name, model, test_data) :
    logging.info("Generating graph for %s", model.dirname)

    try :
        logging.info("Generating Gradients Graph ...")
        grads = pload(model, 'gradients')
        process_grads(grads)
        plot_grads(test_data, grads, dirname=model.dirname)
    except FileNotFoundError :
        logging.warning("Gradient don't exist ...")

    try :
        logging.info("Generating Permutations Graph ...")
        perms = pload(model, 'permutations')
        plot_permutations(test_data, perms, dirname=model.dirname)
    except FileNotFoundError:
        logging.warning("Permutation Outputs doesn't exist")

    try :
        logging.info("Generating Multi Adversarial Graph ...")
        multi_adversarial_outputs = pload(model, 'multi_adversarial')
        _ = plot_multi_adversarial(test_data, multi_adversarial_outputs, dirname=model.dirname)
    except FileNotFoundError :
        logging.warning("Multi Adversarial Output doesn't exists ...")

    try :
        logging.info("Generating Remove and Run Graph ...")
        remove_outputs = pload(model, 'remove_and_run')
        plot_y_diff(test_data, remove_outputs, save_name="pyxc-pyc", dirname=model.dirname)
        logging.info("Generating corr Grad and LOO")
        plot_correlation_between_grad_and_loo(test_data, grads, remove_outputs)
    except FileNotFoundError:
        logging.warning("Remove Outputs doesn't exist")

    logging.info("Pushing Graphs to Directory for %s", model.dirname)
    push_graphs_to_main_directory(model.dirname, exp_name.replace('/', '+'))

    print("="*300)