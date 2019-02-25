from sklearn.metrics import accuracy_score, classification_report
from common import *
from plotting import *

from scipy.stats import kendalltau

import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

#################### Preprocessing Begin ############################################

def evaluate_and_print(model, data) :
    P, Q, E, A = data.P, data.Q, data.E, data.A
    yhat, attn = model.evaluate(P, Q, E)
    yhat = np.array(yhat)
    rep = accuracy_score(A, yhat)
    print("Accuracy Score : ", rep)

    if len(np.unique(np.round(yhat))) < 4 :
        print(classification_report(A, yhat))

    return yhat, attn
            
def print_example(vec, data, predict, attn, n) :
    print(" ".join(vec.map2words(data.Q[n])))
    print("Truth : " , vec.idx2entity[data.A[n]], " | Predicted : ", vec.idx2entity[predict[n]])
    print_attn(sentence=vec.map2words(data.P[n]), attention=attn[n])

###########################################################################################################################

def plot_permutations(permutations, data, yhat, attn, by_class=False, dirname='') :
    ad_y, ad_diffs = permutations
    ad_diffs = 0.5*np.array(ad_diffs)

    median_diff = np.median(ad_diffs, 1)
    
    perms_perc = np.array([(ad_y[i] == yhat[i]).mean() for i in range(len(ad_y))])
    X = data.P

    max_attn = calc_max_attn(X, attn)

    fig, ax = init_gridspec(3, 3, 4)

    plot_scatter_by_class(ax[0], max_attn, perms_perc, yhat, by_class=by_class)
    annotate(ax[0], xlabel=('Max attention'), ylabel=(''))

    plot_scatter_by_class(ax[1], max_attn, median_diff, yhat, by_class=by_class)
    annotate(ax[1], xlabel=('Max attention'), ylabel='$\\Delta\\hat{y}$', xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

    plot_histogram_by_class(ax[2], median_diff, yhat, by_class=by_class, hist_lims=(0, 1))
    annotate(ax[2], xlim=(-0.05, 1.05), left=True, xlabel="Median Output Difference")

    plot_violin_by_class(ax[3], max_attn, median_diff, yhat, xlim=(0, 1.0))
    annotate(ax[3], xlim=(-0.05, 1.05), ylabel="Max attention", xlabel="Median Output Difference", legend=None)

    adjust_gridspec()

    save_axis_in_file(fig, ax[3], dirname, "Permutation_MAvDY")
    save_axis_in_file(fig, ax[2], dirname, 'Permutation_Hist')

    show_gridspec()

##########################################################################################################

def plot_multi_adversarial(data, yhat, attn, adversarial_outputs, epsilon=0.01, by_class=False, dirname='') :
    fig, ax = init_gridspec(3, 3, 2)

    X = data.P

    ad_y, ad_attn, ad_diffs = adversarial_outputs
    ad_y = np.array(ad_y)
    ad_diffs = 0.5 * np.array(ad_diffs)[:, :, 0]

    ad_attn = [np.vstack([ad_attn[i], attn[i]]) for i in range(len(attn))]
    ad_diffs = np.array([np.append(ad_diffs[i], 0) for i in range(len(attn))])
    ad_y = np.array([np.append(ad_y[i], yhat[i]) for i in range(len(attn))])

    K = ad_attn[0].shape[0]

    for k in range(K) :
        print("Accuracy Score : " , accuracy_score(yhat, ad_y[:, k]))

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

    plot_histogram_by_class(ax[0], emax_jds, yhat, hist_lims=(0, 0.7), pval=g30, pvallabel="JSD > 0.34", by_class=by_class)
    annotate(ax[0], xlabel=("Max JS Divergence within $\\epsilon$"), left=True, xlim=(-0.02, 0.72))

    max_attn = calc_max_attn(X, attn)
    plot_violin_by_class(ax[1], max_attn, emax_jds, yhat, by_class=by_class, xlim=(0, 1.0))
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

def process_grads(grads) :
    for k in grads :
        xxe = grads[k]
        for i in range(len(xxe)) :
            xxe[i] = np.abs(xxe[i])
            xxe[i] = xxe[i] / xxe[i].sum()

def plot_grads(data, yhat, attn, grads, by_class=False, dirname='') :
    X = data.P
    fig, ax = init_gridspec(3, 3, len(grads))

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

        pval_tables[k] = plot_SP_histogram_by_class(axes, spcorrs, yhat, by_class=by_class)
        annotate(axes, left=True)

    adjust_gridspec()

    save_axis_in_file(fig, ax[1], dirname, 'GradientXHist')
    save_table_in_file(pval_tables['XxE[X]'], dirname, 'GradientPval')

    show_gridspec()

###########################################################################################################################

def plot_y_diff(data, yhat, attn, y_diff, save_name=None, dirname='') :
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

    fig, ax = init_gridspec(3, 3, 1)

    pval_table = plot_SP_histogram_by_class(ax[0], spcorrs, yhat)
    annotate(ax[0], left=True)

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_SP')
        save_table_in_file(pval_table, dirname, save_name + '_pval')

    show_gridspec()

######################################################################################################################################################

def generate_graphs(dataset, exp_name, model, test_data) :
    logging.info("Generating graph for %s, %s", dataset.name, exp_name)

    logging.info("Generating Gradients Graph ...")
    gradients = model.gradient_mem(test_data.P, test_data.Q, test_data.E)
    process_grads(gradients)
    plot_grads(test_data, test_data.predict, test_data.attn_hat, gradients, by_class=dataset.by_class, dirname=model.dirname)

    try :
        logging.info("Generating Multi Adversarial Graph ...")
        multi_adversarial_outputs = pload(model, 'multi_adversarial')
        _ = plot_multi_adversarial(test_data, test_data.predict, test_data.attn_hat, 
                                                            multi_adversarial_outputs, 
                                                            epsilon=0.05,
                                                            by_class=dataset.by_class, dirname=model.dirname)
    except FileNotFoundError :
        logging.warning("Multi Adversarial Output doesn't exists ...")

    try :
        logging.info("Generating Permutations Graph ...")
        perms = pload(model, 'permutations')
        plot_permutations(perms, test_data, test_data.predict, test_data.attn_hat, by_class=dataset.by_class, dirname=model.dirname)
    except FileNotFoundError :
        logging.warning("Permutation Outputs doesn't exist")

    try :
        logging.info("Generating Remove and Run Graph ...")
        remove_outputs = pload(model, 'remove_and_run')
        plot_y_diff(test_data, test_data.predict, test_data.attn_hat, remove_outputs, save_name="pyxc-pyc", dirname=model.dirname)
    except FileNotFoundError:
        logging.warning("Remove Outputs doesn't exist")

    logging.info("Pushing Graphs to Directory for %s", dataset.name)
    push_graphs_to_main_directory(model, dataset.name+'+'+exp_name)

    print("="*300)