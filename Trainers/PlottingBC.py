from scipy.stats import kendalltau
from Transparency.common_code.kendall_top_k import kendall_top_k
from Transparency.common_code.common import *
from Transparency.common_code.plotting import *
from functools import partial

import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)

#######################################################################################################################

def process_grads(grads) :
    for k in grads :
        xxe = grads[k]
        for i in range(len(xxe)) :
            xxe[i] = np.abs(xxe[i]).sum(0)
            xxe[i] = xxe[i] / xxe[i].sum()

def plot_grads(test_data, gradients, correlation_measure, correlation_measure_name, dirname='') :
    X, yhat, attn = test_data.X, test_data.yt_hat, test_data.attn_hat
    fig, ax = init_gridspec(3, 3, 1)
    pval_tables = {}

    gradlist = gradients['XxE[X]']
    spcorrs_all = []
    
    for i in range(len(X)) :
        L = len(X[i])
        spcorr = correlation_measure(list(attn[i][1:L-1]), list(gradlist[i][1:L-1]))
        spcorrs_all.append(spcorr)

    axes = ax[0]
    pval_tables['XxE[X]'] = plot_SP_histogram_by_class(axes, spcorrs_all, yhat)
    annotate(axes)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, 'GradientXHist_'+correlation_measure_name)
    save_table_in_file(pval_tables['XxE[X]'], dirname, 'GradientPval_'+correlation_measure_name)
    show_gridspec()

def plot_correlation_between_grad_and_loo(test_data, gradients, ynew_list, correlation_measure, correlation_measure_name, dirname='') :
    X, yhat, attn = test_data.X, test_data.yt_hat, test_data.attn_hat
    fig, ax = init_gridspec(3, 3, 2)
    pval_tables = {}

    gradlist = gradients['XxE[X]']
    spcorrs_gl = []
    spcorrs_ag = []
    spcorrs_al = []
    
    for i in range(len(X)) :
        L = len(X[i])
        ydiff = np.abs(ynew_list[i] - yhat[i]).mean(-1)
        spcorr_gl = correlation_measure(list(ydiff[1:L-1]), list(gradlist[i][1:L-1]))
        spcorrs_gl.append(spcorr_gl)

        spcorr_ag = correlation_measure(list(ydiff[1:L-1]), list(attn[i][1:L-1]))
        spcorrs_ag.append(spcorr_ag)

        spcorr_al = correlation_measure(list(attn[i][1:L-1]), list(gradlist[i][1:L-1]))
        spcorrs_al.append(spcorr_al)

    corr_tables = {}

    axes = ax[0]
    corr_tables['ag'] = plot_SP_density_by_class(axes, spcorrs_ag, yhat, linestyle='-').loc['Overall']
    corr_tables['al'] = plot_SP_density_by_class(axes, spcorrs_al, yhat, linestyle='--').loc['Overall']
    corr_tables['gl'] = plot_SP_density_by_class(axes, spcorrs_gl, yhat, linestyle=':').loc['Overall']
    annotate(axes)

    corr_tables = pd.DataFrame(corr_tables).transpose()
    save_table_in_file(corr_tables, dirname, 'CorrStats_'+correlation_measure_name)

    axes = ax[1]
    pval_tables['gl-ag'] = plot_SP_density_by_class(axes, np.array(spcorrs_gl) - np.array(spcorrs_ag), yhat, linestyle='-')
    print(pval_tables['gl-ag'])
    pval_tables['gl-al'] = plot_SP_density_by_class(axes, np.array(spcorrs_gl) - np.array(spcorrs_al), yhat, linestyle='--')
    print(pval_tables['gl-al'])
    annotate(axes)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, 'GradientLOOHist_'+correlation_measure_name)
    save_axis_in_file(fig, ax[1], dirname, 'CorrGL_'+correlation_measure_name)
    save_table_in_file(pval_tables['gl-ag'], dirname, 'CorrGL-AG_'+correlation_measure_name)
    save_table_in_file(pval_tables['gl-al'], dirname, 'CorrGL-AL_'+correlation_measure_name)
    show_gridspec()

###########################################################################################################################

def plot_permutations(test_data, permutations, dirname='') :
    #yhat = (N, O), permutations = (N, P, O)
    X, yhat, attn = test_data.X, test_data.yt_hat, test_data.attn_hat
    med_diff = np.abs(np.array(permutations) - yhat[:, None, :]).mean(-1)
    med_diff = np.median(med_diff, 1)

    max_attn = calc_max_attn(X, attn)
    fig, ax = init_gridspec(3, 3, 1)

    plot_violin_by_class(ax[0], max_attn, med_diff, yhat, xlim=(0, 1.0))
    annotate(ax[0], xlim=(-0.05, 1.05), ylabel="Max attention", xlabel="Median Output Difference", legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "Permutation_MAvDY")

    show_gridspec()

############################################################################################################

def plot_multi_adversarial(test_data, adversarial_outputs, dirname='') :
    X, yhat, attn = test_data.X, test_data.yt_hat, test_data.attn_hat
    N = len(X)
    adv_predictions, ad_attn = adversarial_outputs
    adv_predictions = np.array(adv_predictions) #(B, K, O)
    adv_predictions = np.concatenate([adv_predictions, yhat[:, None, :]], axis=1) #(B, K+1, O)
    delta_y = np.abs((adv_predictions - yhat[:, None, :])).mean(-1)  #(B, K+1, O) 

    ad_attn = [np.vstack([ad_attn[i], attn[i]]) for i in range(N)]
    K = ad_attn[0].shape[0]

    emax_JSD, emax_adv_attn, emax_ad_y = [], [], []

    for i in range(N) :
        L = len(X[i])
        delta_attn = np.array([jsd(attn[i][1:L-1], ad_attn[i][k][1:L-1]) for k in range(K)])
        delta_attn[delta_y[i] > 5e-2] = -100
        pos = np.argmax(delta_attn)
        emax_JSD.append(delta_attn[pos])
        emax_adv_attn.append(ad_attn[i][pos])
        emax_ad_y.append(adv_predictions[i][pos])

    emax_JSD = np.array(emax_JSD)

    fig, ax = init_gridspec(3, 3, 2)

    plot_histogram_by_class(ax[0], emax_JSD, yhat, hist_lims=(0, 0.7))
    annotate(ax[0], xlabel="Max JS Divergence within $\\epsilon$")

    max_attn = calc_max_attn(X, attn)
    plot_violin_by_class(ax[1], max_attn, emax_JSD, yhat, xlim=(0, 1.0))
    annotate(ax[1], xlim=(-0.05, 0.7), ylabel=("Max Attention"), xlabel=("Max JS Divergence within $\\epsilon$"), legend=None)

    adjust_gridspec()
    save_axis_in_file(fig, ax[0], dirname, "eMaxJDS_Hist")
    save_axis_in_file(fig, ax[1], dirname, "eMaxJDS_Scatter")
    show_gridspec()

    return emax_JSD, emax_adv_attn, emax_ad_y

def print_adversarial_examples(dataset, test_data, jds, adv_attn, ad_y, reverse=False, by_class=None, save_name='', dirname='') :
    X, yhat, attn = test_data.X, test_data.yt_hat, test_data.attn_hat
    sorted_jds = np.argsort(jds)
    if reverse :
        sorted_jds = sorted_jds[::-1]
    if by_class is None :
        maxjds = sorted_jds[-10:]
    else :
        if by_class > 0 :
            class_jds = [i for i in sorted_jds if yhat[i][0] > by_class]
        else :
            class_jds = [i for i in sorted_jds if yhat[i][0] < -by_class]
        maxjds = class_jds[-10:]
    
    df = []
    for n in maxjds :
        s1, s2, s3 = print_adversarial_example(dataset.vec.map2words(X[n]), attn[n], adv_attn[n], latex=True)
        print(yhat[n], ad_y[n], test_data.y[n], abs(yhat[n] - ad_y[n]).mean(-1), jds[n])
        df.append({
            'sentence' : " ".join(dataset.vec.map2words(X[n])), 
            'orig' : s1, 
            'adv' : s2, 
            'adv_diff' : s3,
            'orig_y' : list(yhat[n]), 
            'adv_y' : list(ad_y[n]),
            'ydiff' : abs(yhat[n] - ad_y[n]).mean(-1)
        })
        print("="*100)

    save_table_in_file(pd.DataFrame(df), dirname, 'adv_examples_'+save_name)
    

def print_adversarial_example(sentence, attn, attn_new, latex=False) :
    L = len(sentence)
    s1 = print_attn(sentence[1:L-1], attn[1:L-1], latex=latex)
    print('-'*20)
    s2 = print_attn(sentence[1:L-1], attn_new[1:L-1], latex=latex)
    s3 = print_attn(sentence[1:L-1], attn[1:L-1] - attn_new[1:L-1], latex=latex)
    return s1, s2, s3

############################################################################################################

def plot_y_diff(test_data, ynew_list, correlation_measure, correlation_measure_name, save_name=None, dirname='') :
    X, yhat, attn = test_data.X, test_data.yt_hat, test_data.attn_hat
    spcorrs = []

    for i in range(len(attn)) :
        L = len(X[i])
        ydiff = np.abs(ynew_list[i] - yhat[i]).mean(-1)
        spcorrs.append(correlation_measure(attn[i][1:L-1], ydiff[1:L-1]))

    fig, ax = init_gridspec(3, 3, 1)
    pval_table = plot_SP_histogram_by_class(ax[0], spcorrs, yhat)
    annotate(ax[0])

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_SP_'+correlation_measure_name)
        save_table_in_file(pval_table, dirname, save_name + '_pval_'+correlation_measure_name)

    show_gridspec()

def plot_attn_diff(dataset, test_data, diffs, save_name=None, dirname='') :
    X, yhat, attn = test_data.X, test_data.yt_hat, test_data.attn_hat
    ynew_list, attnnew_list = diffs
    y_diff = np.abs(np.array(ynew_list) - np.array(yhat)).mean(-1) #(B, )
    attn_diff = []

    for i in range(len(attn)) :
        L = len(X[i])
        attn_diff.append(jsd(attn[i][1:L-1], attnnew_list[i][1:L-1]))

    fig, ax = init_gridspec(3, 3, 2)
    max_attn = calc_max_attn(X, attn)
    plot_violin_by_class(ax[0], max_attn, attn_diff, yhat, xlim=(0.0, 1.0))
    annotate(ax[0], xlim=(-0.05, 0.7), ylabel="Max Attention", xlabel="JSD (logodds vs normal)", legend=None)
    plot_scatter_by_class(ax[1], attn_diff, y_diff, yhat)
    annotate(ax[1], xlim=(-0.05, 0.7), ylim=(-0.05, 1.05), xlabel="JSD (logodds vs normal)", ylabel='Output Difference', legend=None)

    adjust_gridspec()
    if save_name is not None : 
        save_axis_in_file(fig, ax[0], dirname, save_name + '_Violin')
        save_axis_in_file(fig, ax[1], dirname, save_name + '_Scatter')

    show_gridspec()
    # attn_diff = [x for x in attn_diff]
    # print_adversarial_examples(dataset, test_data, attn_diff, attnnew_list, ynew_list, by_class=None, dirname='.')

    return attn_diff, attnnew_list, ynew_list

##########################################################################################################

def generate_graphs(dataset, exp_name, model, test_data) :
    logging.info("Generating graph for %s", model.dirname)
    average_length = int(np.clip(test_data.get_stats('X')['mean_length'] * 0.1, 10, None))
    logging.info("Average Length of test set %d", average_length)
    kendall_top_k_dataset = partial(kendall_top_k, k=average_length, p=0.5)
    # kendall_top_k_dataset_0 = partial(kendall_top_k, k=average_length, p=0)

    try :
        logging.info("Generating Gradients Graph ...")
        grads = pload(model, 'gradients')
        process_grads(grads)
        plot_grads(test_data, grads, kendalltau, 'kendalltau', dirname=model.dirname)
        # plot_grads(test_data, grads, kendall_top_k_dataset, 'kendalltop', dirname=model.dirname)
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
        plot_y_diff(test_data, remove_outputs, kendalltau, 'kendalltau', save_name="pyxc-pyc", dirname=model.dirname)
        # plot_y_diff(test_data, remove_outputs, kendall_top_k_dataset, 'kendalltop', save_name="pyxc-pyc", dirname=model.dirname)
    except FileNotFoundError:
        logging.warning("Remove Outputs doesn't exist")

    try :
        logging.info("Generating Corr Grad and LOO Graph ...")
        remove_outputs = pload(model, 'remove_and_run')
        plot_correlation_between_grad_and_loo(test_data, grads, remove_outputs, kendalltau, 'kendalltau', dirname=model.dirname)
        plot_correlation_between_grad_and_loo(test_data, grads, remove_outputs, kendall_top_k_dataset, 'kendalltop', dirname=model.dirname)
        # for k in [10, 20, 30, 40, 50, 60, 70, 80, 90] :
        #     kendall_top_k_dataset = partial(kendall_top_k, k=k, p=0.5)
        #     plot_correlation_between_grad_and_loo(test_data, grads, remove_outputs, kendall_top_k_dataset, 'kendalltop_k='+str(k), dirname=model.dirname)
    except FileNotFoundError as e:
        print(e)
        logging.warning("Remove Outputs doesn't exist")

    # try :
    #     logging.info("Generating logodds diff Graph ... ")
    #     logodds_results = pload(model, 'logodds_attention')
    #     plot_attn_diff(dataset, test_data, logodds_results, save_name='logodds_subs', dirname=model.dirname)
    # except FileNotFoundError :
    #     logging.warning("Logodds output doesn't exists ... ")

    logging.info("Pushing Graphs to Directory for %s", model.dirname)
    push_graphs_to_main_directory(model.dirname, exp_name.replace('/', '+'))

    print("="*300)

def plot_adversarial_examples(dataset, exp_name, model, test_data) :
    try :
        logging.info("Generating Multi Adversarial Graph ...")
        multi_adversarial_outputs = pload(model, 'multi_adversarial')
        emax_jds, emax_adv_attn, emax_ad_y = plot_multi_adversarial(test_data, multi_adversarial_outputs, dirname=model.dirname)
        if dataset.name == 'pheno' :
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, by_class=None, save_name='pos', dirname=model.dirname)
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, reverse=True, by_class=None, save_name='pos_rev', dirname=model.dirname)
        else :
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, by_class=0.6, save_name='pos', dirname=model.dirname)
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, by_class=-0.4, save_name='neg', dirname=model.dirname)
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, reverse=True, by_class=0.6, save_name='pos_rev', dirname=model.dirname)
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, reverse=True, by_class=-0.4, save_name='neg_rev', dirname=model.dirname)
    except FileNotFoundError :
        logging.warning("Multi Adversarial Output doesn't exists ...")

def plot_logodds_examples(dataset, exp_name, model, test_data) :
    try :
        logging.info("Generating Log Odds Attentions ...")
        logodds_results = pload(model, 'logodds_attention')
        emax_jds, emax_adv_attn, emax_ad_y = plot_attn_diff(dataset, test_data, logodds_results, save_name='logodds_subs', dirname=model.dirname)
        if dataset.trainer_type == 'Multi_Label' :
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, by_class=None, 
                                        save_name='odds_pos', dirname=model.dirname)
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, reverse=True, by_class=None, 
                                        save_name='odds_pos_rev', dirname=model.dirname)
        else :
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, by_class=0.6, 
                                        save_name='odds_pos', dirname=model.dirname)
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, by_class=-0.4, 
                                        save_name='odds_neg', dirname=model.dirname)
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, reverse=True, by_class=0.6, 
                                        save_name='odds_pos_rev', dirname=model.dirname)
            print_adversarial_examples(dataset, test_data, emax_jds, emax_adv_attn, emax_ad_y, reverse=True, by_class=-0.4, 
                                        save_name='odds_neg_rev', dirname=model.dirname)
    except FileNotFoundError :
        logging.warning("Log Odds Attentions Output doesn't exists ...")
