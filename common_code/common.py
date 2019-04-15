import os
import pickle
import re
import shutil
import sys
import json
from Transparency.preprocess import vectorizer

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import torch
from IPython.core.display import HTML, display
from tqdm import tqdm_notebook, tqdm

from collections import defaultdict

np.set_printoptions(suppress=True)

def permute_list(l, p) :
    return [l[i] for i in p]

def calc_max_attn(X, attn) : 
    return np.array([max(attn[i][1:len(X[i])-1]) for i in range(len(attn))])

#########################################################################################################

def plot_entropy(X, attn) :
    unif_H, attn_H = [], []
    for i in range(len(X)) :
        L = len(X[i])
        h = attn[i][1:L-1]
        a = h * np.log(np.clip(h, a_min=1e-8, a_max=None))
        a = -a.sum()
        unif_H.append(np.log(L-2))
        attn_H.append(a)

    plt.scatter(unif_H, attn_H, s=1)

def print_attn(sentence, attention, idx=None, latex=False) :
    l = []
    latex_str = []
    for i, (w, a) in enumerate(zip(sentence, attention)) :
        w = re.sub('&', '&amp;', w)
        w = re.sub('<', '&lt;', w)
        w = re.sub('>', '&gt;', w)
        
        add_string = ''
        if idx is not None and i == idx :
            add_string = "border-style : solid;"
        
        if a < 0 : hue = '350'
        else : hue = '202'
        a = abs(a)
        v = "{:.2f}".format((1-a) * -0.5 + 0.5)
        l.append('<span style="background-color:hsl(' + hue + ',100%,' + str((1-a) * 50 + 50) + '%);' + add_string + '">' + w + ' </span>')
        latex_str.append('{\\setlength{\\fboxsep}{0pt}\\colorbox[Hsb]{'+hue+', ' + v + ', 1.0}{\\strut ' + w + '}}')
    
    display(HTML(''.join(l)))
    if latex : 
        return " ".join(latex_str)
    else :
        return ""

def get_word_importance(dataset, sentence, attention) :
    words_importance = defaultdict(float)
    for w, a in zip(sentence, attention) :
        words_importance[dataset.vec.idx2word[w]] += a

    return words_importance

def find_top_words(dataset, sentence, attention, n=20) :
    words_importance = get_word_importance(dataset, sentence, attention)
    top_words = dict(sorted(words_importance.items(), key=lambda x: x[1])[-n:])
    return top_words

def find_top_words_in_all(dataset, sentences, attentions, n=20) :
    X = [find_top_words(dataset, s, a, n) for s, a in zip(sentences, attentions)]
    return X

############################################################################################

def kld(a1, a2) :
    #(B, *, A), #(B, *, A)
    a1 = np.clip(a1, 0, 1)
    a2 = np.clip(a2, 0, 1)
    log_a1 = np.log(a1 + 1e-10)
    log_a2 = np.log(a2 + 1e-10)
    kld_v = a1 * (log_a1 - log_a2)

    return kld_v.sum(-1)

def jsd(p, q) :
    m = 0.5 * (p + q)
    jsd_v = 0.5 * (kld(p, m) + kld(q, m))

    return jsd_v

#############################################################################################

def pdump(model, values, filename) :
    pickle.dump(values, open(os.path.join(model.dirname, filename + '_pdump.pkl'), 'wb'))

def pload(model, filename) :
    file = os.path.join(model.dirname, filename + '_pdump.pkl')
    if not os.path.isfile(file) :
        raise FileNotFoundError(file + " doesn't exist")

    return pickle.load(open(file, 'rb'))

def is_pdumped(model, filename) :
    file = os.path.join(model.dirname, filename + '_pdump.pkl')
    return os.path.isfile(file)

import time

def get_all_models(dirname) :
    dirs = [d for d in os.listdir(dirname) if 'evaluate.json' in os.listdir(os.path.join(dirname, d))]
    if len(dirs) == 0 :
        return None
    return [os.path.join(dirname, d) for d in dirs]

def get_latest_model(dirname) :
    dirs = [d for d in os.listdir(dirname) if 'evaluate.json' in os.listdir(os.path.join(dirname, d))]
    if len(dirs) == 0 :
        return None
    max_dir = max(dirs, key=lambda s : time.strptime(s.replace('_', ' ')))
    return os.path.join(dirname, max_dir)

def push_graphs_to_main_directory(model_dirname, name) :
    dirname = model_dirname
    files = os.listdir(dirname)
    files = [f for f in files if f.endswith('svg')]
    
    for f in files :
        outdir = f[:-4]
        output_name = os.path.join('graph_outputs', outdir)
        os.makedirs(output_name, exist_ok=True)
        shutil.copyfile(os.path.join(model_dirname, f), os.path.join(output_name, name + '.svg'))

    files = os.listdir(dirname)
    files = [f for f in files if f.endswith('csv')]
    
    for f in files :
        outdir = f[:-4]
        output_name = os.path.join('graph_outputs', outdir)
        os.makedirs(output_name, exist_ok=True)
        shutil.copyfile(os.path.join(model_dirname, f), os.path.join(output_name, name + '.csv'))

    files = os.listdir(dirname)
    files = [f for f in files if f.endswith('pdf')]
    
    for f in files :
        outdir = f[:-4]
        output_name = os.path.join('graph_outputs', outdir)
        os.makedirs(output_name, exist_ok=True)
        shutil.copyfile(os.path.join(model_dirname, f), os.path.join(output_name, name + '.pdf'))

    files = os.listdir(dirname)
    files = [f for f in files if f == 'evaluate.json']
    
    for f in files :
        outdir = f[:-5]
        output_name = os.path.join('graph_outputs', outdir)
        os.makedirs(output_name, exist_ok=True)
        shutil.copyfile(os.path.join(model_dirname, f), os.path.join(output_name, name + '.json'))