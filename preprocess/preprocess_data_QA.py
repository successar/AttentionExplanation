import argparse
parser = argparse.ArgumentParser(description='Run Preprocessing on dataset')
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument('--all_answers_file', type=str, required=True)
parser.add_argument('--word_vectors_file', type=str, required=True)
parser.add_argument('--min_df', type=float, required=True)

args, extras = parser.parse_known_args()
args.extras = extras

from Transparency.preprocess import vectorizer
vec = vectorizer.Vectorizer(min_df=args.min_df)

import pandas as pd 
import numpy as np

df = pd.read_csv(args.data_file)
assert 'paragraph' in df.columns, "No Paragraph Field"
assert 'question' in df.columns, "No Question Field"
assert 'answer' in df.columns, "No Answer Field"
assert 'exp_split' in df.columns, "No Experimental splits defined"

texts = list(df[df.exp_split == 'train']['paragraph']) + list(df[df.exp_split == 'train']['question'])
vec.fit(texts)

print("Vocabulary size : ", vec.vocab_size)

entities_list = sorted(open(args.all_answers_file).read().strip().split('\n'))
vec.entity2idx = {k:i for i, k in enumerate(entities_list)}
vec.idx2entity = {i:k for k, i in vec.entity2idx.items()}

def generate_label_and_filter(answer, possible_answers=None) :
    entity_mask = np.zeros((len(entities_list), ))
    if possible_answers == None :
        entity_mask = np.ones((len(entities_list), ))
    else :
        for p in possible_answers :
            entity_mask[vec.entity2idx[p]] = 1

    return entity_mask, vec.entity2idx[answer]

vec.paragraphs = {}
vec.questions = {}
vec.entity_masks = {}
vec.answers = {}

splits = df.exp_split.unique()
for k in splits:
    vec.paragraphs[k] = vec.texts_to_sequences(list(df[df.exp_split == k]['paragraph']))
    vec.questions[k] = vec.texts_to_sequences(list(df[df.exp_split == k]['question']))
    vec.entity_masks[k] = []
    vec.answers[k] = []

    answers = list(df[df.exp_split == k]['answer'])
    if 'possible_answers' in df.columns :
        possible_answers = list(df[df.exp_split == k]['possible_answers'])
    else :
        possible_answers = [None] * list(answers)

    for a, p in zip(answers, possible_answers) :
        mask, answer = generate_label_and_filter(a, p)
        vec.entity_masks[k].append(mask)
        vec.answers[k].append(answer)

from gensim.models import KeyedVectors
model = KeyedVectors.load(args.word_vectors_file)

vec.extract_embeddings(model)

import pickle, os
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
pickle.dump(vec, open(args.output_file, 'wb'))