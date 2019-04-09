import argparse
parser = argparse.ArgumentParser(description='Run Preprocessing on dataset')
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument('--word_vectors_file', type=str, required=True)
parser.add_argument('--min_df', type=int, required=True)

args, extras = parser.parse_known_args()
args.extras = extras

from Transparency.preprocess import vectorizer
vec = vectorizer.Vectorizer(min_df=args.min_df)

import pandas as pd 

df = pd.read_csv(args.data_file)
assert 'text' in df.columns, "No Text Field"
assert 'label' in df.columns, "No Label Field"
assert 'exp_split' in df.columns, "No Experimental splits defined"

texts = list(df[df.exp_split == 'train']['text'])
vec.fit(texts)

print("Vocabulary size : ", vec.vocab_size)

vec.seq_text = {}
vec.label = {}
splits = df.exp_split.unique()
for k in splits :
    split_texts = list(df[df.exp_split == k]['text'])
    vec.seq_text[k] = vec.get_seq_for_docs(split_texts)
    vec.label[k] = list(df[df.exp_split == k]['label'])

# from gensim.models import KeyedVectors
# model = KeyedVectors.load(args.word_vectors_file)

# vec.extract_embeddings(model)

from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
vectors = Vectors('wiki.simple.vec', url=url)
# In [10]:
vec.word_dim = vectors.dim
# In [11]:
import numpy as np
vec.embeddings = np.zeros((len(vec.idx2word), vec.word_dim))
# In [12]:
for i, word in vec.idx2word.items() :
    vec.embeddings[i] = vectors[word].numpy()

import pickle, os
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
pickle.dump(vec, open(args.output_file, 'wb'))


