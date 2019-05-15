import argparse
parser = argparse.ArgumentParser(description='Run Preprocessing on dataset')
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument('--word_vectors_type', type=str, choices=['glove.840B.300d', 'fasttext.simple.300d', 'mimic', 'pubmed'], required=True)
parser.add_argument('--min_df', type=int, required=True)

args, extras = parser.parse_known_args()
args.extras = extras

from Transparency.preprocess import vectorizer
vec = vectorizer.Vectorizer(min_df=args.min_df)

import pandas as pd 

df = pd.read_csv(args.data_file) if args.data_file.endswith('.csv') else pd.read_msgpack(args.data_file)
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

if args.word_vectors_type in ['fasttext.simple.300d', 'glove.840B.300d'] :
    vec.extract_embeddings_from_torchtext(args.word_vectors_type)
elif args.word_vectors_type == 'mimic' :
    from gensim.models import KeyedVectors
    model = KeyedVectors.load("../../MIMIC/mimic_embedding_model.wv")
    vec.extract_embeddings(model)
elif args.word_vectors_type == 'pubmed' :
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('../../../../bigdata/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
    vec.extract_embeddings(model)
else :
    vec.embeddings = None

import pickle, os
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
pickle.dump(vec, open(args.output_file, 'wb'))


