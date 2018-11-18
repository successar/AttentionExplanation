import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from math import ceil

from tqdm import tqdm_notebook

SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<0>'
UNK = '<UNK>'

import spacy, re
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

def cleaner(text, spacy=True) :
    text = re.sub(r'\s+', ' ', text.strip())
    if spacy :
        text = [t.text.lower() for t in nlp(text)]
    else :
        text = [t.lower() for t in text.split()]
    text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]
    return " ".join(text)

class Vectorizer:
    def __init__(self, num_words=None, min_df=None):
        self.embeddings = None
        self.word_dim = 200
        self.num_words = num_words
        self.min_df = min_df

    def process_to_docs(self, texts) :
        docs = [t.replace('\n', ' ').strip() for t in texts]
        return docs 

    def process_to_sentences(self, texts) :
        docs = [t.split('\n') for t in texts]                   
        return docs

    def tokenizer(self, text) :
        return text.split(' ')
    
    def fit(self, texts):
        if self.min_df is not None :    
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, min_df=self.min_df, lowercase=False)
        else :
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, lowercase=False)

        bow = self.cvec.fit_transform(texts)
                
        self.word2idx = self.cvec.vocabulary_
        
        for word in self.cvec.vocabulary_ :
            self.word2idx[word] += 4
            
        self.word2idx[PAD] = 0
        self.word2idx[UNK] = 1
        self.word2idx[SOS] = 2
        self.word2idx[EOS] = 3
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        self.cvec.stop_words_ = None
        
    def add_word(self, word) :
        if word not in self.word2idx :
            idx = max(self.word2idx.values()) + 1
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.vocab_size += 1
        
    def fit_docs(self, texts) :
        docs = self.process_to_docs(texts)
        self.fit(docs)

    def convert_to_sequence(self, texts) :
        texts_tokenized = map(self.tokenizer, texts)
        texts_tokenized = map(lambda s : [SOS] + [UNK if word not in self.word2idx else word for word in s] + [EOS],
                              texts_tokenized)
        texts_tokenized = list(texts_tokenized)
        sequences = map(lambda s : [int(self.word2idx[word]) for word in s], texts_tokenized)
        return list(sequences)

    def texts_to_sequences(self, texts):
        unpad_X = self.convert_to_sequence(texts)
        return unpad_X

    def extract_embeddings(self, model):
        self.word_dim, self.vocab_size = model.vector_size, len(self.word2idx)
        self.embeddings = np.zeros([self.vocab_size, self.word_dim])
        in_pre = 0
        for i, word in sorted(self.idx2word.items()):
            if word in model :
                self.embeddings[i] = model[word] 
                in_pre += 1
            else :
                self.embeddings[i] = np.random.randn(self.word_dim)
                
        self.embeddings[0] = np.zeros(self.word_dim)
                
        print("Found " + str(in_pre) + " words in model out of " + str(len(self.idx2word)))
        return self.embeddings

    def get_seq_for_docs(self, texts) :
        docs = self.process_to_docs(texts) #D
        seq = self.texts_to_sequences(docs) #D x W

        return seq

    def get_seq_for_sents(self, texts) :
        sents = self.process_to_sentences(texts) #(D x S)
        seqs = []
        for d in tqdm_notebook(sents) :
            seqs.append(self.texts_to_sequences(d))

        return seqs
    
    def map2words(self, sent) :
        return [self.idx2word[x] for x in sent]
    
    def map2words_shift(self, sent) :
        return [self.idx2word[x+4] for x in sent]
    
    def map2idxs(self, words) :
        return [self.word2idx[x] if x in self.word2idx else self.word2idx[UNK] for x in sent]