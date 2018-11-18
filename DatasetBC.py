from common import *

def sortbylength(X, y) :
    len_t = np.argsort([len(x) for x in X])
    X1 = [X[i] for i in len_t]
    y1 = [y[i] for i in len_t]
    return X1, y1

def add_frequencies(vec, X) :
    freq = np.zeros((vec.vocab_size, ))
    for x in X :
        for w in x :
            freq[w] += 1
    freq = freq / np.sum(freq)
    vec.freq = freq
    
def filterbylength(X, y, min_length = None, max_length = None) :
    lens = [len(x)-2 for x in X]
    min_l = min(lens) if min_length is None else min_length
    max_l = max(lens) if max_length is None else max_length

    idx = [i for i in range(len(X)) if len(X[i]) > min_l+2 and len(X[i]) < max_l+2]
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    return X, y

class DataHolder() :
    def __init__(self, X, y) :
        self.X = X
        self.y = y

        lens = [len(x) - 2 for x in X]
        self.min_length = min(lens)
        self.max_length = max(lens)
        self.mean_length = np.mean(lens)
        self.std_length = np.std(lens)

class Dataset() :
    def __init__(self, name=None, path=None, vec=None, min_length=None, max_length=None) :
        self.name = name
        self.vec = pickle.load(open(path, 'rb')) if vec is None else vec
        add_frequencies(self.vec, self.vec.seq_text['train'])

        X, Xt = self.vec.seq_text['train'], self.vec.seq_text['test']
        y, yt = self.vec.label['train'], self.vec.label['test']

        X, y = filterbylength(X, y, min_length=min_length, max_length=max_length)
        Xt, yt = filterbylength(Xt, yt, min_length=min_length, max_length=max_length)
        Xt, yt = sortbylength(Xt, yt)

        self.train_data = DataHolder(X, y)
        self.test_data = DataHolder(Xt, yt)

        self.vec.hidden_size = 128
        self.weight_decay = 1e-5

    def set_model(self, name) :
        self.model_dirname = name

    def display_stats(self) :
        stats = {}
        stats['vocab_size'] = self.vec.vocab_size
        stats['embed_size'] = self.vec.word_dim
        stats['hidden_size'] = self.vec.hidden_size
        y = np.unique(np.array(self.train_data.y), return_counts=True)
        yt = np.unique(np.array(self.test_data.y), return_counts=True)

        stats['train_size'] = list(zip(y[0].tolist(), y[1].tolist()))
        stats['test_size'] = list(zip(yt[0].tolist(), yt[1].tolist()))

        stats['avg_length'] = self.train_data.mean_length
        stats['min_length'] = self.train_data.min_length
        stats['max_length'] = self.train_data.max_length
        stats['std_length'] = self.train_data.std_length

        outdir = "datastats"
        os.makedirs('graph_outputs/' + outdir, exist_ok=True)

        json.dump(stats, open('graph_outputs/' + outdir + '/' + self.name + '.txt', 'w'))
        print(stats)

def SST_dataset() :
    SST_dataset = Dataset(name='sst', path='preprocess/SST/sst.p', min_length=5)
    SST_dataset.set_model('outputs/attn_word_sst/SatOct2013:00:382018_first_final_sst/')
    return SST_dataset

def IMDB_dataset() :
    IMDB_dataset = Dataset(name='imdb', path='preprocess/IMDB/imdb_data.p', min_length=6)
    IMDB_dataset.set_model('outputs/attn_word_imdb/SatOct2012:40:132018_first_final_imdb/')
    return IMDB_dataset

def News20_dataset() :
    News20_dataset = Dataset(name='20News_sports', path='preprocess/20News/vec_sports.p', min_length=6, max_length=500)
    News20_dataset.set_model('outputs/attn_word_20News_sports/SatOct2011:46:412018_first_final_sports/')
    return News20_dataset

def ADR_dataset() :
    ADR_dataset = Dataset(name='tweet', path='preprocess/Tweets/vec_adr.p', min_length=5, max_length=100)
    ADR_dataset.set_model('outputs/attn_word_tweet/FriOct1918:56:102018_first_final_tweet_adr/')
    return ADR_dataset

def Anemia_dataset() :
    Anemia_dataset = Dataset(name='anemia', path='preprocess/MIMIC/vec_icd9_anemia.p', max_length=4000)
    Anemia_dataset.set_model('outputs/attn_word_anemia/FriOct2616:40:482018_third_final_anemia/')
    return Anemia_dataset

def generate_diabetes() :
    vec = pickle.load(open('preprocess/MIMIC/vec_icd9.p', 'rb'))
    diabetes_label = vec.label2idx['250.00']
    X, Xt = vec.seqs['train'], vec.seqs['test']
    y, yt = vec.label_one_hot['train'][:, diabetes_label], vec.label_one_hot['test'][:, diabetes_label]

    vec.seq_text = {'train' : X, 'test' : Xt}
    vec.label = {'train' : y, 'test' : yt}

    vec.seqs = None
    vec.label_one_hot = None

    return Dataset(name='diab', path=None, vec=vec, min_length=6, max_length=4000)

def Diabetes_dataset() :
    Diabetes_dataset = generate_diabetes()
    Diabetes_dataset.set_model('outputs/attn_word_diab/FriOct2618:41:412018_second_final_diab/')
    return Diabetes_dataset

def AGNews_dataset() :
    AGNews_dataset = Dataset(name='agnews', path='preprocess/ag_news/vec.p')
    AGNews_dataset.set_model('outputs/attn_word_agnews/SatNov1714:18:152018_first_final/')
    return AGNews_dataset

datasets = {
    "sst" : SST_dataset,
    "imdb" : IMDB_dataset,
    "20News_sports" : News20_dataset,
    "tweet" : ADR_dataset ,
    "Anemia" : Anemia_dataset,
    "Diabetes" : Diabetes_dataset,
    "AgNews" : AGNews_dataset
}

from sklearn.metrics import classification_report, f1_score
import model.Attn_Word_Pert as M
Model = M.Model

def train(dataset, name='', exp_dirname='') :
    train_data = dataset.train_data
    test_data = dataset.test_data

    model = Model(dataset.vec.vocab_size, dataset.vec.word_dim, 64, 
                    dirname=dataset.name, 
                    hidden_size=dataset.vec.hidden_size, 
                    pre_embed=dataset.vec.embeddings, 
                    weight_decay=dataset.weight_decay)
    best_f1 = 0.0
    for i in tqdm_notebook(range(10)) :
        loss = model.train(train_data.X, train_data.y)
        o, he = model.evaluate(test_data.X)
        o = np.array(o)
        rep = classification_report(test_data.y, (o > 0.5))
        f1 = f1_score(test_data.y, (o > 0.5), pos_label=1)
        print(rep)
        stmt = '%s, %s' % (i, loss)
        if f1 > best_f1 and i > 2 :
            best_f1 = f1
            dirname = model.save_values(add_name=name, add_dirname=exp_dirname, save_model=True)
            print("Model Saved", f1)
        else :
            dirname = model.save_values(add_name=name, add_dirname=exp_dirname, save_model=False)
            print("Model not saved", f1)
        f = open(dirname + '/epoch.txt', 'a')
        f.write(stmt + '\n')
        f.write(rep + '\n')
        f.close()
    
    return model

def load_model(dataset) :
    model = Model(dataset.vec.vocab_size, dataset.vec.word_dim, 64, dirname=dataset.name, hidden_size=dataset.vec.hidden_size, pre_embed=dataset.vec.embeddings)
    model.dirname = dataset.model_dirname
    model.load_values(model.dirname)
    return model