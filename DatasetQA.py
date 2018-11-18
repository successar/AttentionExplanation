from common import *

def add_frequencies(vec, X) :
    freq = np.zeros((vec.vocab_size, ))
    for x in X :
        for w in x :
            freq[w] += 1
    freq = freq / np.sum(freq)
    vec.freq = freq

class DataHolder() :
    def __init__(self, **kwargs) :
        for n, v in kwargs.items() :
            setattr(self, n, v)

        lens = [len(x) - 2 for x in self.P]
        self.min_length = min(lens)
        self.max_length = max(lens)
        self.mean_length = np.mean(lens)
        self.std_length = np.std(lens)

def getFromDict(dataDict, maplist):
    first, rest = maplist[0], maplist[1:]
    if rest: return getFromDict(dataDict[first], rest)
    else: return dataDict[first]

def get_data_from_vec(vec, _types, sort=False) :
    get = lambda x : getFromDict(x, _types)
    P, Q, E, A = get(vec.paragraphs), get(vec.questions), get(vec.entity_masks), get(vec.answers)
    if sort :
        sorting_idx = np.argsort([len(x) for x in P])
        sort = lambda l : permute_list(l, sorting_idx)
        
        P, Q, E, A = sort(P), sort(Q), sort(E), sort(A)
        
    data = DataHolder(P=P, Q=Q, E=E, A=A)
    return data

class Dataset() :
    def __init__(self, name, path=None, vec=None, filters=[]) :
        self.name = name
        self.vec = pickle.load(open(path, 'rb')) if vec is None else vec
        self.vec.entity_size = len(self.vec.entity2idx)

        self.train_data = get_data_from_vec(self.vec, ['train'] + filters)
        self.test_data = get_data_from_vec(self.vec, ['test'] + filters, sort=True)

        self.bsize = 100
        self.by_class = False

        add_frequencies(self.vec, self.train_data.P)

    def set_model(self, name) :
        self.model_dirname = name

    def set_hs(self, hidden_size) :
        self.vec.hidden_size = hidden_size
        return self

    def display_stats(self) :
        stats = {}
        stats['vocab_size'] = self.vec.vocab_size
        stats['embed_size'] = self.vec.word_dim
        stats['hidden_size'] = self.vec.hidden_size

        if self.by_class :
            y = np.unique(np.array(self.train_data.A), return_counts=True)
            yt = np.unique(np.array(self.test_data.A), return_counts=True)

            stats['train_size'] = list(zip(y[0].tolist(), y[1].tolist()))
            stats['test_size'] = list(zip(yt[0].tolist(), yt[1].tolist()))
        else :
            stats['train_size'] = [("Overall", len(self.train_data.A))]
            stats['test_size'] = [("Overall", len(self.test_data.A))]

        outdir = "datastats"
        os.makedirs('graph_outputs/' + outdir, exist_ok=True)

        stats['avg_length'] = self.train_data.mean_length
        stats['min_length'] = self.train_data.min_length
        stats['max_length'] = self.train_data.max_length
        stats['std_length'] = self.train_data.std_length

        json.dump(stats, open('graph_outputs/' + outdir + '/' + self.name + '.txt', 'w'))
        print(stats)

def get_SNLI() :
    SNLI_dataset = Dataset(name='snli', path='preprocess/SNLI/vec_snli.p').set_hs(128)
    SNLI_dataset.set_model('outputs/attn_QA_snli/WedNov1406:56:432018_final_1/')
    SNLI_dataset.bsize = 32
    SNLI_dataset.by_class = True
    return SNLI_dataset

def get_CNN() :
    CNN_dataset = Dataset(name='cnn', path='preprocess/CNN/vec_cnn.p').set_hs(128)
    CNN_dataset.set_model('outputs/attn_QA_cnn/TueNov1321:14:112018_final_1/')
    CNN_dataset.bsize = 30
    return CNN_dataset

def get_Babi_1() :
    Babi_1_dataset = Dataset(name='babi_1', path='preprocess/Babi/babi.p', filters=['qa1_single-supporting-fact_']).set_hs(30)
    Babi_1_dataset.set_model('outputs/attn_QA_babi_1/WedNov1416:50:062018_/')
    Babi_1_dataset.vec.word_dim = 50
    return Babi_1_dataset

def get_Babi_2() :
    Babi_2_dataset = Dataset(name='babi_2', path='preprocess/Babi/babi.p', filters=['qa2_two-supporting-facts_']).set_hs(30)
    Babi_2_dataset.set_model('outputs/attn_QA_babi_2/WedNov1416:51:082018_/')
    Babi_2_dataset.vec.word_dim = 50

    return Babi_2_dataset

def get_Babi_3() :
    Babi_3_dataset = Dataset(name='babi_3', path='preprocess/Babi/babi.p', filters=['qa3_three-supporting-facts_']).set_hs(30)
    Babi_3_dataset.set_model('outputs/attn_QA_babi_3/WedNov1416:52:402018_/')
    Babi_3_dataset.vec.word_dim = 50

    return Babi_3_dataset

datagens = (get_SNLI, get_CNN, get_Babi_1, get_Babi_2, get_Babi_3)

import model.Attn_QA_Pert as M
Model = M.Model

from sklearn.metrics import classification_report, accuracy_score

def train(dataset, name='') :
    model = Model(dataset.vec.vocab_size, dataset.vec.word_dim, dataset.vec.entity_size, dataset.bsize, 
                    dirname=dataset.name, 
                    hidden_size=dataset.vec.hidden_size, 
                    pre_embed=dataset.vec.embeddings)

    train_data = dataset.train_data
    test_data = dataset.test_data
    best_acc = 0.0
    for i in tqdm_notebook(range(25)) :
        loss = model.train(train_data.P, train_data.Q, train_data.E, train_data.A)
        stmt = '%s, %s' % (i, loss)
        print(stmt)
        predict, attn = model.evaluate(test_data.P, test_data.Q, test_data.E)
        acc = accuracy_score(test_data.A, predict)
        save = False
        if acc > best_acc :
            save = True
            best_acc = acc
            print("Model Saved!")
            
        dirname = model.save_values(name, save_model=save)
        f = open(dirname + '/epochs.txt', 'a') 
        f.write(str(acc) + " Model Saved : " + str(save) + '\n')
        f.close()
        print(acc)
    
    return model

def load_model(dataset) :
    model = Model(dataset.vec.vocab_size, dataset.vec.word_dim, dataset.vec.entity_size, dataset.bsize, 
                    dirname=dataset.name, 
                    hidden_size=dataset.vec.hidden_size, 
                    pre_embed=dataset.vec.embeddings)
    model.dirname = dataset.model_dirname
    model.load_values(model.dirname)
    return model