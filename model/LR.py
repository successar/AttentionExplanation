from Transparency.preprocess.vectorizer import BoWder
from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputClassifier
from Transparency.common_code.metrics import *
import time, json
import os, pickle

def normalise_output(y) :
    if y.shape[0] == 1 :
        return y[0]
    return y[:, :, 1].T

class LR :
    def __init__(self, config) :
        vocab = config['vocab']
        stop_words = config.get('stop_words', False)
        self.metrics = metrics_map[config['type']]

        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = config['exp_name']

        self.bowder = BoWder(vocab=vocab, stop_words=stop_words)
        self.tf_idf_classifier = MultiOutputClassifier(LogisticRegression(class_weight='balanced', penalty='l1'), n_jobs=8)

        gen_dirname = lambda x : os.path.join('outputs/', self.exp_name, x, self.time_str)
        self.tf_dirname = gen_dirname('LR+TFIDF')

    def train(self, train_data) :
        docs = train_data.X
        # self.bowder.fit_tfidf(docs)
        train_tf = self.bowder.get_bow(docs)

        self.tf_idf_classifier.fit(train_tf, train_data.y)
        print("Fit TFIDF Classifier ...")

        self.estimator_logodds_map = {}
        for e in range(len(self.tf_idf_classifier.estimators_)) :
            map_back_logodds = {}
            coefs = self.tf_idf_classifier.estimators_[e].coef_[0][:len(self.bowder.words_to_keep)] 
            for k, v in self.bowder.vocab.word2idx.items() :
                if v in self.bowder.map_vocab_to_bow :
                    map_back_logodds[v] = coefs[self.bowder.map_vocab_to_bow[v]]
                else :
                    map_back_logodds[v] = None
            
            self.estimator_logodds_map[e] = map_back_logodds

    def save_estimator_logodds(self) :
        pickle.dump(self.estimator_logodds_map, open(os.path.join(self.tf_dirname, 'logodds.p'), 'wb'))
        
    def evaluate_classifier(self, name, classifier, X, y, dirname, save_results) :
        pred = normalise_output(np.array(classifier.predict_proba(X)))
        metrics = self.metrics(y, pred)
        print(name)
        print_metrics(metrics)
        if save_results :
            os.makedirs(dirname, exist_ok=True)
            f = open(dirname + '/evaluate.json', 'w')
            json.dump(metrics, f)
            f.close()

    def evaluate(self, data, save_results=False) :
        docs = data.X
        tf = self.bowder.get_bow(docs)
        self.evaluate_classifier('TFIDF', self.tf_idf_classifier, tf, data.y, self.tf_dirname, save_results)

    def get_features(self, estimator=0, n=100) :
        return [self.bowder.vocab.idx2word[self.bowder.map_bow_to_vocab[x]] for x in 
                    np.argsort(self.tf_idf_classifier.estimators_[estimator].coef_[0][:len(self.bowder.words_to_keep)])[-n:]]

    def print_all_features(self, n=20) :
        for i in range(len(self.tf_idf_classifier.estimators_)) :
            print(" ".join(self.get_features(estimator=i, n=n)))
            print('-'*10)
            print('='*25)