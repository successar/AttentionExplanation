from common import *
from metrics import *
import model.Attn_QA_Multi as M
Model = M.Model

class Trainer() :
    def __init__(self, dataset, config) :
        exp_name = os.path.join(dataset.name, config['exp_dirname'])
        self.model = Model(vocab_size=dataset.vec.vocab_size, 
                            embed_size=dataset.vec.word_dim, 
                            output_size=dataset.vec.entity_size,
                            bsize=config.get('bsize', 64), 
                            dirname=exp_name, 
                            hidden_size=config['hidden_size'], 
                            pre_embed=dataset.vec.embeddings, 
                            weight_decay=config['weight_decay'],
                            attention=config['attention'])
        self.metrics = calc_metrics_classification
        self.display_metrics = True
    
    def train(self, train_data, test_data, n_iters=20, save_on_metric='accuracy') :
        best_metric = 0.0
        for i in tqdm_notebook(range(n_iters)) :
            self.model.train(train_data)
            predictions, attentions = self.model.evaluate(test_data)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.y, predictions)
            if self.display_metrics :
                print_metrics(test_metrics)

            metric = test_metrics[save_on_metric]
            if metric > best_metric and i > 0 :
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)
            else :
                save_model = False
                print("Model not saved on ", save_on_metric, metric)
            
            dirname = self.model.save_values(save_model=save_model)
            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()

class Evaluator() :
    def __init__(self, dataset, dirname) :
        self.model = Model.init_from_config(dirname)
        self.model.dirname = dirname
        self.metrics = calc_metrics_classification
        self.display_metrics = True

    def evaluate(self, test_data, save_results=False) :
        predictions, attentions = self.model.evaluate(test_data)
        predictions = np.array(predictions)

        test_metrics = self.metrics(test_data.y, predictions)
        if self.display_metrics :
            print_metrics(test_metrics)

        if save_results :
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        test_data.yt_hat = predictions
        test_data.attn_hat = attentions
        return predictions, attentions

    def permutation_experiment(self, test_data) :
        perms = self.model.permute_attn(test_data)
        pdump(self.model, perms, 'permutations')

    def adversarial_experiment(self, test_data) :
        multi_adversarial_outputs = self.model.adversarial_multi(test_data)
        pdump(self.model, multi_adversarial_outputs, 'multi_adversarial')

    def remove_and_run_experiment(self, test_data) :
        remove_outputs = self.model.remove_and_run(test_data)
        pdump(self.model, remove_outputs, 'remove_and_run')

    def gradient_experiment(self, test_data) :
        grads = self.model.gradient_mem(test_data)
        pdump(self.model, grads, 'gradients')