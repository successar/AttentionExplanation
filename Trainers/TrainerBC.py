from Transparency.common_code.common import *
from Transparency.common_code.metrics import *
import Transparency.model.Binary_Classification as BC

metrics_type = {
    'Single_Label' : calc_metrics_classification, 
    'Multi_Label' : calc_metrics_multilabel
}

class Trainer() :
    def __init__(self, dataset, config, _type="Single_Label") :
        Model = BC.Model
        self.model = Model(config, pre_embed=dataset.vec.embeddings)
        self.metrics = metrics_type[_type]
        self.display_metrics = True
    
    def train(self, train_data, test_data, n_iters=20, save_on_metric='roc_auc') :
        best_metric = 0.0
        for i in tqdm(range(n_iters)) :
            self.model.train(train_data.X, train_data.y)
            predictions, attentions = self.model.evaluate(test_data.X)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.y, predictions)
            if self.display_metrics :
                print_metrics(test_metrics)

            metric = test_metrics[save_on_metric]
            save_model = True
            print("Model Saved on ", save_on_metric, metric)
            
            dirname = self.model.save_values(save_model=save_model)
            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()

class Evaluator() :
    def __init__(self, dataset, dirname, _type='Single_Label') :
        Model = BC.Model
        self.model = Model.init_from_config(dirname)
        self.model.dirname = dirname
        self.metrics = metrics_type[_type]
        self.display_metrics = True

    def evaluate(self, test_data, save_results=False) :
        predictions, attentions = self.model.evaluate(test_data.X)
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

    def permutation_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'permutations') :
            perms = self.model.permute_attn(test_data.X)
            pdump(self.model, perms, 'permutations')

    def adversarial_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'multi_adversarial') :
            multi_adversarial_outputs = self.model.adversarial_multi(test_data.X)
            pdump(self.model, multi_adversarial_outputs, 'multi_adversarial')

    def remove_and_run_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'remove_and_run') :
            remove_outputs = self.model.remove_and_run(test_data.X)
            pdump(self.model, remove_outputs, 'remove_and_run')

    def gradient_experiment(self, test_data, force_run=False) :
        if force_run or not is_pdumped(self.model, 'gradients') :
            grads = self.model.gradient_mem(test_data.X)
            pdump(self.model, grads, 'gradients')

    def logodds_attention_experiment(self, test_data, logodds, save_results=False) :
        logodds_combined = defaultdict(float)
        for e in logodds :
            for k, v in logodds[e].items() :
                if v is not None :
                    logodds_combined[k] += abs(v) / len(logodds.keys())
                else :
                    logodds_combined[k] = None

        predictions, attentions = self.model.logodds_attention(test_data.X, logodds_combined)
        predictions = np.array(predictions)

        pdump(self.model, [predictions, attentions], 'logodds_attention')

        test_metrics = self.metrics(test_data.y, predictions)
        pred_metrics = self.metrics(np.where(test_data.yt_hat < 0.5, 0, 1), predictions)
        if self.display_metrics :
            print_metrics(test_metrics)
            print_metrics(pred_metrics)

        time_str = os.path.split(self.model.dirname)[1]
        model_name = os.path.split(os.path.split(self.model.dirname)[0])[1]
        basename = os.path.split(os.path.split(self.model.dirname)[0])[0]
        dirname = os.path.join(basename, model_name + '+logodds(posthoc)', time_str)
        os.makedirs(dirname, exist_ok=True)
        if save_results :
            f = open(dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()  

        test_data.logodds_combined = logodds_combined      

    def logodds_substitution_experiment(self, test_data, logodds, save_results=False) :
        logodds_combined = defaultdict(dict)
        for e in logodds :
            sorted_e = sorted(logodds[e].items(), key=lambda x : x[1] if x[1] is not None else 0.0)
            logodds_combined[e] = {
                0 : [x[0] for x in sorted_e[:5]], 
                1 : [x[0] for x in sorted_e[-5:]]
            }

        predictions, attentions, new_X = self.model.logodds_substitution(test_data.X, logodds_combined)
        new_X = [x[:len(y)] for x, y in zip(new_X, test_data.X)]
        predictions = np.array(predictions)

        pdump(self.model, [predictions, attentions, new_X], 'logodds_substitution')

        test_metrics = self.metrics(test_data.y, predictions)
        pred_metrics = self.metrics(np.where(test_data.yt_hat < 0.5, 0, 1), predictions)
        if self.display_metrics :
            print_metrics(test_metrics)
            print_metrics(pred_metrics)

        test_data.opp_yt_hat = predictions
        test_data.opp_attn = attentions
        test_data.opp_X = new_X

        # time_str = os.path.split(self.model.dirname)[1]
        # model_name = os.path.split(os.path.split(self.model.dirname)[0])[1]
        # basename = os.path.split(os.path.split(self.model.dirname)[0])[0]
        # dirname = os.path.join(basename, model_name + '+logodds(posthoc)', time_str)
        # os.makedirs(dirname, exist_ok=True)
        # if save_results :
        #     f = open(dirname + '/evaluate.json', 'w')
        #     json.dump(test_metrics, f)
        #     f.close()  

        test_data.logodds_combined = logodds_combined
