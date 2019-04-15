import json
import os
import shutil
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from tqdm import tqdm
from allennlp.common import Params

from .modelUtils import isTrue, get_sorting_index_with_noise_from_lengths
from .modelUtils import BatchHolder, BatchMultiHolder

from Transparency.model.modules.Decoder import AttnDecoderQA
from Transparency.model.modules.Encoder import Encoder

from .modelUtils import jsd as js_divergence

file_name = os.path.abspath(__file__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AdversaryMulti(nn.Module) :
    def __init__(self, decoder=None) :
        super().__init__()
        self.decoder = decoder
        self.K = 5

    def forward(self, data) :
        data.P.hidden_volatile = data.P.hidden.detach()
        data.Q.last_hidden_volatile = data.Q.last_hidden.detach()

        new_attn = torch.log(data.P.generate_uniform_attn()).unsqueeze(1).repeat(1, self.K, 1) #(B, 10, L)
        new_attn = new_attn + torch.randn(new_attn.size()).to(device)*3

        new_attn.requires_grad = True
        
        data.log_attn_volatile = new_attn 
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01, amsgrad=True)
        data.multiattention = True

        for _ in range(500) :
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.P.masks.unsqueeze(1), -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn) #(B, 10, L)
            self.decoder.get_output(data)
            
            predict_new = data.predict_volatile #(B, *, O)
            y_diff = self.output_diff(predict_new, data.predict.detach().unsqueeze(1))
            diff = nn.ReLU()(y_diff - 1e-2) #(B, *, 1)

            jsd = js_divergence(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, *, 1)

            cross_jsd = js_divergence(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))

            loss =  -(jsd**1) + 500 * diff #(B, *, 1)
            loss = loss.sum() - cross_jsd.sum(0).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()


        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.P.masks.unsqueeze(1), -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)

    def output_diff(self, p, q) :
        #p : (B, *, O)
        #q : (B, *, O)
        softmax = nn.Softmax(dim=-1)
        y_diff = torch.abs(softmax(p) - softmax(q)).sum(-1).unsqueeze(-1) #(B, *, 1)

        return y_diff

class Model() :
    def __init__(self, configuration, pre_embed=None) :
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed

        encoder_copy = deepcopy(configuration['model']['encoder'])
        self.Pencoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)
        self.Qencoder = Encoder.from_params(Params(encoder_copy)).to(device)

        configuration['model']['decoder']['hidden_size'] = self.Pencoder.output_size
        self.decoder = AttnDecoderQA.from_params(Params(configuration['model']['decoder'])).to(device)

        self.bsize = configuration['training']['bsize']

        self.adversary_multi = AdversaryMulti(self.decoder)

        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.params, weight_decay=weight_decay, amsgrad=True)
        # self.optim = torch.optim.Adagrad(self.params, lr=0.05, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        import time
        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)

    @classmethod
    def init_from_config(cls, dirname, **kwargs) :
        config = json.load(open(dirname + '/config.json', 'r'))
        config.update(kwargs)
        obj = cls(config)
        obj.load_values(dirname)
        return obj

    def train(self, train_data, train=True) :
        docs_in = train_data.P
        question_in = train_data.Q
        entity_masks_in = train_data.E
        target_in = train_data.A

        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in docs_in], noise_frac=0.1)
        docs = [docs_in[i] for i in sorting_idx]
        questions = [question_in[i] for i in sorting_idx]
        entity_masks = [entity_masks_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]
        
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        bsize = self.bsize
        N = len(questions)
        loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        for n in tqdm(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.LongTensor(batch_target).to(device)

            ce_loss = self.criterion(batch_data.predict, batch_target)

            loss = ce_loss

            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train :
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def evaluate(self, data) :
        docs = data.P
        questions = data.Q
        entity_masks = data.E
        
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        bsize = self.bsize
        N = len(questions)

        outputs = []
        attns = []
        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            batch_data.predict = torch.argmax(batch_data.predict, dim=-1)
            if self.decoder.use_attention :
                attn = batch_data.attn
                attns.append(attn.cpu().data.numpy())

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)
            
            

        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        
        return outputs, attns

    def save_values(self, use_dirname=None, save_model=True) :
        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.Pencoder.state_dict(), dirname + '/encP.th')
            torch.save(self.Qencoder.state_dict(), dirname + '/encQ.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.Pencoder.load_state_dict(torch.load(dirname + '/encP.th'))
        self.Qencoder.load_state_dict(torch.load(dirname + '/encQ.th'))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th'))

    def permute_attn(self, data, num_perm=100) :
        docs = data.P
        questions = data.Q
        entity_masks = data.E

        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        bsize = self.bsize
        N = len(questions)

        permutations_predict = []
        permutations_diff = []

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            predict_true = batch_data.predict.clone().detach()

            batch_perms_predict = np.zeros((batch_data.P.B, num_perm))
            batch_perms_diff = np.zeros((batch_data.P.B, num_perm))

            for i in range(num_perm) :
                batch_data.permute = True
                self.decoder(batch_data)

                predict = torch.argmax(batch_data.predict, dim=-1)
                batch_perms_predict[:, i] = predict.cpu().data.numpy()
            
                predict_difference = self.adversary_multi.output_diff(batch_data.predict, predict_true)
                batch_perms_diff[:, i] = predict_difference.squeeze(-1).cpu().data.numpy()
                
            permutations_predict.append(batch_perms_predict)
            permutations_diff.append(batch_perms_diff)

        permutations_predict = [x for y in permutations_predict for x in y]
        permutations_diff = [x for y in permutations_diff for x in y]
        
        return permutations_predict, permutations_diff

    def adversarial_multi(self, data) :
        docs = data.P
        questions = data.Q
        entity_masks = data.E

        self.Pencoder.eval()
        self.Qencoder.eval()
        self.decoder.eval()

        print(self.adversary_multi.K)
        
        self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())

        for p in self.params :
            p.requires_grad = False

        bsize = self.bsize
        N = len(questions)
        batches = list(range(0, N, bsize))

        outputs, attns, diffs = [], [], []

        for n in tqdm(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            self.adversary_multi(batch_data)

            predict_volatile = torch.argmax(batch_data.predict_volatile, dim=-1)
            outputs.append(predict_volatile.cpu().data.numpy())
            
            attn = batch_data.attn_volatile
            attns.append(attn.cpu().data.numpy())

            predict_difference = self.adversary_multi.output_diff(batch_data.predict_volatile, batch_data.predict.unsqueeze(1))
            diffs.append(predict_difference.cpu().data.numpy())

        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        diffs = [x for y in diffs for x in y]
        
        return outputs, attns, diffs

    def gradient_mem(self, data) :
        docs = data.P
        questions = data.Q
        entity_masks = data.E

        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        bsize = self.bsize
        N = len(questions)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in range(0, N, bsize) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            batch_data.P.keep_grads = True
            batch_data.detach = True

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)
            
            max_predict = torch.argmax(batch_data.predict, dim=-1)
            prob_predict = nn.Softmax(dim=-1)(batch_data.predict)

            max_class_prob = torch.gather(prob_predict, -1, max_predict.unsqueeze(-1))
            max_class_prob.sum().backward()

            g = batch_data.P.embedding.grad
            em = batch_data.P.embedding
            g1 = (g * em).sum(-1)
            
            grads['XxE[X]'].append(g1.cpu().data.numpy())
            
            g1 = (g * self.Pencoder.embedding.weight.sum(0)).sum(-1)
            grads['XxE'].append(g1.cpu().data.numpy())
            
            g1 = batch_data.P.hidden.grad.sum(-1)
            grads['H'].append(g1.cpu().data.numpy())


        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]
                    
        return grads       

    def remove_and_run(self, data) :
        docs = data.P
        questions = data.Q
        entity_masks = data.E

        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        bsize = self.bsize
        N = len(questions)
        output_diffs = []

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = BatchHolder(batch_doc)
            batch_ques = BatchHolder(batch_ques)

            batch_data = BatchMultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            po = np.zeros((batch_data.P.B, batch_data.P.maxlen))

            for i in range(1, batch_data.P.maxlen - 1) :
                batch_doc = BatchHolder(docs[n:n+bsize])

                batch_doc.seq = torch.cat([batch_doc.seq[:, :i], batch_doc.seq[:, i+1:]], dim=-1)
                batch_doc.lengths = batch_doc.lengths - 1
                batch_doc.masks = torch.cat([batch_doc.masks[:, :i], batch_doc.masks[:, i+1:]], dim=-1)

                batch_data_loop = BatchMultiHolder(P=batch_doc, Q=batch_ques)
                batch_data_loop.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).to(device)

                self.Pencoder(batch_data_loop.P)
                self.decoder(batch_data_loop)

                predict_difference = self.adversary_multi.output_diff(batch_data_loop.predict, batch_data.predict)

                po[:, i] = predict_difference.squeeze(-1).cpu().data.numpy()

            output_diffs.append(po)

        output_diffs = [x for y in output_diffs for x in y]
        
        return output_diffs
