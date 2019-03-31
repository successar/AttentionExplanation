import json
import os
import shutil
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from allennlp.common import Params
from sklearn.utils import shuffle
from tqdm import tqdm_notebook

from Transparency.model.modules.Decoder import AttnDecoder
from Transparency.model.modules.Encoder import Encoder

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths
from .modelUtils import jsd as js_divergence

file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AdversaryMulti(nn.Module) :
    def __init__(self, decoder=None) :
        super().__init__()
        self.decoder = decoder
        self.K = 5

    def forward(self, data) :
        data.hidden_volatile = data.hidden.detach()

        new_attn = torch.log(data.generate_uniform_attn()).unsqueeze(1).repeat(1, self.K, 1) #(B, 10, L)
        new_attn = new_attn + torch.randn(new_attn.size()).to(device)*3

        new_attn.requires_grad = True
        
        data.log_attn_volatile = new_attn 
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01, amsgrad=True)

        for _ in range(500) :
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.masks.unsqueeze(1), -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn) #(B, 10, L)
            self.decoder.get_output(data)
            predict_new = data.predict_volatile #(B, 10, O)

            y_diff = torch.sigmoid(predict_new) - torch.sigmoid(data.predict.detach()).unsqueeze(1) #(B, 10, O)
            diff = nn.ReLU()(torch.abs(y_diff).sum(-1, keepdim=True) - 1e-2) #(B, 10, 1)

            jsd = js_divergence(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, 10, 1)
            cross_jsd = js_divergence(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))
            
            loss =  -(jsd**1) + 500 * diff
            loss = loss.sum() - cross_jsd.sum(0).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.masks.unsqueeze(1), -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)
        data.predict_volatile = torch.sigmoid(data.predict_volatile)

class Model() :
    def __init__(self, configuration, pre_embed=None) :
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed
        self.encoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)

        configuration['model']['decoder']['hidden_size'] = self.encoder.output_size
        self.decoder = AttnDecoder.from_params(Params(configuration['model']['decoder'])).to(device)

        self.encoder_params = list(self.encoder.parameters())
        self.attn_params = list([v for k, v in self.decoder.named_parameters() if 'attention' in k])
        self.decoder_params = list([v for k, v in self.decoder.named_parameters() if 'attention' not in k])

        self.bsize = configuration['training']['bsize']
        
        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.encoder_optim = torch.optim.Adam(self.encoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.attn_optim = torch.optim.Adam(self.attn_params, lr=0.001, weight_decay=0, amsgrad=True)
        self.decoder_optim = torch.optim.Adam(self.decoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.adversarymulti = AdversaryMulti(decoder=self.decoder)

        pos_weight = configuration['training'].get('pos_weight', [1.0]*self.decoder.output_size)
        self.pos_weight = torch.Tensor(pos_weight).to(device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)

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

    def train(self, data_in, target_in, train=True) :
        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]
        
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.Tensor(batch_target).to(device)

            if len(batch_target.shape) == 1 : #(B, )
                batch_target = batch_target.unsqueeze(-1) #(B, 1)

            bce_loss = self.criterion(batch_data.predict, batch_target)
            weight = batch_target * self.pos_weight + (1 - batch_target)
            bce_loss = (bce_loss * weight).mean(1).sum()

            loss = bce_loss

            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train :
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                self.attn_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
                self.attn_optim.step()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def evaluate(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict)
            if self.decoder.use_attention :
                attn = batch_data.attn.cpu().data.numpy()
                attns.append(attn)

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        if self.decoder.use_attention :
            attns = [x for y in attns for x in y]
        
        return outputs, attns 

    def gradient_mem(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]

            grads_xxe = []
            grads_xxex = []
            grads_H = []
            
            for i in range(self.decoder.output_size) :
                batch_data = BatchHolder(batch_doc)
                batch_data.keep_grads = True
                batch_data.detach = True

                self.encoder(batch_data) 
                self.decoder(batch_data)

                torch.sigmoid(batch_data.predict[:, i]).sum().backward()
                g = batch_data.embedding.grad
                em = batch_data.embedding
                g1 = (g * em).sum(-1)
                
                grads_xxex.append(g1.cpu().data.numpy())
                
                g1 = (g * self.encoder.embedding.weight.sum(0)).sum(-1)
                grads_xxe.append(g1.cpu().data.numpy())
                
                g1 = batch_data.hidden.grad.sum(-1)
                grads_H.append(g1.cpu().data.numpy())

            grads_xxe = np.array(grads_xxe).swapaxes(0, 1)
            grads_xxex = np.array(grads_xxex).swapaxes(0, 1)
            grads_H = np.array(grads_H).swapaxes(0, 1)

            grads['XxE'].append(grads_xxe)
            grads['XxE[X]'].append(grads_xxex)
            grads['H'].append(grads_H)

        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]
                    
        return grads       
    
    def remove_and_run(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)
            po = np.zeros((batch_data.B, batch_data.maxlen, self.decoder.output_size))

            for i in range(1, batch_data.maxlen - 1) :
                batch_data = BatchHolder(batch_doc)

                batch_data.seq = torch.cat([batch_data.seq[:, :i], batch_data.seq[:, i+1:]], dim=-1)
                batch_data.lengths = batch_data.lengths - 1
                batch_data.masks = torch.cat([batch_data.masks[:, :i], batch_data.masks[:, i+1:]], dim=-1)

                self.encoder(batch_data)
                self.decoder(batch_data)

                po[:, i] = torch.sigmoid(batch_data.predict).cpu().data.numpy()

            outputs.append(po)

        outputs = [x for y in outputs for x in y]
                    
        return outputs
    
    def permute_attn(self, data, num_perm=100) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        permutations = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            batch_perms = np.zeros((batch_data.B, num_perm, self.decoder.output_size))

            self.encoder(batch_data)
            self.decoder(batch_data)
            
            for i in range(num_perm) :
                batch_data.permute = True
                self.decoder(batch_data)
                output = torch.sigmoid(batch_data.predict)
                batch_perms[:, i] = output.cpu().data.numpy()

            permutations.append(batch_perms)

        permutations = [x for y in permutations for x in y]
                    
        return permutations
    
    def save_values(self, use_dirname=None, save_model=True) :
        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.encoder.load_state_dict(torch.load(dirname + '/enc.th', map_location={'cuda:1': 'cuda:0'}))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th', map_location={'cuda:1': 'cuda:0'}))

    def adversarial_multi(self, data) :
        self.encoder.eval()
        self.decoder.eval()

        for p in self.encoder.parameters() :
            p.requires_grad = False

        for p in self.decoder.parameters() :
            p.requires_grad = False

        bsize = self.bsize
        N = len(data)

        adverse_attn = []
        adverse_output = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            self.adversarymulti(batch_data)

            attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, 10, L)
            predict_volatile = batch_data.predict_volatile.cpu().data.numpy() #(B, 10, O)

            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]
        
        return adverse_output, adverse_attn

    def logodds_attention(self, data, logodds_map:Dict) :
        self.encoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        N = len(data)

        adverse_attn = []
        adverse_output = []

        logodds = np.zeros((self.encoder.vocab_size, ))
        for k, v in logodds_map.items() :
            if v is not None :
                logodds[k] = abs(v)
            else :
                logodds[k] = float('-inf')
        logodds = torch.Tensor(logodds).to(device)

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            attn = batch_data.attn #(B, L)
            batch_data.attn_logodds = logodds[batch_data.seq]
            self.decoder.get_output_from_logodds(batch_data)

            attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, L)
            predict_volatile = torch.sigmoid(batch_data.predict_volatile).cpu().data.numpy() #(B, O)

            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]
        
        return adverse_output, adverse_attn

    def logodds_substitution(self, data, top_logodds_words:Dict) :
        self.encoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        N = len(data)

        adverse_X = []
        adverse_attn = []
        adverse_output = []

        words_neg = torch.Tensor(top_logodds_words[0][0]).long().cuda().unsqueeze(0)
        words_pos = torch.Tensor(top_logodds_words[0][1]).long().cuda().unsqueeze(0)

        words_to_select = torch.cat([words_neg, words_pos], dim=0) #(2, 5)

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)
            predict_class = (torch.sigmoid(batch_data.predict).squeeze(-1) > 0.5)*1 #(B,)

            attn = batch_data.attn #(B, L)
            top_val, top_idx = torch.topk(attn, 5, dim=-1)
            subs_words = words_to_select[1 - predict_class.long()] #(B, 5)

            batch_data.seq.scatter_(1, top_idx, subs_words)

            self.encoder(batch_data)
            self.decoder(batch_data)

            attn_volatile = batch_data.attn.cpu().data.numpy() #(B, L)
            predict_volatile = torch.sigmoid(batch_data.predict).cpu().data.numpy() #(B, O)
            X_volatile = batch_data.seq.cpu().data.numpy()

            adverse_X.append(X_volatile)
            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_X = [x for y in adverse_X for x in y]
        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]
        
        return adverse_output, adverse_attn, adverse_X
