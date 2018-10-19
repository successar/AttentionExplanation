import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients

import numpy as np
from sklearn.utils import shuffle
import os,shutil

file_name = os.path.abspath(__file__)

from .modelUtils import isTrue

from tqdm import tqdm_notebook

class Holder() :
    
    def __init__(self, data, do_sort=False) :
        maxlen = max([len(x) for x in data])
        self.maxlen = maxlen
        self.B = len(data)
        lengths = []
        expanded = []
        masks = []
        masks_no_ends = []

        for _, d in enumerate(data) :
            rem = maxlen - len(d)
            expanded.append(d + [0]*rem)
            lengths.append(len(d))
            masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))
            masks_no_ends.append([0]*len(d) + [1]*(rem))


        lengths = np.array(lengths)  
        self.orig_lengths = lengths.copy()

        expanded = np.array(expanded, dtype='int64')
        idxs = np.flip(np.argsort(lengths), axis=0).copy()

        self.do_sort = do_sort

        if do_sort :
            lengths = lengths[idxs]
            expanded = expanded[idxs]

        self.lengths = torch.LongTensor(lengths).cuda()
        self.seq = torch.LongTensor(expanded).cuda()

        masks = np.array(masks)
        masks = masks #NOT IDXED
        self.masks = torch.ByteTensor(masks).cuda()

        self.masks_no_ends = torch.ByteTensor(np.array(masks_no_ends)).cuda()

        self.correcting_idxs = torch.LongTensor(np.argsort(idxs)).cuda()
        self.sorting_idxs = torch.LongTensor(idxs).cuda()

        self.hidden_seq = []
        self.hidden = None

        self.predict = None
        self.attn = None

    def sort(self, seq) :
        return seq[self.sorting_idxs]

    def correct(self, seq) :
        return seq[self.correcting_idxs]

    def generate_permutation(self) :
        perm_idx = np.tile(np.arange(self.maxlen), (self.B, 1))

        for i, x in enumerate(self.orig_lengths) :
            perm = np.random.permutation(x-2) + 1
            perm_idx[i, 1:x-1] = perm

        return perm_idx

    def generate_uniform_attn(self) :
        attn = np.zeros((self.B, self.maxlen))
        inv_l = 1. / self.orig_lengths
        attn += inv_l[:, None]
        return torch.Tensor(attn).cuda()

class EncoderRNN(nn.Module) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

        self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

    def gen_cells(self) :

        weights = ['bias_hh_l0', 'bias_ih_l0', 'weight_hh_l0', 'weight_ih_l0']
        weights = {k:getattr(self.rnn, k) for k in weights}
        weights_reverse = ['bias_hh_l0_reverse', 'bias_ih_l0_reverse', 'weight_hh_l0_reverse', 'weight_ih_l0_reverse']
        weights_reverse = {k:getattr(self.rnn, k) for k in weights_reverse}

        self.cell = nn.LSTMCell(weights['weight_ih_l0'].shape[1], weights['weight_ih_l0'].shape[0]//4).cuda()
        self.cell_reverse = nn.LSTMCell(weights['weight_ih_l0'].shape[1], weights['weight_ih_l0'].shape[0]//4).cuda()

        for k in weights :
            setattr(self.cell, k[:-3], weights[k])

        for k in weights_reverse :
            setattr(self.cell_reverse, k[:-11], weights_reverse[k])

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths

        embedding = self.embedding(seq) #(B, L, E)

        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True)

        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = data.correct(output)

        if isTrue(data, 'keep_grads') :
            data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()

    def run_cell_lstm(self, data) :
        seq = data.seq

        weights = ['weight_hh', 'weight_ih', 'bias_hh', 'bias_ih']
        for k in weights :
            assert (getattr(self.cell, k) == getattr(self.rnn, k + '_l0')).all()
            assert (getattr(self.cell_reverse, k) == getattr(self.rnn, k + '_l0_reverse')).all()

        embedding = self.embedding(seq) #(B, L, E)
        embedding = data.correct(embedding)
        embedding.retain_grad()

        T = embedding.shape[1]
        Hf1, Hb1 = [], []
        Hf, Hb = [], []
        Cf, Cb = [], []

        for t in range(T) :
            if t == 0: 
                hf, cf = self.cell(embedding[:, t])
                hb, cb = self.cell_reverse(embedding[:, T-t-1])
            else :
                hf, cf = self.cell(embedding[:, t], (Hf[-1], Cf[-1]))
                hb, cb = self.cell_reverse(embedding[:, T-t-1], (Hb[-1], Cb[-1]))

            hf.retain_grad()
            hb.retain_grad()

            Hf1.append(hf)
            Hb1.append(hb)

            hf = hf * (1 - data.masks_no_ends[:, t]).unsqueeze(-1).float()
            hb = hb * (1 - data.masks_no_ends[:, T-t-1]).unsqueeze(-1).float()

            cf = cf * (1 - data.masks_no_ends[:, t]).unsqueeze(-1).float()
            cb = cb * (1 - data.masks_no_ends[:, T-t-1]).unsqueeze(-1).float()

            Hf.append(hf)
            Cf.append(cf)
            Hb.append(hb)
            Cb.append(cb)

        Hb = Hb[::-1]
        Hb1 = Hb1[::-1]
        data.Hf = Hf1
        data.Hb = Hb1

        data.embedding_cell = embedding

        data.Hcell = [torch.cat([Hf[x], Hb[x]], dim=-1) for x in range(T)]        

    def run_cell_lstm_zero(self, data) :
        seq = data.seq

        weights = ['weight_hh', 'weight_ih', 'bias_hh', 'bias_ih']
        for k in weights :
            assert (getattr(self.cell, k) == getattr(self.rnn, k + '_l0')).all()
            assert (getattr(self.cell_reverse, k) == getattr(self.rnn, k + '_l0_reverse')).all()

        embedding = self.embedding(seq) #(B, L, E)
        embedding = data.correct(embedding)

        B, _, _ = embedding.shape

        T = embedding.shape[1]
        Hf1, Hb1 = [], []
        Hf, Hb = [], []
        Cf, Cb = [], []

        for t in range(T) :
            phf = torch.zeros((B, self.cell.hidden_size)).cuda()
            pcf = torch.zeros((B, self.cell.hidden_size)).cuda()

            hf, cf = self.cell(embedding[:, t], (phf, pcf))
            hb, cb = self.cell_reverse(embedding[:, T-t-1], (phf, pcf))

            hf = hf * (1 - data.masks_no_ends[:, t]).unsqueeze(-1).float()
            hb = hb * (1 - data.masks_no_ends[:, T-t-1]).unsqueeze(-1).float()

            Hf1.append(hf.unsqueeze(1))
            Hb1.append(hb.unsqueeze(1))

        Hb1 = Hb1[::-1]
        data.Hcell_zero = torch.cat([torch.cat([Hf1[x], Hb1[x]], dim=-1) for x in range(T)], dim=1)
       

class AttnDecoder(nn.Module) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size*2, 1)
        self.attn1 = nn.Linear(hidden_size*2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, data) :
        output = data.hidden
        mask = data.masks

        attn1 = nn.Tanh()(self.attn1(output))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn2.masked_fill_(mask, -float('inf'))
        attn = nn.Softmax(dim=-1)(attn2)

        if isTrue(data, 'detach') :
            attn = attn.detach()

        if isTrue(data, 'permute') :
            permutation = data.generate_permutation()
            attn = torch.gather(attn, -1, torch.LongTensor(permutation).cuda())

        predict = (attn.unsqueeze(-1) * output).sum(1)
        predict = self.linear_1(predict)

        data.predict = predict
        data.attn = attn

        if isTrue(data, 'keep_grads') :
            data.attn.retain_grad()

    def get_attention(self, data) :
        output = data.hidden_volatile
        mask = data.masks

        attn1 = nn.Tanh()(self.attn1(output))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn2.masked_fill_(mask, -float('inf'))
        attn = nn.Softmax(dim=-1)(attn2)

        data.attn_volatile = attn

    def get_output(self, data) :
        output = data.hidden_volatile
        attn = data.attn_volatile

        predict = (attn.unsqueeze(-1) * output).sum(1)
        predict = self.linear_1(predict)

        data.predict_volatile = predict

    def output_individual(self, data) :
        output = data.hidden_zero_run
        predict = self.linear_1(output)
        data.predict_zero = predict


class Adversary(nn.Module) :
    def __init__(self, decoder=None) :
        super().__init__()
        self.decoder = decoder

    def kld(self, a1, a2) :
        #(B, *, A), #(B, *, A)
        a1 = torch.clamp(a1, 0, 1)
        a2 = torch.clamp(a2, 0, 1)
        log_a1 = torch.log(a1 + 1e-10)
        log_a2 = torch.log(a2 + 1e-10)

        kld = a1 * (log_a1 - log_a2)
        kld = kld.sum(-1)

        return kld

    def jsd(self, p, q) :
        m = 0.5 * (p + q)
        jsd = 0.5 * (self.kld(p, m) + self.kld(q, m))

        return jsd.unsqueeze(-1)

    def forward(self, data) :
        data.hidden_volatile = data.hidden

        if data.adversary_type == 'perturb' :
            new_attn = torch.log(data.attn.detach().clone()) 
            new_attn += torch.randn(new_attn.size()).cuda()
        else :
            new_attn = torch.log(data.generate_uniform_attn())
        new_attn.requires_grad = True
        
        data.log_attn_volatile = new_attn 
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01)

        for _ in range(500) :
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.masks, -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
            self.decoder.get_output(data)
            predict_new = data.predict_volatile
            diff = nn.ReLU()(torch.abs(torch.sigmoid(predict_new) - torch.sigmoid(data.predict.detach())) - 1e-2)
            jsd = self.jsd(data.attn_volatile, data.attn.detach())
            loss =  -(jsd**1) + 500 * diff
            loss = loss.sum()
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.masks, -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)
        data.predict_volatile = torch.sigmoid(data.predict_volatile)


class Model() :
    def __init__(self, vocab_size, embed_size, bsize, hidden_size=128, pre_embed=None, pos_weight=1, dirname='') :
        self.bsize = bsize
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(vocab_size, embed_size, self.hidden_size, pre_embed=pre_embed).cuda()
        self.decoder = AttnDecoder(self.hidden_size).cuda()

        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.params, lr=0.001, weight_decay=1e-5)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).cuda())

        self.adversary = Adversary(decoder=self.decoder)

        import time
        self.time_str = time.ctime().replace(' ', '')
        self.dirname = 'outputs/attn_word_' + dirname + '/' + self.time_str

    def get_batch_variable(self, data) :
        data = Holder(data, do_sort=True)
        
        return data

    def train(self, data_in, target_in, train=True) :
        sorting_idx = np.argsort([len(x) for x in data_in])
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
            batch_data = self.get_batch_variable(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.Tensor(batch_target).cuda()

            bce_loss = self.criterion(batch_data.predict, batch_target.unsqueeze(-1))

            loss = bce_loss

            if train :
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def evaluate(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []

        for n in range(0, N, bsize) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict)
            attn = batch_data.attn.cpu().data.numpy()
            attns.append(attn)

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)


        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        
        return outputs, attns

    def sampling_all(self, data, sample_vocab=200) :
        self.encoder.eval()
        self.decoder.eval()
        bsize = self.bsize
        N = len(data)

        perts_predict = []
        perts_attn = []
        words_sampled = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)
            pp = np.zeros((batch_data.B, batch_data.maxlen, sample_vocab))
            pa = np.zeros((batch_data.B, batch_data.maxlen, sample_vocab, batch_data.maxlen))
            ws = np.zeros((batch_data.B, sample_vocab))

            if hasattr(self, 'vec') :
                freqs = self.vec.freq.copy()
                per = np.percentile(freqs, 90)
                freqs[freqs < per] = 0
                freqs = torch.Tensor(freqs).cuda()
                freqs = freqs / torch.sum(freqs)
            else :
                raise "No Vec !!!"

            for v in tqdm_notebook(range(sample_vocab)) :
                # sv = torch.randint(4, self.vocab_size, (batch_data.B, )).cuda()
                sv = torch.multinomial(freqs, batch_data.B).cuda()
                ws[:, v] = sv.cpu().data.numpy()
                for i in range(batch_data.maxlen) :
                    batch_data = self.get_batch_variable(batch_doc)
                    
                    batch_data.seq[:, i] = sv
                    self.encoder(batch_data)
                    self.decoder(batch_data)

                    attn = batch_data.attn.cpu().squeeze().data.numpy()
                    pa[:, i, v, :] = attn

                    predict = batch_data.predict.cpu().data.numpy()
                    pp[:, i, v] = predict[:, 0]
                
            perts_attn.append(pa)
            perts_predict.append(pp)
            words_sampled.append(ws)
        
        perts_attn = [x for y in perts_attn for x in y]
        perts_predict = [x for y in perts_predict for x in y]
        words_sampled = [x for y in words_sampled for x in y]

        return perts_predict, perts_attn, words_sampled

    def sampling_top(self, data, sample_vocab=100, topnum=10) :
        self.encoder.train()
        self.decoder.train()

        bsize = self.bsize
        N = len(data)

        best_attn_idxs = []
        perts_attn = []
        perts_output = []
        words_sampled = []

        if hasattr(self, 'vec') :
            freqs = self.vec.freq.copy()
            per = np.percentile(freqs, 90)
            freqs[freqs < per] = 0
            freqs = torch.Tensor(freqs).cuda()
            freqs = freqs / torch.sum(freqs)
            print("Non Zero : ", sum(freqs != 0))
        else :
            raise NotImplementedError("No Vec !!!")

        for n in tqdm_notebook(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)

            pa = np.zeros((batch_data.B, batch_data.maxlen, sample_vocab, topnum))
            ws = np.zeros((batch_data.B, sample_vocab))
            po = np.zeros((batch_data.B, sample_vocab, topnum))
            
            self.encoder(batch_data)
            self.decoder(batch_data)
            batch_data.predict = torch.sigmoid(batch_data.predict)

            best_attn_idx = torch.topk(batch_data.attn, k=topnum, dim=1)[1]
            best_attn_idxs.append(best_attn_idx.cpu().data.numpy())

            for v in tqdm_notebook(range(sample_vocab)) :
                sv = torch.multinomial(freqs, batch_data.B).cuda()
                ws[:, v] = sv.cpu().data.numpy()

                for k in range(topnum) :
                    new_batch_doc = []
                    for i, x in enumerate(batch_doc) :
                        y = [w for w in x]
                        if best_attn_idx[i, k] < len(y) :
                            y[best_attn_idx[i, k]] = sv[i]
                        new_batch_doc.append(y)

                    batch_data = self.get_batch_variable(new_batch_doc)

                    self.encoder(batch_data)
                    self.decoder(batch_data)
                    batch_data.predict = torch.sigmoid(batch_data.predict)

                    attn = batch_data.attn
                    pa[:, :, v, k] = attn.cpu().data.numpy()

                    po[:, v, k] = batch_data.predict.squeeze(-1).cpu().data.numpy()
                
            perts_attn.append(pa)
            words_sampled.append(ws)
            perts_output.append(po)


        perts_attn = [x for y in perts_attn for x in y]
        words_sampled = [x for y in words_sampled for x in y]
        best_attn_idxs = [x for y in best_attn_idxs for x in y]
        perts_output = [x for y in perts_output for x in y]

        return perts_attn, words_sampled, best_attn_idxs, perts_output
    
    def gradient(self, data, detach=False) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in range(0, N, bsize) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)
            batch_data.keep_grads = True
            batch_data.detach = detach

            self.encoder(batch_data)
            self.decoder(batch_data)
            
            torch.sigmoid(batch_data.predict).sum().backward()
            g = batch_data.correct(batch_data.embedding.grad)
            g = torch.matmul(g, self.encoder.embedding.weight.transpose(0, 1))
            
            g1 = torch.gather(g, -1, batch_data.correct(batch_data.seq).unsqueeze(-1))
            grads['XxE[X]'].append(g1.squeeze(-1).cpu().data.numpy())
            
            g1 = g.sum(-1)
            grads['XxE'].append(g1.cpu().data.numpy())
            
            g1 = batch_data.hidden.grad.sum(-1)
            grads['H'].append(g1.cpu().data.numpy())


        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]
                    
        return grads 

    def gradient_mem(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in range(0, N, bsize) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)
            batch_data.keep_grads = True

            self.encoder(batch_data)
            self.decoder(batch_data)
            
            torch.sigmoid(batch_data.predict).sum().backward()
            g = batch_data.correct(batch_data.embedding.grad)
            em = batch_data.correct(batch_data.embedding)
            g1 = (g * em).sum(-1)
            
            grads['XxE[X]'].append(g1.cpu().data.numpy())
            
            g1 = (g * self.encoder.embedding.weight.sum(0)).sum(-1)
            grads['XxE'].append(g1.cpu().data.numpy())
            
            g1 = batch_data.hidden.grad.sum(-1)
            grads['H'].append(g1.cpu().data.numpy())


        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]
                    
        return grads       

    def gradient_wrt_H(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'prev_h' : [], 'subs_h' : [], 'input_g' : []}

        for n in range(0, N, bsize) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)

            self.encoder(batch_data)
            self.encoder.run_cell_lstm(batch_data)

            # breakpoint()

            for t in range(1, batch_data.maxlen) :
                try :
                    assert torch.allclose(batch_data.Hcell[t][:, :], batch_data.hidden[:, t, :], atol=1e-4)
                except AssertionError:
                    breakpoint()

            prev_h = []
            subs_h = []
            input_g = []

            for i in range(1, batch_data.hidden.shape[1]-1) :
                zero_gradients(batch_data.Hf)
                zero_gradients(batch_data.Hb)

                batch_data.Hcell[i].sum().backward(retain_graph=True)

                g1 = batch_data.Hf[i-1].grad #(B, H)
                g2 = batch_data.Hb[i+1].grad #(B, H)
                e1 = batch_data.embedding_cell.grad[:, i] #(B, E)

                norm_g1 = torch.norm(g1, 1, -1, keepdim=True) / g1.shape[-1] #(B,)
                norm_g2 = torch.norm(g2, 1, -1, keepdim=True) / g2.shape[-1] #(B,)
                norm_e1 = torch.norm(e1, 1, -1, keepdim=True) / e1.shape[-1] #(B,)

                prev_h.append(norm_g1)
                subs_h.append(norm_g2)
                input_g.append(norm_e1)

                # breakpoint()

            grads['prev_h'].append(torch.cat(prev_h, dim=-1).cpu().data.numpy())
            grads['subs_h'].append(torch.cat(subs_h, dim=-1).cpu().data.numpy())
            grads['input_g'].append(torch.cat(input_g, dim=-1).cpu().data.numpy())

        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]
                    
        return grads 

    def zero_H_run(self, data) :
        self.encoder.train()
        self.decoder.train()

        self.encoder.gen_cells()
        bsize = self.bsize
        N = len(data)

        grads = []
        outputs = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)

            self.encoder(batch_data)
            self.encoder.run_cell_lstm_zero(batch_data)

            diff = torch.abs(batch_data.hidden - batch_data.Hcell_zero)
            diff = diff.mean(-1)
            diff = diff / diff.sum(-1).unsqueeze(-1)

            batch_data.hidden_zero_run = batch_data.Hcell_zero
            self.decoder.output_individual(batch_data)
            output = torch.sigmoid(batch_data.predict_zero).squeeze(-1).cpu().data.numpy()

            outputs.append(output)
            grads.append(diff.cpu().data.numpy())

        grads = [x for y in grads for x in y]
        outputs = [x for y in outputs for x in y]
                    
        return outputs, grads 

    def copy_H_run(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = []
        H_change = []
        outputs = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)

            pa = np.zeros((batch_data.B, batch_data.maxlen))
            po = np.zeros((batch_data.B, batch_data.maxlen))
            ph = np.zeros((batch_data.B, batch_data.maxlen))

            self.encoder(batch_data)
            for i in range(1, batch_data.maxlen-1) :
                hidden_copy = batch_data.hidden.clone()
                H = self.encoder.rnn.hidden_size
                hidden_copy[:, i, H:] = hidden_copy[:, i+1, H:]
                hidden_copy[:, i, :H] = hidden_copy[:, i-1, :H]

                batch_data.hidden_volatile = hidden_copy
                self.decoder.get_attention(batch_data)
                self.decoder.get_output(batch_data)

                ph[:, i] = torch.abs(batch_data.hidden[:, i] - hidden_copy[:, i]).mean(-1).cpu().data.numpy()
                pa[:, i] = batch_data.attn_volatile[:, i].cpu().data.numpy()
                po[:, i] = torch.sigmoid(batch_data.predict_volatile).squeeze(-1).cpu().data.numpy()

            grads.append(pa)
            outputs.append(po)
            H_change.append(ph)
            

        grads = [x for y in grads for x in y]
        outputs = [x for y in outputs for x in y]
        H_change = [x for y in H_change for x in y]
                    
        return outputs, grads, H_change 
    
    def remove_and_run(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)
            po = np.zeros((batch_data.B, batch_data.maxlen))

            for i in range(1, batch_data.maxlen - 1) :
                batch_data = self.get_batch_variable(batch_doc)

                batch_data.seq = torch.cat([batch_data.seq[:, :i], batch_data.seq[:, i+1:]], dim=-1)
                batch_data.lengths = batch_data.lengths - 1
                batch_data.masks = torch.cat([batch_data.masks[:, :i], batch_data.masks[:, i+1:]], dim=-1)

                self.encoder(batch_data)
                self.decoder(batch_data)

                po[:, i] = torch.sigmoid(batch_data.predict).squeeze(-1).cpu().data.numpy()

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
            batch_data = self.get_batch_variable(batch_doc)

            batch_perms = np.zeros((batch_data.B, num_perm))

            self.encoder(batch_data)
            self.decoder(batch_data)
            
            for i in tqdm_notebook(range(num_perm)) :
                batch_data.permute = True
                self.decoder(batch_data)
                output = torch.sigmoid(batch_data.predict).squeeze(-1)
                batch_perms[:, i] = output.cpu().data.numpy()

            permutations.append(batch_perms)

        permutations = [x for y in permutations for x in y]
                    
        return permutations
    
    def save_values(self, add_name='', save_model=True) :
        dirname = self.dirname + '_' + add_name
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        if save_model :
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')
            torch.save(self.optim.state_dict(), dirname + '/optimizer.th')

        return dirname

    def load_values(self, dirname) :
        self.encoder.load_state_dict(torch.load(dirname + '/enc.th'))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th'))
        self.optim.load_state_dict(torch.load(dirname + '/optimizer.th'))

    def adversarial(self, data, _type='perturb') :
        self.encoder.eval()
        self.decoder.eval()
        bsize = self.bsize
        N = len(data)

        adverse_attn = []
        adverse_output = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)
            batch_data.adversary_type = _type

            self.encoder(batch_data)
            self.decoder(batch_data)

            self.adversary(batch_data)

            attn_volatile = batch_data.attn_volatile.cpu().data.numpy()
            predict_volatile = batch_data.predict_volatile.cpu().data.numpy()

            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]
        
        return adverse_output, adverse_attn

    def perturbation_embedding(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)
            po = np.zeros((batch_data.B, batch_data.maxlen))

            for i in range(1, batch_data.maxlen - 1) :
                batch_data = self.get_batch_variable(batch_doc)

                batch_data.seq = torch.cat([batch_data.seq[:, :i], batch_data.seq[:, i+1:]], dim=-1)
                batch_data.lengths = batch_data.lengths - 1
                batch_data.masks = torch.cat([batch_data.masks[:, :i], batch_data.masks[:, i+1:]], dim=-1)

                self.encoder(batch_data)
                self.decoder(batch_data)

                po[:, i] = torch.sigmoid(batch_data.predict).squeeze(-1).cpu().data.numpy()

            outputs.append(po)

        outputs = [x for y in outputs for x in y]
                    
        return outputs