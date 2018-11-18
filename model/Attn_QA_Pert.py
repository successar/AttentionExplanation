import torch
import torch.nn as nn

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
        self.masks = torch.ByteTensor(masks).cuda()

        self.masks_no_ends = torch.ByteTensor(np.array(masks_no_ends)).cuda()

        self.correcting_idxs = torch.LongTensor(np.argsort(idxs)).cuda()
        self.sorting_idxs = torch.LongTensor(idxs).cuda()

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

class MultiHolder() :
    def __init__(self, **holders) :
        for name, value in holders.items() :
            setattr(self, name, value)

class EncoderRNN(nn.Module) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

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

        if isTrue(data, 'perturb_E') :
            pidx = data.perturb_idx
            embedding[:, pidx, :] += torch.randn(embedding.shape[0], embedding.shape[2]).cuda() * 0.01

        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True)

        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = data.correct(output)
        data.last_hidden = data.correct(torch.cat([h[0], h[1]], dim=-1))

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
    def __init__(self, hidden_size, output_size) :
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1q = nn.Linear(hidden_size*2, hidden_size)
        self.linear_1p = nn.Linear(hidden_size*2, hidden_size)

        self.linear_2 = nn.Linear(hidden_size, output_size)

        self.attn1p = nn.Linear(hidden_size*2, hidden_size)
        self.attn1q = nn.Linear(hidden_size*2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1, bias=False)

    def decode(self, Poutput, Qoutput, entity_mask) :
        predict = self.linear_2(nn.Tanh()(self.linear_1p(Poutput) + self.linear_1q(Qoutput))) #(B, O)
        predict.masked_fill_(1 - entity_mask, -float('inf'))

        return predict

    def forward(self, data) :
        Poutput = data.P.hidden #(B, H, L)
        Qoutput = data.Q.last_hidden #(B, H)
        mask = data.P.masks

        attn1 = nn.Tanh()(self.attn1p(Poutput) + self.attn1q(Qoutput).unsqueeze(1))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn2.masked_fill_(mask, -float('inf'))
        attn = nn.Softmax(dim=-1)(attn2) #(B, L)

        if isTrue(data, 'detach') :
            attn = attn.detach()

        if isTrue(data, 'permute') :
            permutation = data.P.generate_permutation()
            attn = torch.gather(attn, -1, torch.LongTensor(permutation).cuda())

        predict = (attn.unsqueeze(-1) * Poutput).sum(1) #(B, H)
        predict = self.decode(predict, Qoutput, data.entity_mask)

        data.predict = predict
        data.attn = attn

    def get_output(self, data) :
        Poutput = data.P.hidden_volatile #(B, L, H)
        Qoutput = data.Q.last_hidden_volatile #(B, H)

        attn = data.attn_volatile #(B, *, L)

        if isTrue(data, 'multiattention') :
            predict = (attn.unsqueeze(-1) * Poutput.unsqueeze(1)).sum(2) #(B, *, H)
            predict = self.decode(predict, Qoutput.unsqueeze(1), data.entity_mask.unsqueeze(-1))
        else :
            predict = (attn.unsqueeze(-1) * Poutput).sum(1) #(B, H)
            predict = self.decode(predict, Qoutput, data.entity_mask)

        data.predict_volatile = predict
        
    def output_individual(self, data) :
        Qoutput = data.Q.last_hidden #(B, H)
        Poutput = data.P.hidden_zero_run #(B, L, H)        
        predict = self.decode(Poutput, Qoutput.unsqueeze(1), data.entity_mask.unsqueeze(1))
        data.predict_volatile = predict


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

        Hp = torch.clamp(p, 0, 1) * torch.log(torch.clamp(p, 0, 1) + 1e-10)
        Hp = (-Hp).sum(-1)
        
        return jsd.unsqueeze(-1) #- Hp.unsqueeze(-1)

    def forward(self, data) :
        data.P.hidden_volatile = data.P.hidden.detach()
        data.Q.last_hidden_volatile = data.Q.last_hidden.detach()

        new_attn = torch.log(data.P.generate_uniform_attn())
        new_attn.requires_grad = True
        
        data.log_attn_volatile = new_attn 
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01)
        data.multiattention = False

        for _ in range(500) :
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.P.masks, -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
            self.decoder.get_output(data)
            predict_new = data.predict_volatile
            diff = nn.ReLU()(torch.abs(nn.Softmax(dim=-1)(predict_new) - nn.Softmax(dim=-1)(data.predict.detach())).sum(-1) - 1e-2)
            jsd = self.jsd(data.attn_volatile, data.attn.detach())
            loss =  -(jsd**1) + 200 * diff
            loss = loss.sum()
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.P.masks, -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)

class AdversaryMulti(nn.Module) :
    def __init__(self, decoder=None) :
        super().__init__()
        self.decoder = decoder
        self.K = 5

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
        data.P.hidden_volatile = data.P.hidden.detach()
        data.Q.last_hidden_volatile = data.Q.last_hidden.detach()

        new_attn = torch.log(data.P.generate_uniform_attn()).unsqueeze(1).repeat(1, self.K, 1) #(B, 10, L)
        new_attn = new_attn + torch.randn(new_attn.size()).cuda()*3

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

            jsd = self.jsd(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, *, 1)

            cross_jsd = self.jsd(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))

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
    def __init__(self, vocab_size, embed_size, output_size, bsize, hidden_size=128, pre_embed=None, dirname='') :
        self.bsize = bsize
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.Pencoder = EncoderRNN(vocab_size, embed_size, self.hidden_size, pre_embed=pre_embed).cuda()
        self.Qencoder = EncoderRNN(vocab_size, embed_size, self.hidden_size, pre_embed=pre_embed).cuda()
        self.decoder = AttnDecoder(self.hidden_size, output_size).cuda()
        self.adversary = Adversary(self.decoder)
        self.adversary_multi = AdversaryMulti(self.decoder)

        self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())
        # self.optim = torch.optim.Adam(self.params, lr=0.001, weight_decay=1e-5, amsgrad=True)
        self.optim = torch.optim.Adagrad(self.params, lr=0.05, weight_decay=1e-5)

        self.criterion = nn.CrossEntropyLoss()

        import time
        self.time_str = time.ctime().replace(' ', '')
        self.dirname = 'outputs/attn_QA_' + dirname + '/' + self.time_str

    def get_batch_variable(self, data) :
        return Holder(data, do_sort=True)

    def train(self, docs_in, question_in, entity_masks_in, target_in, train=True) :
        sorting_idx = np.argsort([len(x) for x in docs_in])
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

        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.LongTensor(batch_target).cuda()

            ce_loss = self.criterion(batch_data.predict, batch_target)

            loss = ce_loss

            if train :
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            loss_total += float(loss.data.cpu().item())
        return loss_total*bsize/N

    def evaluate(self, docs, questions, entity_masks) :
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        bsize = self.bsize
        N = len(questions)

        batches = list(range(0, N, bsize))

        outputs = []
        attns = []
        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            batch_data.predict = torch.argmax(batch_data.predict, dim=-1)
            attn = batch_data.attn

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)
            
            attns.append(attn.cpu().data.numpy())

        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        
        return outputs, attns

    def save_values(self, add_name='', save_model=True) :
        dirname = self.dirname + '_' + add_name
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        if save_model :
            torch.save(self.Pencoder.state_dict(), dirname + '/encP.th')
            torch.save(self.Qencoder.state_dict(), dirname + '/encQ.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.Pencoder.load_state_dict(torch.load(dirname + '/encP.th'))
        self.Qencoder.load_state_dict(torch.load(dirname + '/encQ.th'))

        self.decoder.load_state_dict(torch.load(dirname + '/dec.th'))

    def adversarial(self, docs, questions, entity_masks) :
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())

        for p in self.params :
            p.requires_grad = False

        bsize = self.bsize
        N = len(questions)

        batches = list(range(0, N, bsize))

        outputs = []
        attns = []
        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            self.adversary(batch_data)

            batch_data.predict_volatile = torch.argmax(batch_data.predict_volatile, dim=-1)
            attn = batch_data.attn_volatile
            
            predict = batch_data.predict_volatile.cpu().data.numpy()
            outputs.append(predict)
            
            attns.append(attn.cpu().data.numpy())

        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        
        return outputs, attns

    def permute_attn(self, docs, questions, entity_masks, num_perm=100) :
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        bsize = self.bsize
        N = len(questions)

        batches = list(range(0, N, bsize))

        permutations_predict = []
        permutations_diff = []

        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

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

    def adversarial_multi(self, docs, questions, entity_masks) :
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        print(self.adversary_multi.K)
        
        self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())

        for p in self.params :
            p.requires_grad = False

        bsize = self.bsize
        N = len(questions)
        batches = list(range(0, N, bsize))

        outputs, attns, diffs = [], [], []

        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

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

    def gradient_mem(self, docs, questions, entity_masks) :
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()
        
        bsize = self.bsize
        N = len(questions)

        batches = list(range(0, N, bsize))

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in batches :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

            batch_data.P.keep_grads = True
            batch_data.detach = True

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)
            
            max_predict = torch.argmax(batch_data.predict, dim=-1)
            prob_predict = nn.Softmax(dim=-1)(batch_data.predict)

            max_class_prob = torch.gather(prob_predict, -1, max_predict.unsqueeze(-1))
            max_class_prob.sum().backward()

            g = batch_data.P.correct(batch_data.P.embedding.grad)
            em = batch_data.P.correct(batch_data.P.embedding)
            g1 = (g * em).sum(-1)
            
            grads['XxE[X]'].append(g1.cpu().data.numpy())
            
            g1 = (g * self.Pencoder.embedding.weight.sum(0)).sum(-1)
            grads['XxE'].append(g1.cpu().data.numpy())
            
            g1 = batch_data.P.hidden.grad.sum(-1)
            grads['H'].append(g1.cpu().data.numpy())


        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]
                    
        return grads       

    def zero_H_run(self, docs, questions, entity_masks) :
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        self.Pencoder.gen_cells()
        
        bsize = self.bsize
        N = len(questions)

        batches = list(range(0, N, bsize))

        outputs = []
        output_diffs = []
        hidden_diffs = []

        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

            self.Pencoder(batch_data.P)
            self.Pencoder.run_cell_lstm_zero(batch_data.P)
            diff = torch.abs(batch_data.P.hidden - batch_data.P.Hcell_zero)
            diff = diff.mean(-1)
            hidden_diffs.append(diff.cpu().data.numpy())

            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            batch_data.P.hidden_zero_run = batch_data.P.Hcell_zero
            self.decoder.output_individual(batch_data)

            predict_volatile = torch.argmax(batch_data.predict_volatile, dim=-1)
            outputs.append(predict_volatile.cpu().data.numpy())

            predict_difference = self.adversary_multi.output_diff(batch_data.predict_volatile, batch_data.predict.unsqueeze(1))
            output_diffs.append(predict_difference.cpu().data.numpy())

        outputs = [x for y in outputs for x in y]
        output_diffs = [x for y in output_diffs for x in y]
        hidden_diffs = [x for y in hidden_diffs for x in y]
        
        return outputs, output_diffs, hidden_diffs

    def remove_and_run(self, docs, questions, entity_masks) :
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        self.Pencoder.gen_cells()
        
        bsize = self.bsize
        N = len(questions)

        batches = list(range(0, N, bsize))

        output_diffs = []

        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            po = np.zeros((batch_data.P.B, batch_data.P.maxlen))

            for i in range(1, batch_data.P.maxlen - 1) :
                batch_doc = self.get_batch_variable(docs[n:n+bsize])

                batch_doc.seq = torch.cat([batch_doc.seq[:, :i], batch_doc.seq[:, i+1:]], dim=-1)
                batch_doc.lengths = batch_doc.lengths - 1
                batch_doc.masks = torch.cat([batch_doc.masks[:, :i], batch_doc.masks[:, i+1:]], dim=-1)

                batch_data_loop = MultiHolder(P=batch_doc, Q=batch_ques)
                batch_data_loop.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

                self.Pencoder(batch_data_loop.P)
                self.decoder(batch_data_loop)

                predict_difference = self.adversary_multi.output_diff(batch_data_loop.predict, batch_data.predict)

                po[:, i] = predict_difference.squeeze(-1).cpu().data.numpy()

            output_diffs.append(po)

        output_diffs = [x for y in output_diffs for x in y]
        
        return output_diffs

    def sampling_top(self, docs, questions, entity_masks, sample_vocab=100, topnum=10) :
        self.Pencoder.train()
        self.Qencoder.train()
        self.decoder.train()

        bsize = self.bsize
        N = len(questions)

        batches = list(range(0, N, bsize))

        best_attn_idxs = []
        perts_attn = []
        perts_diffs = []
        words_sampled = []

        if hasattr(self, 'vec') :
            freqs = self.vec.freq.copy()
            freqs = torch.Tensor(freqs).cuda()
            freqs = freqs / torch.sum(freqs)
            sample_vocab = min(sample_vocab, self.vec.vocab_size)
        else :
            raise NotImplementedError("No Vec !!!")

        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = docs[n:n+bsize]
            batch_ques = questions[n:n+bsize]
            batch_entity_masks = entity_masks[n:n+bsize]

            batch_doc = self.get_batch_variable(batch_doc)
            batch_ques = self.get_batch_variable(batch_ques)

            batch_data = MultiHolder(P=batch_doc, Q=batch_ques)
            batch_data.entity_mask = torch.ByteTensor(np.array(batch_entity_masks)).cuda()

            if topnum is None :
                topnum = batch_data.P.maxlen

            topnum = min(topnum, batch_data.P.maxlen)

            pa = np.zeros((batch_data.P.B, batch_data.P.maxlen, sample_vocab, topnum))
            ws = np.zeros((batch_data.P.B, sample_vocab))
            po = np.zeros((batch_data.P.B, sample_vocab, topnum))
            
            self.Pencoder(batch_data.P)
            self.Qencoder(batch_data.Q)
            self.decoder(batch_data)

            predict_true = batch_data.predict.clone().detach()

            best_attn_idx = torch.topk(batch_data.attn, k=topnum, dim=1)[1]
            best_attn_idxs.append(best_attn_idx.cpu().data.numpy())

            for v in tqdm_notebook(range(sample_vocab)) :
                sv = torch.multinomial(freqs, batch_data.P.B, replacement=True).cuda()
                ws[:, v] = sv.cpu().data.numpy()

                for k in range(topnum) :
                    new_batch_doc = []
                    for i, x in enumerate(docs[n:n+bsize]) :
                        y = [w for w in x]
                        if best_attn_idx[i, k] < len(y) :
                            y[best_attn_idx[i, k]] = sv[i]
                        new_batch_doc.append(y)

                    batch_data.P = self.get_batch_variable(new_batch_doc)

                    self.Pencoder(batch_data.P)
                    self.decoder(batch_data)

                    attn = batch_data.attn
                    pa[:, :, v, k] = attn.cpu().data.numpy()

                    predict = batch_data.predict
                    predict_difference = self.adversary_multi.output_diff(predict, predict_true)
                    po[:, v, k] = predict_difference.squeeze(-1).cpu().data.numpy()
                
            perts_attn.append(pa)
            words_sampled.append(ws)
            perts_diffs.append(po)


        perts_attn = [x for y in perts_attn for x in y]
        words_sampled = [x for y in words_sampled for x in y]
        best_attn_idxs = [x for y in best_attn_idxs for x in y]
        perts_diffs = [x for y in perts_diffs for x in y]

        return perts_attn, words_sampled, best_attn_idxs, perts_diffs

