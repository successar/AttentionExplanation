import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from tqdm import tqdm_notebook

from .modelUtils import isTrue, get_sorting_index_with_noise_from_lengths

file_name = os.path.abspath(__file__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BatchHolder() : 
    def __init__(self, data) :
        maxlen = max([len(x) for x in data])
        self.maxlen = maxlen
        self.B = len(data)

        lengths = []
        expanded = []
        masks = []

        for _, d in enumerate(data) :
            rem = maxlen - len(d)
            expanded.append(d + [0]*rem)
            lengths.append(len(d))
            masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))

        self.lengths = torch.LongTensor(np.array(lengths)).to(device)
        self.seq = torch.LongTensor(np.array(expanded, dtype='int64')).to(device)
        self.masks = torch.ByteTensor(np.array(masks)).to(device)

        self.hidden = None
        self.predict = None
        self.attn = None

    def generate_permutation(self) :
        perm_idx = np.tile(np.arange(self.maxlen), (self.B, 1))

        for i, x in enumerate(self.lengths) :
            perm = np.random.permutation(x.item()-2) + 1
            perm_idx[i, 1:x-1] = perm

        return perm_idx

    def generate_uniform_attn(self) :
        attn = np.zeros((self.B, self.maxlen))
        inv_l = 1. / self.lengths.cpu().data.numpy()
        attn += inv_l[:, None]
        return torch.Tensor(attn).to(device)

class BatchMultiHolder() :
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

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths
        embedding = self.embedding(seq) #(B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = output
        data.last_hidden = torch.cat([h[0], h[1]], dim=-1)

        if isTrue(data, 'keep_grads') :
            data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()

class TanhAttention(nn.Module) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1p = nn.Linear(hidden_size*2, hidden_size)
        self.attn1q = nn.Linear(hidden_size*2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, Poutput, Qoutput, mask) :
        attn1 = nn.Tanh()(self.attn1p(Poutput) + self.attn1q(Qoutput).unsqueeze(1))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn2.masked_fill_(mask, -float('inf'))
        attn = nn.Softmax(dim=-1)(attn2) #(B, L)

        return attn
        
class DotAttention(nn.Module) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, Poutput, Qoutput, mask) :
        attn1 = torch.bmm(Poutput, Qoutput.unsqueeze(-1)) / self.hidden_size**0.5
        attn1 = attn1.squeeze(-1)
        attn1.masked_fill_(mask, -float('inf'))
        attn = nn.Softmax(dim=-1)(attn1) #(B, L)

        return attn

class AttnDecoder(nn.Module) :
    def __init__(self, hidden_size, output_size, attention='tanh') :
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1q = nn.Linear(hidden_size*2, hidden_size)
        self.linear_1p = nn.Linear(hidden_size*2, hidden_size)

        self.linear_2 = nn.Linear(hidden_size, output_size)
        if attention == 'tanh' :
            self.attention = TanhAttention(hidden_size)
        elif attention == 'dot' :
            self.attention = DotAttention(hidden_size)
        else :
            raise NotImplementedError("No Attention Specified !!!")
         
    def decode(self, Poutput, Qoutput, entity_mask) :
        predict = self.linear_2(nn.Tanh()(self.linear_1p(Poutput) + self.linear_1q(Qoutput))) #(B, O)
        predict.masked_fill_(1 - entity_mask, -float('inf'))

        return predict

    def forward(self, data) :
        Poutput = data.P.hidden #(B, H, L)
        Qoutput = data.Q.last_hidden #(B, H)
        mask = data.P.masks

        attn = self.attention(Poutput, Qoutput, mask) #(B, L)

        if isTrue(data, 'detach') :
            attn = attn.detach()

        if isTrue(data, 'permute') :
            permutation = data.P.generate_permutation()
            attn = torch.gather(attn, -1, torch.LongTensor(permutation).to(device))

        context = (attn.unsqueeze(-1) * Poutput).sum(1) #(B, H)
        predict = self.decode(context, Qoutput, data.entity_mask)

        data.predict = predict
        data.attn = attn

    def get_output(self, data) :
        Poutput = data.P.hidden_volatile #(B, L, H)
        Qoutput = data.Q.last_hidden_volatile #(B, H)

        attn = data.attn_volatile #(B, K, L)

        if len(attn.shape) == 3 :
            predict = (attn.unsqueeze(-1) * Poutput.unsqueeze(1)).sum(2) #(B, K, H)
            predict = self.decode(predict, Qoutput.unsqueeze(1), data.entity_mask.unsqueeze(1))
        else :
            predict = (attn.unsqueeze(-1) * Poutput).sum(1) #(B, H)
            predict = self.decode(predict, Qoutput, data.entity_mask)

        data.predict_volatile = predict

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
    def __init__(self, vocab_size, embed_size, output_size, bsize, 
                       hidden_size=128, pre_embed=None, weight_decay=1e-5, 
                       attention='tanh', dirname='') :
        self.bsize = bsize
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.attention = attention

        self.config = {
            'dirname' : dirname,
            'vocab_size' : vocab_size,
            'embed_size' : embed_size,
            'output_size' : output_size,
            'bsize' : bsize,
            'hidden_size' : hidden_size,
            'weight_decay' : weight_decay,
            'attention' : attention
        }

        self.Pencoder = EncoderRNN(vocab_size, embed_size, self.hidden_size, pre_embed=pre_embed).to(device)
        self.Qencoder = EncoderRNN(vocab_size, embed_size, self.hidden_size, pre_embed=pre_embed).to(device)
        self.decoder = AttnDecoder(self.hidden_size, output_size, attention).to(device)
        self.adversary_multi = AdversaryMulti(self.decoder)

        self.params = list(self.Pencoder.parameters()) + list(self.Qencoder.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adagrad(self.params, lr=0.05, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        import time
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join('outputs', dirname, self.time_str)

    @classmethod
    def init_from_config(cls, dirname, **kwargs) :
        config = json.load(open(dirname + '/config.json', 'r'))
        obj = cls(**config, **kwargs)
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

        for n in tqdm_notebook(batches) :
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

        batches = list(range(0, N, bsize))

        outputs = []
        attns = []
        for n in tqdm_notebook(batches) :
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
            attn = batch_data.attn

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)
            
            attns.append(attn.cpu().data.numpy())

        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        
        return outputs, attns

    def save_values(self, use_dirname=None, save_model=True) :
        if use_dirname is not None :
            dirname = use_dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.config, open(dirname + '/config.json', 'w'))

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

        batches = list(range(0, N, bsize))

        permutations_predict = []
        permutations_diff = []

        for n in tqdm_notebook(batches) :
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

        batches = list(range(0, N, bsize))

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in batches :
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
