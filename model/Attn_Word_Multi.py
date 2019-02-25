import torch
import torch.nn as nn

import numpy as np
from sklearn.utils import shuffle
import os,shutil,json

file_name = os.path.abspath(__file__)

from .modelUtils import isTrue, get_sorting_index_with_noise_from_lengths

from tqdm import tqdm_notebook

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

class EncoderRNN(nn.Module) :
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, pre_embed=None) :
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

    def forward(self, data: BatchHolder) :
        seq = data.seq
        lengths = data.lengths
        embedding = self.embedding(seq) #(B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)
        data.hidden = output

        if isTrue(data, 'keep_grads') :
            data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()
        
class TanhAttention(nn.Module) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1 = nn.Linear(hidden_size * 2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, masks) :
        #hidden : (B, L, H), masks : (B, L)

        attn1 = nn.Tanh()(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn2.masked_fill_(masks, -float('inf'))
        attn = nn.Softmax(dim=-1)(attn2)
        
        return attn

class DotAttention(nn.Module) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1 = nn.Linear(hidden_size*2, 1, bias=False)
        self.hidden_size = hidden_size

    def forward(self, hidden, masks) :
        #hidden = (B, L, H), masks = (B, L)
        attn1 = self.attn1(hidden) / (self.hidden_size)**0.5
        attn1 = attn1.squeeze(-1)
        attn1.masked_fill_(masks, -float('inf'))
        attn = nn.Softmax(dim=-1)(attn1)

        return attn
       
class AttnDecoder(nn.Module) :
    def __init__(self, hidden_size:int , attention: str='tanh') :
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size*2, 1)
        if attention == 'tanh' :
            self.attention = TanhAttention(hidden_size)
        elif attention == 'dot' :
            self.attention = DotAttention(hidden_size)
        else :
            raise NotImplementedError("No Attention Mechanism specified")
               
    def decode(self, predict) :
        predict = self.linear_1(predict)
        return predict

    def forward(self, data: BatchHolder) :
        output = data.hidden
        mask = data.masks

        attn = self.attention(output, mask)

        if isTrue(data, 'detach') :
            attn = attn.detach()

        if isTrue(data, 'permute') :
            permutation = data.generate_permutation()
            attn = torch.gather(attn, -1, torch.LongTensor(permutation).to(device))

        context = (attn.unsqueeze(-1) * output).sum(1)
        predict = self.decode(context)

        data.predict = predict
        data.attn = attn

        if isTrue(data, 'keep_grads') :
            data.attn.retain_grad()

    def get_attention(self, data) :
        output = data.hidden_volatile
        mask = data.masks
        attn = self.attention(output, mask)
        data.attn_volatile = attn

    def get_output(self, data) :
        output = data.hidden_volatile #(B, L, H)
        attn = data.attn_volatile #(B, *, L)

        if len(attn.shape) == 3 :
            context = (attn.unsqueeze(-1) * output.unsqueeze(1)).sum(2) #(B, *, H)
            predict = self.decode(context)
        else :
            context = (attn.unsqueeze(-1) * output).sum(1)
            predict = self.decode(context)

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

    def forward_kld(self, p, q) :
        return self.kld(p, q).unsqueeze(-1)

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
            data.multiattention = True
            self.decoder.get_output(data)
            predict_new = data.predict_volatile #(B, 10, 1)

            y_diff = torch.sigmoid(predict_new) - torch.sigmoid(data.predict.detach()).unsqueeze(1) #(B, 10, 1)
            diff = nn.ReLU()(torch.abs(y_diff) - 1e-2) #(B, 10, 1)

            jsd = self.forward_kld(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, 10, 1)

            cross_jsd = self.jsd(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))
            
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
    def __init__(self, vocab_size, embed_size, bsize, hidden_size, 
                       pre_embed=None, pos_weight=1, weight_decay=1e-5, 
                       attention='tanh', dirname='') :
        self.bsize = bsize
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attention = attention

        self.config = {
            'dirname' : dirname,
            'vocab_size' : vocab_size,
            'embed_size' : embed_size,
            'bsize' : bsize,
            'hidden_size' : hidden_size,
            'pos_weight' : pos_weight,
            'weight_decay' : weight_decay,
            'attention' : attention
        }

        self.encoder = EncoderRNN(vocab_size, embed_size, self.hidden_size, pre_embed=pre_embed).to(device)
        self.decoder = AttnDecoder(self.hidden_size, attention=self.attention).to(device)

        self.encoder_params = list(self.encoder.parameters())
        self.attn_params = list([v for k, v in self.decoder.named_parameters() if 'attention' in k])
        self.decoder_params = list([v for k, v in self.decoder.named_parameters() if 'attention' not in k])

        self.encoder_optim = torch.optim.Adam(self.encoder_params, lr=0.001, weight_decay=weight_decay)
        self.attn_optim = torch.optim.Adam(self.attn_params, lr=0.001, weight_decay=0)
        self.decoder_optim = torch.optim.Adam(self.decoder_params, lr=0.001, weight_decay=weight_decay)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to(device))

        self.adversarymulti = AdversaryMulti(decoder=self.decoder)

        import time
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join('outputs', dirname, self.time_str)
        
    @classmethod
    def init_from_config(cls, dirname, **kwargs) :
        config = json.load(open(dirname + '/config.json', 'r'))
        obj = cls(**config, **kwargs)
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

            bce_loss = self.criterion(batch_data.predict, batch_target.unsqueeze(-1))

            loss = bce_loss

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

        for n in range(0, N, bsize) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

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
    
    def gradient(self, data, detach=False) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in range(0, N, bsize) :
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)
            batch_data.keep_grads = True
            batch_data.detach = detach

            self.encoder(batch_data)
            self.decoder(batch_data)
            
            torch.sigmoid(batch_data.predict).sum().backward()
            g = batch_data.embedding.grad
            g = torch.matmul(g, self.encoder.embedding.weight.transpose(0, 1))
            
            g1 = torch.gather(g, -1, batch_data.seq.unsqueeze(-1))
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
            batch_data = BatchHolder(batch_doc)
            batch_data.keep_grads = True
            batch_data.detach = True

            self.encoder(batch_data)
            self.decoder(batch_data)
            
            torch.sigmoid(batch_data.predict).sum().backward()
            g = batch_data.embedding.grad
            em = batch_data.embedding
            g1 = (g * em).sum(-1)
            
            grads['XxE[X]'].append(g1.cpu().data.numpy())
            
            g1 = (g * self.encoder.embedding.weight.sum(0)).sum(-1)
            grads['XxE'].append(g1.cpu().data.numpy())
            
            g1 = batch_data.hidden.grad.sum(-1)
            grads['H'].append(g1.cpu().data.numpy())


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
            po = np.zeros((batch_data.B, batch_data.maxlen))

            for i in range(1, batch_data.maxlen - 1) :
                batch_data = BatchHolder(batch_doc)

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
            batch_data = BatchHolder(batch_doc)

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
    
    def save_values(self, use_dirname=None, save_model=True) :
        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.config, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.encoder.load_state_dict(torch.load(dirname + '/enc.th', map_location={'cuda:1': 'cuda:0'}))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th', map_location={'cuda:1': 'cuda:0'}))

    def adversarial_multi(self, data, _type='perturb') :
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
            batch_data.adversary_type = _type

            self.encoder(batch_data)
            self.decoder(batch_data)

            self.adversarymulti(batch_data)

            attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, 10, L)
            predict_volatile = batch_data.predict_volatile.squeeze(-1).cpu().data.numpy() #(B, 10, 1)

            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]
        
        return adverse_output, adverse_attn
