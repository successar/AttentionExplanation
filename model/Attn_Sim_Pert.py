import torch
import torch.nn as nn

import numpy as np
from sklearn.utils import shuffle
import os,shutil

file_name = os.path.abspath(__file__)

from tqdm import tqdm_notebook

class Holder() :
    
    def __init__(self, data, vocab_size) :
        self.B = len(data)
        self.len = len(data[0])
        lengths = []
        expanded = []

        eye = np.eye(vocab_size)

        for i, d in enumerate(data) :
            expanded.append(d)
            lengths.append(len(d))

        lengths = np.array(lengths)       
        expanded = np.array(expanded)

        self.lengths = torch.LongTensor(lengths).cuda()
        expanded = eye[expanded]
        self.seq = torch.FloatTensor(expanded).cuda()
        self.seq.requires_grad_()

        self.hidden_seq = []
        self.hidden = None

        self.predict = None
        self.attn = None

class EncoderRNN(nn.Module) :
    def __init__(self, vocab_size, embed_size, hidden_size) :
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths
        data.seq.retain_grad()

        embedding = torch.matmul(seq, self.embedding.weight) #(B, L, E)
        B, L, E = embedding.shape

        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True)

        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = output
        data.embedding = embedding
        data.embedding.retain_grad()
        data.hidden.retain_grad()

class AttnDecoder(nn.Module) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size*2, 1)
        self.attn1 = nn.Linear(hidden_size*2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1)

    def forward(self, data) :
        output = data.hidden

        attn1 = nn.Tanh()(self.attn1(output))
        attn2 = self.attn2(attn1).squeeze()
        data.sims = attn2
        attn = nn.Softmax(dim=-1)(attn2)

        predict = (attn.unsqueeze(-1) * output).sum(1)
        predict = torch.sigmoid(self.linear_1(predict)) #(B, 1)
        predict = torch.clamp(predict, 0, 1)

        data.predict = predict
        data.attn = attn

class Model() :
    def __init__(self, vocab_size, embed_size, bsize, hidden_size=128, dirname='') :
        self.bsize = bsize
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(vocab_size, embed_size, self.hidden_size).cuda()
        self.decoder = AttnDecoder(self.hidden_size).cuda()

        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.params, lr=0.001, weight_decay=0.000001)
        self.criterion = nn.BCELoss()

        import time
        self.time_str = time.ctime().replace(' ', '')
        self.dirname = 'outputs/attn_sim_pert_' + dirname + '/' + self.time_str

    def get_batch_variable(self, data) :
        data = Holder(data, vocab_size=self.vocab_size)
        
        return data

    def train(self, data_in, target_in, train=True) :
        data, target = shuffle(data_in, target_in)
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))

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
        perts_predict = []
        perts_attn = []

        for n in range(0, N, bsize) :
            batch_doc = np.array(data[n:n+bsize])
            batch_data = self.get_batch_variable(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data, detach=False)

            attn = batch_data.attn.cpu().squeeze().data.numpy()
            attns.append(attn)

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)


        outputs = [x for y in outputs for x in y]
        attns = [x for y in attns for x in y]
        
        return outputs, attns

    def sampling(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        perts_predict = []
        perts_attn = []

        for n in range(0, N, bsize) :
            batch_doc = np.array(data[n:n+bsize])
            batch_data = self.get_batch_variable(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data, detach=False)
            
            pp = np.zeros((batch_data.B, len(data[0]), self.vocab_size))
            pa = np.zeros((batch_data.B, len(data[0]), self.vocab_size, len(data[0])))

            for i in range(batch_data.len) :
                for v in range(self.vocab_size) :
                    val = batch_doc[:, i].copy()
                    batch_doc[:, i] = v
                    batch_data = self.get_batch_variable(batch_doc)

                    self.encoder(batch_data)
                    self.decoder(batch_data, detach=False)

                    attn = batch_data.attn.cpu().squeeze().data.numpy()
                    pa[:, i, v, :] = attn

                    predict = batch_data.predict.cpu().data.numpy()
                    pp[:, i, v] = predict[:, 0]

                    batch_doc[:, i] = val

            perts_attn.append(pa)
            perts_predict.append(pp)
        
        perts_attn = [x for y in perts_attn for x in y]
        perts_predict = [x for y in perts_predict for x in y]
        return perts_predict, perts_attn
    
    def gradient(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in range(0, N, bsize) :
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)

            self.encoder(batch_data)
            batch_data.detach = False
            self.decoder(batch_data)
            
            batch_data.predict.sum().backward()
            
            g = batch_data.seq.grad
            g1 = (g * batch_data.seq).sum(-1).unsqueeze(-1)
            grads['XxE[X]'].append(g1.cpu().data.numpy())
            
            g1 = g.sum(-1).unsqueeze(-1)
            grads['XxE'].append(g1.cpu().data.numpy())
            
            g1 = batch_data.hidden.grad
            grads['H'].append(g1.cpu().data.numpy())


        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]
            
        return grads
    
    
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
