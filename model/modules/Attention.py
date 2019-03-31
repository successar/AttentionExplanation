import torch
import torch.nn as nn
from allennlp.common import Registrable
from Transparency.model.modelUtils import jsd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
import pickle

def masked_softmax(attn_odds, masks) :
    attn_odds.masked_fill_(masks, -float('inf'))
    attn = nn.Softmax(dim=-1)(attn_odds)
    return attn

class Attention(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")

@Attention.register('tanh')
class TanhAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        
    def forward(self, input_seq, hidden, masks) :
        #input_seq = (B, L), hidden : (B, L, H), masks : (B, L)

        attn1 = nn.Tanh()(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = masked_softmax(attn2, masks)
        
        return attn

@Attention.register('dot')
class DotAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1 = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size

    def forward(self, input_seq, hidden, masks) :
        #input_seq = (B, L), hidden = (B, L, H), masks = (B, L)
        attn1 = self.attn1(hidden) / (self.hidden_size)**0.5
        attn1 = attn1.squeeze(-1)
        attn = masked_softmax(attn1, masks)

        return attn

@Attention.register('tanh_qa')
class TanhQAAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1p = nn.Linear(hidden_size, hidden_size // 2)
        self.attn1q = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        
    def forward(self, input_seq, hidden_1, hidden_2, masks) :
        #input_seq = (B, L), hidden : (B, L, H), masks : (B, L)

        attn1 = nn.Tanh()(self.attn1p(hidden_1) + self.attn1q(hidden_2).unsqueeze(1))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = masked_softmax(attn2, masks)

        return attn

@Attention.register('dot_qa')
class DotQAAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_seq, hidden_1, hidden_2, masks) :
        #input_seq = (B, L), hidden = (B, L, H), masks = (B, L)
        attn1 = torch.bmm(hidden_1, hidden_2.unsqueeze(-1)) / self.hidden_size**0.5
        attn1 = attn1.squeeze(-1)
        attn = masked_softmax(attn1, masks)

        return attn

from collections import defaultdict
@Attention.register('logodds')
class LogOddsAttention(Attention) :
    def __init__(self, hidden_size, logodds_file:str) :
        super().__init__()
        logodds = pickle.load(open(logodds_file, 'rb'))
        logodds_combined = defaultdict(float)
        for e in logodds :
            for k, v in logodds[e].items() :
                if v is not None :
                    logodds_combined[k] += abs(v) / len(logodds.keys())
                else :
                    logodds_combined[k] = None
                    
        logodds_map = logodds_combined
        vocab_size = max(logodds_map.keys())+1
        logodds = np.zeros((vocab_size, ))
        for k, v in logodds_map.items() :
            if v is not None :
                logodds[k] = abs(v)
            else :
                logodds[k] = float('-inf')
        self.logodds = torch.Tensor(logodds).to(device)
        
        self.linear_1 = nn.Linear(hidden_size, 1)

    def forward(self, input_seq, hidden, masks) :
        #input_seq = (B, L), hidden = (B, L, H), masks = (B, L)
        attn1 = self.logodds[input_seq]
        attn = masked_softmax(attn1, masks)

        return attn

    def regularise(self, input_seq, hidden, masks, previous_attn) :
        attn = self.forward(input_seq, hidden, masks)
        js_divergence = jsd(attn, previous_attn)
        return js_divergence.mean()