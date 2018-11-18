def isTrue(obj, attr) :
    return hasattr(obj, attr) and getattr(obj, attr)

import torch

def masked_softmax(tensor, mask, dim=-1) :
    # tensor : (x1, x2, x3, ..., xn) Tensor
    # mask : (x1, x2, x3, ..., xn) LongTensor containing 1/0 
    #        where 1 if element to be masked else 0
    # dim : dimension over which to do softmax
    tensor.masked_fill_(mask.long(), -float('inf'))
    return torch.nn.Softmax(dim=dim)(tensor)