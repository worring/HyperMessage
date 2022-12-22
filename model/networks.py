import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

from torch.autograd import Variable
from model import utils 
import ipdb


class HyperMSG(nn.Module):
    def __init__(self, E, X, d, l, c, dropout, cuda):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperMSG, self).__init__()
        cuda = cuda and torch.cuda.is_available()
        
        self.hmsg1 = utils.HyperMSGConvolution(d,16)
        self.hmsg2 = utils.HyperMSGConvolution(16,c)
        self.do, self.l = dropout,l 

    def forward(self, structure, H, input_weight):
        """
        an l-layer GCN
        """
        do = self.do
        H = F.relu(self.hmsg1(structure, H, input_weight))
        H = F.dropout(H, do, training=self.training)
        H = self.hmsg2(structure, H, input_weight)      
        return F.log_softmax(H, dim=1)

class HyperMSG_multimedia(nn.Module):
    def __init__(self, E, X, d, c, dropout, cuda):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperMSG_multimedia, self).__init__()
        #d, c = args.d, args.c
        cuda = cuda and torch.cuda.is_available()

        self.hmsg1 = utils.HyperMSGConvolution(d,32)
        self.hmsg2 = utils.HyperMSGConvolution(32,16)
        self.hmsg3 = utils.HyperMSGConvolution(16,c)
        self.do = dropout

    def forward(self, structure, H, input_weight):
        """
        an l-layer GCN
        """
        do = self.do
        H = F.relu(self.hmsg1(structure, H, input_weight))
        H = F.dropout(H, do, training=self.training)
        H = F.relu(self.hmsg2(structure, H, input_weight))
        H = F.dropout(H, do, training=self.training)
        H = self.hmsg3(structure, H, input_weight)
        return torch.sigmoid(H)
