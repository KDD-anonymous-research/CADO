import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint

class reward_surrogate(nn.Module):
    def __init__(self, hidden_dim=64):
        super(reward_surrogate, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_embed = nn.Linear(2, hidden_dim)
        self.edge_embed = nn.Linear(1, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
    def forward(self, node_features):
        # node_features: [B, V, 2]
        # edge_features: [B, V, V]
        # edge_index: [2, E]
        h = self.node_embed(node_features)
        # e = self.edge_embed(edge_features[:,:,:,None])
        # e = F.relu(e)

            # Linear transformations for node update
        # Uh = self.U(h)  # B x V x H
        # # Vh = self.V(h).unsqueeze(1).expand(-1, node_features.shape[1], -1, -1)  # B x V x V x H
       
        # # Linear transformations for edge update and gating
        # Ah = self.A(h)  # B x V x H, source
        # Bh = self.B(h)  # B x V x H, target
        # Ce = self.C(e)  # B x V x V x H / E x H

        # # Update edge features and compute edge gates
        # e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H


        # gates = torch.sigmoid(e)  # B x V x V x H / E x H

        # # Update node features
        # h = Uh + (gates * Vh).sum(dim=2)  # B x V x H
        # x = node_features + edge_features
        # h = F.relu(self.linear1(h))
        # h = F.relu(self.linear2(h))
        h = self.linear1(h)
        h = self.linear3(h)
        h = h.mean(dim=1)
        return h
