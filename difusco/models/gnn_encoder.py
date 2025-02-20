import functools
import math

import torch
import torch.nn.functional as F
from torch import nn

from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint
from torch.distributions import Bernoulli
from difusco.models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
import copy as cp

class LoRALinear(nn.Module):
    def __init__(self, module, in_features, out_features, rank, bias=False):
        super(LoRALinear, self).__init__()
        # self.pretrained = nn.Linear(in_features, out_features, bias=bias)
        # self.pretrained.weight = nn.Parameter(module.weight.detach().clone())
        # self.pretrained.bias = nn.Parameter(module.bias.detach().clone())
        # self.pretrained.weight.requires_grad = False # freeze the weights
        # self.pretrained.bias.requires_grad = False # freeze the bias
        self.pretrained = module

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        if bias:
            self.bias = nn.Linear(out_features, bias=True)
        else:
            pass
        nn.init.normal_(self.lora_down.weight, std=1)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.pretrained(x) + 1/self.rank * self.lora_up(self.lora_down(x))
    

def AddLora(model, config):
    def set_nested_attr(obj, attr_name, value):
      """
      model.model.name 이런식으로 ㅅ팅하려고 
      """
      attrs = attr_name.split('.')
      inner_obj = obj
      for attr in attrs[:-1]:
          inner_obj = getattr(inner_obj, attr)
      setattr(inner_obj, attrs[-1], value)
    changed_modules = []

    for name, module in model.named_modules():
        # if isinstance(module, nn.Linear) and not ('time_embed' in name):
        if config.lora_range==0:
          if isinstance(module, nn.Linear) and ('layers' in name) and not ('time_embed' in name):
              in_features = module.in_features
              out_features = module.out_features
              rank = config.lora_rank
              bias = False
              lora_module = LoRALinear(module, in_features, out_features, rank, bias)
              lora_module = lora_module.to(module.weight.device)
              changed_modules.append((name, lora_module))
        elif config.lora_range==1:
          # new_lora = False
          if not config.lora_new:
          # try:
            if isinstance(module, nn.Linear) and ('layer' in name) and not ('time_embed' in name): 
                in_features = module.in_features
                out_features = module.out_features
                rank = config.lora_rank
                bias = False
                lora_module = LoRALinear(module, in_features, out_features, rank, bias)
                lora_module = lora_module.to(module.weight.device)
                changed_modules.append((name, lora_module))
          else:
          # except:
            num_layers = len(model.model.layers)
            # print('num_layers',num_layers)
            name_list_lora = ['layers', 'per_layer_out'] # lora updated lists
            if isinstance(module, nn.Linear):
              name_splits = name.split('.')
              for i in range(len(name_splits)-1):
                if name_splits[i] in name_list_lora and int(name_splits[i+1]) < num_layers - config.last_train:
                # if name_splits[i] in name_list_lora: 
                  in_features = module.in_features
                  out_features = module.out_features
                  rank = config.lora_rank
                  bias = False
                  lora_module = LoRALinear(module, in_features, out_features, rank, bias)
                  lora_module = lora_module.to(module.weight.device)
                  changed_modules.append((name, lora_module))
        elif config.lora_range==2:
          if isinstance(module, nn.Linear) and not ('time_embed' in name):
              in_features = module.in_features
              out_features = module.out_features
              rank = config.lora_rank
              bias = False
              lora_module = LoRALinear(module, in_features, out_features, rank, bias)
              lora_module = lora_module.to(module.weight.device)
              changed_modules.append((name, lora_module))
              
    for name, lora_module in changed_modules:
        # setattr(model, name, lora_module)
        set_nested_attr(model, name, lora_module)

    return model

class GNNLayer(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(GNNLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"

    self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h, e, graph, mode="residual", edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          h: Input node features (B x V x H)
          e: Input edge features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          mode: str
        In Sparse version:
          h: Input node features (V x H)
          e: Input edge features (E x H)
          graph: torch_sparse.SparseTensor
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Updated node and edge features
    """
    # import pdb
    # pdb.set_trace()
    if not sparse:
      batch_size, num_nodes, hidden_dim = h.shape
    else:
      batch_size = None
      num_nodes, hidden_dim = h.shape
    h_in = h
    e_in = e

    # Linear transformations for node update
    Uh = self.U(h)  # B x V x H

    if not sparse:
      Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H
    else:
      Vh = self.V(h[edge_index[1]])  # E x H

    # Linear transformations for edge update and gating
    Ah = self.A(h)  # B x V x H, source
    Bh = self.B(h)  # B x V x H, target
    Ce = self.C(e)  # B x V x V x H / E x H

    # Update edge features and compute edge gates
    if not sparse:
      e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
    else:
      e = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H

    # Update node features
    h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=sparse)  # B x V x H

    # Normalize node features
    if not sparse:
      h = self.norm_h(
          h.view(batch_size * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
    else:
      h = self.norm_h(h) if self.norm_h else h

    # Normalize edge features
    if not sparse:
      e = self.norm_e(
          e.view(batch_size * num_nodes * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e
    else:
      e = self.norm_e(e) if self.norm_e else e

    # Apply non-linearity
    h = F.relu(h)
    e = F.relu(e)

    # Make residual connection
    if mode == "residual":
      h = h_in + h
      e = e_in + e

    return h, e

  def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
        In Sparse version:
          Vh: Neighborhood features (E x H)
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    Vh = gates * Vh  # B x V x V x H

    # Enforce graph structure through masking
    # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

    # Aggregate neighborhood features
    if not sparse:
      if (mode or self.aggregation) == "mean":
        return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
      elif (mode or self.aggregation) == "max":
        return torch.max(Vh, dim=2)[0]
      else:
        return torch.sum(Vh, dim=2)
    else:
      sparseVh = SparseTensor(
          row=edge_index[0],
          col=edge_index[1],
          value=Vh,
          sparse_sizes=(graph.size(0), graph.size(1))
      )

      if (mode or self.aggregation) == "mean":
        return sparse_mean(sparseVh, dim=1)

      elif (mode or self.aggregation) == "max":
        return sparse_max(sparseVh, dim=1)

      else:
        return sparse_sum(sparseVh, dim=1)


class PositionEmbeddingSine(nn.Module):
  """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    y_embed = x[:, :, 0]
    x_embed = x[:, :, 1]
    if self.normalize:
      # eps = 1e-6
      y_embed = y_embed * self.scale
      x_embed = x_embed * self.scale

    # dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device, requires_grad=True)
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
    return pos


class ScalarEmbeddingSine(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return pos_x


class ScalarEmbeddingSine1D(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x


def run_sparse_layer(layer, time_layer, out_layer, adj_matrix, edge_index, add_time_on_edge=True):
  def custom_forward(*inputs):
    x_in = inputs[0]
    e_in = inputs[1]
    time_emb = inputs[2]
    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
    if add_time_on_edge:
      e = e + time_layer(time_emb)
    else:
      x = x + time_layer(time_emb)
    x = x_in + x
    e = e_in + out_layer(e)
    return x, e
  return custom_forward


class GNNEncoder(nn.Module):
  """Configurable GNN Encoder
  """
  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False, sparse_factor=-1, return_condition=False, condition_dropout=0.1,
               *args, **kwargs):
    super(GNNEncoder, self).__init__()
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.hidden_dim = hidden_dim
    self.return_condition = return_condition
    if self.return_condition :
      time_embed_dim = hidden_dim // 2
    else :
      time_embed_dim = hidden_dim // 2
    
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    self.condition_dropout = condition_dropout
    self.sparse_factor = sparse_factor

    if not node_feature_only:
      self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
      self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    else:
      self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    if self.return_condition:
      return_embed_dim = time_embed_dim

      
      self.returns_mlp = nn.Sequential(
                  nn.Linear(1, return_embed_dim),
                  nn.ReLU(),
                  nn.Linear(return_embed_dim, return_embed_dim * 4),
                  nn.ReLU(),
                  nn.Linear(return_embed_dim * 4, return_embed_dim),
              )
      self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
      time_embed_dim = return_embed_dim + time_embed_dim
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    
    
    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  def generate_layer_aux(self, num_aux=1):
    """
    Generate some several aux layers to learn aux tasks
    """
    self.num_aux = num_aux
    self.layer_aux = nn.ModuleList([])
    self.per_layer_out_aux = nn.ModuleList([])
    for i in range(num_aux):
      self.layer_aux.append(cp.deepcopy(self.layers[-self.num_aux+i]))
      self.per_layer_out_aux.append(cp.deepcopy(self.per_layer_out[-self.num_aux+i]))
    self.out_aux = cp.deepcopy(self.out)
  
    # print('after self.layer_aux.device',self.layer_aux.device)
    # initialize_weights(self.layer_aux, False)
    # initialize_weights(self.per_layer_out_aux, False)
    # initialize_weights(self.out_aux, False)
        
    # self.per_layer_out의 마지막 Sequential 복사 및 추가
  def dense_forward(self, x, graph, timesteps, edge_index=None, returns=None,use_dropout=True,force_dropout=False, opt_dropout=None):
    """
    Args:
        x: Input node coordinates (B x V x 2)
        graph: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
        edge_index: Edge indices (2 x E)
    Returns:
        Updated edge features (B x V x V)
    """
    # with torch.autograd.set_detect_anomaly(True):
    del edge_index
    x = self.node_embed(self.pos_embed(x))   
    e = self.edge_embed(self.edge_pos_embed(graph))
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    
    if self.return_condition:
        assert returns is not None
        returns_embed = self.returns_mlp(returns)
        if use_dropout:
            mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
            returns_embed = mask*returns_embed
        if opt_dropout is not None :
          returns_embed = opt_dropout*returns_embed
        if force_dropout:
            returns_embed = 0*returns_embed
        if time_emb.shape[0] != returns_embed.shape[0] :
          time_emb=time_emb.repeat(returns_embed.shape[0],1)
        time_emb = torch.cat([time_emb, returns_embed], dim=-1)

    # time_emb.requires_grad = True
    graph = torch.ones_like(graph).long()
    # graph = torch.ones_like(graph).float()
    # graph.requires_grad = True
    count = 0
    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      if self.aux and count == len(self.layers)-self.num_aux:
        x_aux, e_aux = x.clone().detach(), e.clone().detach()
        # x_aux, e_aux = x.clone(), e.clone()

      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError

      x, e = layer(x, e, graph, mode="direct")
      if not self.node_feature_only:
        e = e + time_layer(time_emb)[:, None, None, :]
      else:
        x = x + time_layer(time_emb)[:, None, :]
      x = x_in + x
      e = e_in + out_layer(e)
      count +=1
    e = self.out(e.permute((0, 3, 1, 2)).contiguous())
    
    if self.aux:
      for i in range(self.num_aux):
        x_in_aux, e_in_aux = x_aux, e_aux
        x_aux, e_aux = self.layer_aux[i](x_aux, e_aux, graph, mode="direct")

        if not self.node_feature_only:
          e_aux = e_aux + self.time_embed_layers[-self.num_aux+i](time_emb)[:, None, None, :]
        else:
          x_aux = x_aux + self.time_embed_layers[-self.num_aux+i](time_emb)[:, None, :]
        x_aux = x_in_aux + x_aux
        e_aux = e_in_aux + self.per_layer_out_aux[i](e_aux)

      e_aux = self.out_aux(e_aux.permute(0, 3, 1, 2).contiguous())
      return e, e_aux
    else:
      return e, None
  def sparse_forward(self, x, graph, timesteps, edge_index, returns=None,use_dropout=True,force_dropout=False, opt_dropout=None):
    """
    Args:
        x: Input node coordinates (V x 2)
        graph: Graph edge features (E)
        timesteps: Input edge timestep features (E)
        edge_index: Adjacency matrix for the graph (2 x E)
    Returns:
        Updated edge features (E x H)
    """
    if opt_dropout:
      raise Exception("opt_dropout is not done in sparse")
    # Embed edge features
    x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
    e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    if self.return_condition:
        assert returns is not None
        returns_embed = self.returns_mlp(returns)
        if use_dropout:
            mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
            returns_embed = mask*returns_embed
        if force_dropout:
            returns_embed = 0*returns_embed
        if time_emb.shape[0] != e.shape[0] :
          time_emb=time_emb.repeat(returns_embed.shape[0],1)
          time_emb = torch.cat([time_emb, returns_embed], dim=-1) 
          time_emb = time_emb.unsqueeze(1).repeat(1,e.shape[0]//time_emb.shape[0],1).reshape(-1,time_emb.shape[-1]) #B*feat -> B*e_dim*feat ->Be_dim*feat
        else :
          returns_embed = returns_embed.unsqueeze(1).repeat(1,e.shape[0]//returns_embed.shape[0],1).reshape(-1,returns_embed.shape[-1])
          time_emb = torch.cat([time_emb, returns_embed], dim=-1) 
          
        
    edge_index = edge_index.long()
    x, e, x_aux, e_aux = self.sparse_encoding(x, e, edge_index, time_emb)
    e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
    e = self.out(e).reshape(-1, edge_index.shape[1]).permute((1, 0))

    if e_aux is not None:
      e_aux = e_aux.reshape((1, x.shape[0], -1, e_aux.shape[-1])).permute((0, 3, 1, 2))
      e_aux = self.out_aux(e_aux).reshape(-1, edge_index.shape[1]).permute((1, 0))

    return e, e_aux 

  def sparse_forward_node_feature_only(self, x, timesteps, edge_index):
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e, x_aux, e_aux = self.sparse_encoding(x, e, edge_index, time_emb)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))

    if x_aux is not None:
      x_aux = x_aux.reshape((1, x_shape[0], -1, x_aux.shape[-1])).permute((0, 3, 1, 2))
      x_aux = self.out_aux(x_aux).reshape(-1, x_shape[0]).permute((1, 0))
    
    return x, x_aux

  def sparse_encoding(self, x, e, edge_index, time_emb):
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones_like(edge_index[0].float()),
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    adj_matrix = adj_matrix.to(x.device)


    count = 0
    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      if self.aux and count == len(self.layers)-self.num_aux:
        x_aux, e_aux = x.clone().detach(), e.clone().detach()
        # x_aux, e_aux = x.clone(), e.clone()
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        single_time_emb = time_emb[:1]

        run_sparse_layer_fn = functools.partial(
            run_sparse_layer,
            add_time_on_edge=not self.node_feature_only
        )

        out = activation_checkpoint.checkpoint(
            run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
            x_in, e_in, single_time_emb
        )
        x = out[0]
        e = out[1]
      else:
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
        if not self.node_feature_only:
          e = e + time_layer(time_emb)
        else:
          x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
      count += 1

    if self.aux:
      for i in range(self.num_aux):
        x_in_aux, e_in_aux = x_aux, e_aux
        x_aux, e_aux = self.layer_aux[i](x_in_aux, e_in_aux, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
        if not self.node_feature_only:
          e_aux = e_aux + self.time_embed_layers[-self.num_aux+i](time_emb)
        else:
          x_aux = x_aux + self.time_embed_layers[-self.num_aux+i](time_emb)
        x_aux = x_in_aux + x_aux
        e_aux = e_in_aux + self.per_layer_out_aux[i](e_aux)
    else:
      x_aux, e_aux = None, None
    return x, e, x_aux, e_aux
  
  def forward(self, x, timesteps, graph=None, edge_index=None,returns=None,use_dropout=True,force_dropout=False,opt_dropout=None):
    if self.node_feature_only:
      if self.sparse:
        return self.sparse_forward_node_feature_only(x, timesteps, edge_index)
      else:
        raise NotImplementedError
    else:
      if self.sparse:
        if self.return_condition :
          return self.sparse_forward(x, graph, timesteps, edge_index,returns,use_dropout,force_dropout,opt_dropout)
        else :
            
          return self.sparse_forward(x, graph, timesteps, edge_index)
      
      else:
        if self.return_condition :
          return self.dense_forward(x, graph, timesteps, edge_index,returns,use_dropout,force_dropout,opt_dropout)
        else:
          return self.dense_forward(x, graph, timesteps, edge_index)
def initialize_weights(layer, zero=False):
  for name, m in layer.named_modules():
    if isinstance(m, nn.Linear):
        if zero:
            nn.init.zeros_(m.weight)
        else:
          nn.init.xavier_uniform_(m.weight)  # 가중치를 Xavier 초기화
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        # print('name', name)