"""Lightning module for training the DIFUSCO TSP model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours, make_tour_to_graph
from pl_tsp_model import TSPModel
import time


import pdb

class TSPModelFreeGuide(TSPModel):
  def __init__(self,
               param_args=None):
    super(TSPModelFreeGuide, self).__init__(param_args=param_args)
    self.cost_mean = self.train_dataset.cost_mean
    self.cost_std = self.train_dataset.cost_std
    self.cost_min = self.cost_mean-2*self.cost_std
    self.cost_max = self.cost_mean+2*self.cost_std
    
  def cost_normalize(self,cost) :
    #0~1
    cost = (cost - self.cost_min)/(self.cost_max-self.cost_min)
    #-1~0
    cost = cost -1
    #cost = (cost - self.cost_mean)/self.cost_std
    return cost
  def forward(self, x, adj, t, edge_index, returns, use_dropout, force_dropout):
    return self.model(x, t, adj, edge_index,returns, use_dropout, force_dropout)

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    if not self.sparse:
      _, points, adj_matrix, _,cost = batch
      t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
    else:
      _, graph_data, point_indicator, edge_indicator, _,cost= batch
      t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
    if self.return_condition :
      cost = self.cost_normalize(cost)
      
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    xt = self.diffusion.sample(adj_matrix_onehot, t)
    xt = xt * 2 - 1
    xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

    if self.sparse:
      t = torch.from_numpy(t).float()
      t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      xt = xt.reshape(-1)
      adj_matrix = adj_matrix.reshape(-1)
      points = points.reshape(-1, 2)
      edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)
    else:
      t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

    # Denoise
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
        returns = cost.to(adj_matrix.device),
        use_dropout=True,
        force_dropout = False,
    )
    
    # print(x0_pred.shape)
    # print(adj_matrix.shape)
    # Compute loss
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, adj_matrix.long())
  
    self.log("train/loss", loss)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
    _, points, adj_matrix, _, cost = batch

    if self.return_condition :
      cost = self.cost_normalize(cost)
    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
    xt, epsilon = self.diffusion.sample(adj_matrix, t)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    
    #use_dropout = train uncond/cond together
    epsilon_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index = None,
        returns = cost.to(adj_matrix.device),
        use_dropout=True,
        force_dropout = False,
    )
 
    
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss
  
  def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None, classifier=None,returns=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      # pdb.set_trace()

      # print('points', points[:5], points.size())
      # print('xt', xt[:5], xt.size())
      # print('points', points[:5], points.size())
      # print('points', points[:5], points.size())

      x0_pred_uncond = self.forward(
          points.float().to(device),
          (xt).float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
          returns = returns.to(device),
          use_dropout=False,
          force_dropout = True,
      )
      x0_pred_cond = self.forward(
          points.float().to(device),
          (xt).float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
          returns = returns.to(device),
          use_dropout=False,
          force_dropout = False,
      )
      x0_pred = x0_pred_uncond + self.condition_guidance_w*(x0_pred_cond - x0_pred_uncond)
      
      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)

      return xt#,xt_prob

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None,returns=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)

      epsilon_pred_uncond = self.forward(
        points.float().to(device),
        xt.float().to(device),
        t.float().to(device),
        edge_index.long().to(device) if edge_index is not None else None,
        returns = returns.to(device),
        use_dropout=False,
        force_dropout = True,
    )
      #without return
      epsilon_pred_cond = self.forward(
        points.float().to(device),
        xt.float().to(device),
        t.float().to(device),
        edge_index.long().to(device) if edge_index is not None else None,
        returns = returns.to(device),
        use_dropout=False,
        force_dropout=False,
      )
      pred = epsilon_pred_uncond + self.condition_guidance_w*(epsilon_pred_cond - epsilon_pred_uncond)
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt
    

  def denoise_test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    time1 = time.perf_counter()

    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour, cost = batch
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy()[0]
      batch_size = points.shape[0]
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour,cost = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      points = points.reshape((-1, 2))
      edge_index = edge_index.reshape((2, -1))
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      np_edge_index = edge_index.cpu().numpy()
    if self.return_condition :
      cost = -torch.ones([points.shape[0]],device=points.device).unsqueeze(-1)
      
    stacked_tours = []
    ns, merge_iterations = 0, 0


    time2 = time.perf_counter()
    print('time for make batch', time2-time1)
    pdb.set_trace()

    if self.args.parallel_sampling > 1:
      if not self.sparse:
        points = points.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)
    # print(points.shape)
    start_time = time.time()
    
    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)
        xt = torch.randn_like(xt)
        # print(xt.shape)
      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)
      time4 = time.perf_counter()
      pdb.set_trace()
      time5=time.perf_counter()
      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        #t1 = (np.ones(points.shape[0])*t1).astype(int)
        t2 = np.array([t2]).astype(int)

        if self.diffusion_type == 'gaussian':
          xt = self.gaussian_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2,returns=cost)
        else:
          xt = self.categorical_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2,returns=cost)
      # print(xt.shape)
      if self.diffusion_type == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6
      
      # print(adj_mat.dtype)
      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

      diffusion_time = time.time()
      time55 = time.perf_counter()
      # print("inference-time: ", diffusion_time - start_time)

      # print(len(np_edge_index[0]))
      # print(adj_mat.shape)
      
      start = time.time()
      tours, merge_iterations = merge_tours(
          adj_mat, np_points, np_edge_index,
          sparse_graph=self.sparse,
          parallel_sampling=batch_size, guided=True,
      )
      # pdb.set_trace()
      ## without 2-opt
      time6=time.perf_counter()
      pdb.set_trace()
      time7=time.perf_counter()
      if batch_size == 1:
        if not self.sparse :
          np_points=np_points[0]
        tsp_solver = TSPEvaluator(np_points)
        wo_2opt_costs = tsp_solver.evaluate(tours[0])
        # print("without_2opt : ", wo_2opt_costs)

        # Refine using 2-opt
        solved_tours, ns = batched_two_opt_torch(
            np_points.astype("float64"),
            np.array(tours).astype("int64"),
            max_iterations=10,
            device=device,
        )

        stacked_tours.append(solved_tours)

      else:
          # calculate before 2-opt cost

          tsp_solver = TSPEvaluator(np_points, batch=True)
          tours = torch.tensor(tours)
          wo_2opt_costs = tsp_solver.evaluate(tours).mean()
          # print("without_2opt : ", wo_2opt_costs)

          # Refine using 2-opt
          solved_tours, ns = batched_two_opt_torch(
              np_points.astype("float64"),
              np.array(tours).astype("int64"),
              max_iterations=10,
              device=device,
              batch=True,
          )
          stacked_tours.append(solved_tours)



    solved_tours = np.concatenate(stacked_tours, axis=0)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

    total_sampling = (
        self.args.parallel_sampling * self.args.sequential_sampling
    )
    all_solved_costs = [
        tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)
    ]
    best_solved_cost = np.min(all_solved_costs)

    time8=time.perf_counter()
    pdb.set_trace()
    metrics = {
      f"{split}/wo_2opt_cost": wo_2opt_costs,
      f"{split}/gt_cost": gt_cost,
  }
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(
      f"{split}/solved_cost",
      best_solved_cost,
      prog_bar=True,
      on_epoch=True,
      sync_dist=True,
  )
    return metrics
