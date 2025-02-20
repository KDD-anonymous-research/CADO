"""Lightning module for training the DIFUSCO TSP model."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from lightning.pytorch.utilities import rank_zero_info

from difusco.co_datasets.tsp_graph_dataset import TSPGraphDataset
from difusco.pl_meta_model import COMetaModel
from difusco.utils.diffusion_schedulers import InferenceSchedule
from difusco.utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours, make_tour_to_graph
import time
# import pdb
# from pytorch_memlab import profile, profile_every, set_target_gpu

class TSPModel(COMetaModel):
    def __init__(self,
                             param_args=None):
        super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)
        if self.relabel_epoch > 0 :
            preload = True
        else :
            preload= False
        if self.args.use_env == False :
            if self.optimality_dropout:
                # if self.args.training_split:
                
                self.train_dataset = TSPGraphDataset(
                    data_file=os.path.join(self.args.storage_path, self.args.training_split), num_testset=param_args.num_trainset,
                    sparse_factor=self.args.sparse_factor,preload=preload,optimality_dropout_rate=self.args.optimality_dropout_rate,
                )

                
                # self.subopt_train_dataset = TSPGraphDataset(
                #         data_file=os.path.join(self.args.storage_path, self.args.training_split),
                #         sparse_factor=self.args.sparse_factor,preload=preload,opt=False,
                # )
                #self.cost_mean = self.train_dataset.cost_mean
                #self.cost_std = self.train_dataset.cost_std
            else :
                if self.args.training_split:
                    print(' self.args.training_split', self.args.training_split)
                    print('num_trainset', param_args.num_trainset)

                    self.train_dataset = TSPGraphDataset(
                        data_file=os.path.join(self.args.storage_path, self.args.training_split),num_testset=param_args.num_trainset,
                        sparse_factor=self.args.sparse_factor,preload=preload,
                    )
            #self.cost_mean = self.train_dataset.cost_mean
            #self.cost_std = self.train_dataset.cost_std
        self.test_dataset = TSPGraphDataset(
                data_file=os.path.join(self.args.storage_path, self.args.test_split),
                sparse_factor=self.args.sparse_factor,num_testset=self.args.num_testset
        )
        print('load test dataset', 'cost_mean', self.test_dataset.cost_mean, 'cost_std', self.test_dataset.cost_std, 'dataset size', len(self.test_dataset))
        # exit(-1)
        # self.validation_dataset = TSPGraphDataset(
        #         data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        #         sparse_factor=self.args.sparse_factor,
        # )
         
    def forward(self, x, adj, t, edge_index, aux=False):

        return self.model(x, t, adj, edge_index)

    # @profile
    # @profile_every(1)
    def categorical_training_step(self, batch, batch_idx):
        # gpu_id = self.args.gpu_id[0]
        # set_target_gpu(gpu_id)
        edge_index = None
        if not self.sparse:
            _, points, adj_matrix, _, cost = batch # 5개 
            # import pdb
            # pdb.set_trace()
            t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
        else:
            _, graph_data, point_indicator, edge_indicator, _,cost= batch # 6개
            t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))

        # print(points[0][0][0])
        # Sample from diffusion
        adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).to(torch.float64)
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

        # # Denoise
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.to(self.device)
        x0_pred = self.forward(points.float().to(self.device), xt.float().to(self.device), t.float().to(self.device), edge_index,)

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred[0], adj_matrix.long().to(self.device))

        # self.log("train/loss", loss)
        return loss

    def gaussian_training_step(self, batch, batch_idx):
        if self.sparse:
            # TODO: Implement Gaussian diffusion with sparse graphs
            raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
        _, points, adj_matrix, _,cost = batch

        adj_matrix = adj_matrix * 2 - 1
        adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
        # Sample from diffusion
        t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
        xt, epsilon = self.diffusion.sample(adj_matrix, t)

        t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
        # Denoise
        epsilon_pred = self.forward(
                points.float().to(adj_matrix.device),
                xt.float().to(adj_matrix.device),
                t.float().to(adj_matrix.device),
                None,
        )
        epsilon_pred = epsilon_pred.squeeze(1)

        # Compute loss
        loss = F.mse_loss(epsilon_pred, epsilon.float())
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):

        if self.diffusion_type == 'gaussian':
            return self.gaussian_training_step(batch, batch_idx)
        elif self.diffusion_type == 'categorical':
            if self.args.two_opt_target:
                return self.categorical_training_step_two_opt_target(batch, batch_idx)
            else:
                return self.categorical_training_step(batch, batch_idx)


    def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None, classifier=None, returns=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)

            if self.args.return_condition:
                # print('returns',returns)
                x0_pred_uncond = self.forward(
                    points.float().to(device),
                    (xt).float().to(device),
                    t.float().to(device),
                    edge_index.long().to(device) if edge_index is not None else None,
                    returns = returns.to(device),
                    use_dropout=False,
                    force_dropout = True,
                    opt_dropout=None
                )
                x0_pred_cond = self.forward(
                        points.float().to(device),
                        (xt).float().to(device),
                        t.float().to(device),
                        edge_index.long().to(device) if edge_index is not None else None,
                        returns = returns.to(device),
                        use_dropout=False,
                        force_dropout = False,
                        opt_dropout=None
                )
                x0_pred = x0_pred_uncond + self.condition_guidance_w * (x0_pred_cond - x0_pred_uncond)
            else: # unconditional
                x0_pred = self.forward(
                        points.float().to(device),
                        (xt).float().to(device),
                        t.float().to(device),
                        edge_index.long().to(device) if edge_index is not None else None,
                )

            if not self.sparse:
                x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            else:
                x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)

            return xt #,xt_prob


    # def categorical_denoise_step_240129(self, points, xt, t, device, edge_index=None, target_t=None, classifier=None):
    #     """Before comebine denoise
    #     """
    #     with torch.no_grad():
    #         t = torch.from_numpy(t).view(1)
    #         x0_pred = self.forward(
    #                 points.float().to(device),
    #                 (xt).float().to(device),
    #                 t.float().to(device),
    #                 edge_index.long().to(device) if edge_index is not None else None,
    #         )
    #         if not self.sparse:
    #             x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
    #         else:
    #             x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)
    #         # print(x0_pred_prob.shape)
    #         # print(x0_pred_prob.shape)
    #         xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
    #         # print(xt.shape)
    #         return xt#,xt_prob


    def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None,returns=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            #without return
            # if not (returns==None):
            if self.args.return_condition:
                epsilon_pred_uncond = self.forward(
                points.float().to(device),
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
                returns = returns.to(device),
                use_dropout=False,
                force_dropout = True,
        )
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
            else:
                pred = self.forward(
                    points.float().to(device),
                    xt.float().to(device),
                    t.float().to(device),
                    edge_index.long().to(device) if edge_index is not None else None,
            )
            pred = pred.squeeze(1)
            xt = self.gaussian_posterior(target_t, t, pred, xt)
            return xt
        

    # def gaussian_denoise_step_240129(self, points, xt, t, device, edge_index=None, target_t=None):
    #     with torch.no_grad():
    #         t = torch.from_numpy(t).view(1)
    #         pred = self.forward(
    #                 points.float().to(device),
    #                 xt.float().to(device),
    #                 t.float().to(device),
    #                 edge_index.long().to(device) if edge_index is not None else None,
    #         )
    #         pred = pred.squeeze(1)
    #         xt = self.gaussian_posterior(target_t, t, pred, xt)
    #         return xt
        
    def test_step(self, batch, batch_idx, split='test'):

        return self.denoise_test_step(batch, batch_idx, split)

    # @profile
    # @profile_every(1)
    def denoise_test_step(self, batch, batch_idx, split='test'):
        # gpu_id = self.args.gpu_id[0]
        # set_target_gpu(gpu_id)
        edge_index = None
        np_edge_index = None
        device = self.device
        if not self.sparse:
            real_batch_idx, points, adj_matrix, gt_tour, cost = batch
            np_points = points.cpu().numpy()
            np_gt_tour = gt_tour.cpu().numpy()
            batch_size = points.shape[0]
        else:
            real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour,cost = batch
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x 
            edge_index = graph_data.edge_index 
            num_edges = edge_index.shape[1] # batch * sparse * 100
            batch_size = point_indicator.shape[0]    # batch
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size)) # [batch, sparse*100]
            points = points.reshape((-1, 2)) # [batch * 100, 2]
            edge_index = edge_index.reshape((2, -1)) # [2, batch * sparse * 100]
            np_points = points.cpu().numpy()    # [batch * 100, 2]
            np_gt_tour = gt_tour.cpu().numpy().reshape(-1) # [batch * (100 + 1)]
            np_edge_index = edge_index.cpu().numpy() # [2, batch * sparse * 100]
            print('np_gt_tour', np_gt_tour.shape, 'np_edge_index', np_edge_index.shape, 'np_points', np_points.shape )
        
        # pdb.set_trace()

        if self.return_condition:
            if self.args.target_opt_value:
                cost = self.cost_normalize(cost)
            else:
                cost = - torch.ones([points.shape[0]], device=points.device).unsqueeze(-1)
            
        unsolved_tours_list, solved_tours_list = [], []
        ns, merge_iterations = 0, 0

        # if self.args.parallel_sampling > 1:
        #     if not self.sparse:
        #         points = points.repeat(self.args.parallel_sampling, 1, 1) # [batch*parallel_sampling, 100, 2]
        #     else:
        #         points = points.repeat(self.args.parallel_sampling, 1)    # [batch*parallel_sampling*100, 2]
        #         edge_index_temp = cp.deepcopy(edge_index)
        #         edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device) # [2, batch*parallel_sampling*100]
                
        # pdb.set_trace() # points, 
        
        tsp_solvers = [] # generate tsp_solvers
        for batch_idx in range(batch_size): 
            tsp_solvers.append(TSPEvaluator(np_points.reshape([batch_size, -1, 2])[batch_idx], batch=True))

        for _ in range(self.args.sequential_sampling):
            xt = torch.randn_like(adj_matrix.float()) # (sparse) xt = [batch, sparse*100]
                
                #    print(xt.shape)
            if self.diffusion_type == 'gaussian':
                xt.requires_grad = True
            else:
                xt = (xt > 0).long()

            if self.sparse:
                xt = xt.reshape(-1)

            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                                                                T=self.diffusion.T, inference_T=steps)

            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                #t1 = (np.ones(points.shape[0])*t1).astype(int)
                t2 = np.array([t2]).astype(int)

                if self.diffusion_type == 'gaussian':
                    xt = self.gaussian_denoise_step(
                            points, xt, t1, device, edge_index, target_t=t2, returns=cost)
                else:
                    xt = self.categorical_denoise_step(
                            points, xt, t1, device, edge_index, target_t=t2, returns=cost)
                    
            if self.diffusion_type == 'gaussian':
                adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
            else:
                adj_mat = xt.float().cpu().detach().numpy() + 1e-6
            
            if self.args.save_numpy_heatmap:
                self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)
        
            tours, merge_iterations, _ = merge_tours(adj_mat, np_points, np_edge_index,
                    sparse_graph=self.sparse, batch_size=batch_size, guided=True,tsp_decoder=self.args.tsp_decoder) 
            # (dense) adj_mat=[batch, 100, 100] np_points = [batch, 100, 2], [batch, 100 + 1] (double-list)
            # (sparse) adj_mat = [batch*sparse*parallel*100], np_points= [batch*100, 2], np_edge_index = [2, sparse*batch*100]
                
            # pdb.set_trace()

            if batch_size == 1:
                if not self.sparse:
                    np_points_single=np_points[0]
                else:
                    np_points_single=np_points
                tsp_solver = TSPEvaluator(np_points_single)
                wo_2opt_costs = tsp_solver.evaluate(tours[0])

                # Refine using 2-opt
                solved_tours, ns = batched_two_opt_torch(
                        np_points_single.astype("float64"),
                        np.array(tours).astype("int64"),
                        max_iterations=self.args.two_opt_iterations,
                        device=device,
                )

                solved_tours_list.append(solved_tours)
            else:
                unsolved_tours = torch.tensor(tours)
                unsolved_tours_list.append(unsolved_tours) # [1, 8, 100 + 1]

                np_points_reshape = np_points.reshape([batch_size, -1, 2])
                # tours_reshape = tours.reshape([batch_size, -1])

                for idx in range(batch_size):
                    solved_tours, ns = batched_two_opt_torch(
                            np_points_reshape[idx].astype("float64"),
                            np.array([tours[idx]]).astype("int64"),
                            max_iterations=self.args.two_opt_iterations,
                            device=device,
                            # batch=True,
                    )
                    solved_tours_list.append(solved_tours) # [1, 8, 100 + 1]
        # print("solved_tours_list",solved_tours_list)
        # pdb.set_trace() # unsolved_tours_list, solved_tours_list
        if batch_size==1:
            solved_tours = np.concatenate(solved_tours_list, axis=0)
            gt_cost = tsp_solver.evaluate(np_gt_tour)

            total_sampling = (
                    self.args.parallel_sampling * self.args.sequential_sampling
            )
            all_solved_costs = [
                    tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)
            ]
            best_solved_cost = np.min(all_solved_costs)

        else:
            unsolved_tours_list = np.concatenate(unsolved_tours_list, axis=0)
            unsolved_tours_list = torch.tensor(unsolved_tours_list)
            unsolved_tours_list = unsolved_tours_list.view(self.args.sequential_sampling, batch_size, -1)
            
            solved_tours_list = np.concatenate(solved_tours_list, axis=0)
            solved_tours_list = torch.tensor(solved_tours_list)
            solved_tours_list = solved_tours_list.view(self.args.sequential_sampling, batch_size, -1)
            gt_costs, best_unsolved_costs, best_solved_costs = [], [], []

            # pdb.set_trace()
            for batch_idx in range(batch_size):
                tsp_solver = tsp_solvers[batch_idx]
                gt_costs.append(tsp_solver.evaluate(gt_tour[batch_idx].unsqueeze(0)))
                best_unsolved_costs.append(tsp_solver.evaluate(unsolved_tours_list[:,batch_idx]).min())
                best_solved_costs.append(tsp_solver.evaluate(solved_tours_list[:,batch_idx]).min())

            gt_cost = torch.tensor(gt_costs).mean()
            wo_2opt_costs = torch.tensor(best_unsolved_costs).mean()
            best_solved_cost = torch.tensor(best_solved_costs).mean()
        # pdb.set_trace()
        # pdb.set_trace()
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


    def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
        if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
            raise NotImplementedError("Save numpy heatmap only support single sampling")
        exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
        heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
        rank_zero_info(f"Saving heatmap to {heatmap_path}")
        os.makedirs(heatmap_path, exist_ok=True)
        real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
        np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
        np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')
