"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch
import os
import tsplib95
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch as GraphBatch
import json

class TSPGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sparse_factor=-1,preload=False,optimality_dropout_rate=1,num_testset=-1,use_env=False):
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.file_lines = open(data_file).read().splitlines()

        if self.file_lines[-1].split(' ')[0]=='mean':
            self.cost_mean = float(self.file_lines[-1].split(' ')[1])
            self.cost_std = float(self.file_lines[-1].split(' ')[3])
            del self.file_lines[-1]

        if num_testset >0:
            self.file_lines = self.file_lines[0:num_testset]

            print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')
            print(f'cost_mean "{self.cost_mean}" cost_std {self.cost_std}')

        else:
            costs = []
            for line in self.file_lines:
                try:
                    cost = line.split(' cost ')[1]
                except:
                    print(line)
                    import pdb
                    pdb.set_trace()
                costs.append(float(cost))
            self.cost_mean = np.mean(costs)
            self.cost_std = np.std(costs)


        self.preload = preload
        self.length = len(self.file_lines)
        self.optimality_dropout_rate = optimality_dropout_rate
        
        if self.preload :
            self.data = []
            self.preload = False
            for i in range(0,self.__len__()) :
                self.data.append(list(self.get_example(i)))
            self.data_file = None
            self.file_lines = None    
            self.preload = True
    
    def __len__(self):
        return self.length

    def get_example(self, idx):
        if self.optimality_dropout_rate < 1 :
            if np.random.rand() < self.optimality_dropout_rate :
                idx = np.random.randint(0,100000)
                opt = 1
            else :
                idx = np.random.randint(100000,100000+(self.length-100000))        
                opt = 0
        if self.preload :
            points, tour, cost = self.data[idx]
            return points, tour, cost
        # Select sample
        line = self.file_lines[idx]
        # Clear leading/trailing characters
        line = line.strip()

        # Extract points
        points = line.split(' output ')[0]
        points = points.split(' ')
        points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
        # Extract tour
        tour = line.split(' output ')[1]
        if 'cost' in tour :
            cost = float(tour.split(' cost ')[1])
            tour = tour.split(' cost ')[0]
        else :
            cost = 0
        tour = tour.split(' ')
        tour = line.split(' output ')[1].split(' cost ')[0]
        tour = tour.split(' ')
        tour = np.array([int(t) for t in tour])
        tour -= 1
        if self.optimality_dropout_rate < 1:
            return points, tour, cost, opt
        else :
            
            return points, tour, cost

    def __getitem__(self, idx):
        
        if self.optimality_dropout_rate < 1:
            points, tour, cost, opt    = self.get_example(idx)
        else :
            points, tour, cost = self.get_example(idx)
            
        if self.sparse_factor <= 0:
            # Return a densely connected graph
            adj_matrix = np.zeros((points.shape[0], points.shape[0]))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
            # return points, adj_matrix, tour
            
            
            if self.optimality_dropout_rate < 1:
                return (
                    torch.LongTensor(np.array([idx], dtype=np.int64)),
                    torch.from_numpy(points).float(),
                    torch.from_numpy(adj_matrix).float(),
                    torch.from_numpy(tour).long(),
                    torch.FloatTensor([cost]),
                    torch.FloatTensor([opt]),
                )
            else :
                # print(
                #         torch.LongTensor(np.array([idx], dtype=np.int64)),
                #         torch.from_numpy(points).float(),
                #         torch.from_numpy(adj_matrix).float(),
                #         torch.from_numpy(tour).long(),
                #         torch.FloatTensor([cost]),)
                return (
                        torch.LongTensor(np.array([idx], dtype=np.int64)),
                        torch.from_numpy(points).float(),
                        torch.from_numpy(adj_matrix).float(),
                        torch.from_numpy(tour).long(),
                        torch.FloatTensor([cost]),
                )

            
        else:
            # Return a sparse graph where each node is connected to its k nearest neighbors
            # k = self.sparse_factor
            sparse_factor = self.sparse_factor
            kdt = KDTree(points, leaf_size=30, metric='euclidean')
            dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

            edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

            edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

            tour_edges = np.zeros(points.shape[0], dtype=np.int64)
            tour_edges[tour[:-1]] = tour[1:]
            tour_edges = torch.from_numpy(tour_edges)
            tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
            graph_data = GraphData(x=torch.from_numpy(points).float(),
                                                         edge_index=edge_index,
                                                         edge_attr=tour_edges)

            point_indicator = np.array([points.shape[0]], dtype=np.int64)
            edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
            
            
            
        if self.optimality_dropout_rate < 1:
            return (
                    torch.LongTensor(np.array([idx], dtype=np.int64)),
                    graph_data,
                    torch.from_numpy(point_indicator).long(),
                    torch.from_numpy(edge_indicator).long(),
                    torch.from_numpy(tour).long(),
                    torch.FloatTensor([cost]),
                    torch.FloatTensor([opt]),
            )
        else :
            return (
                    torch.LongTensor(np.array([idx], dtype=np.int64)),
                    graph_data,
                    torch.from_numpy(point_indicator).long(),
                    torch.from_numpy(edge_indicator).long(),
                    torch.from_numpy(tour).long(),
                    torch.FloatTensor([cost]),
            )

class TSPGraphEnvironment():
    def __init__(self, ins_size,sparse_factor=-1):
        self.ins_size = int(ins_size)
        self.points = torch.rand(size = [self.ins_size, 2])
        self.kdt = 0
        self.cost = 0
        self.ins_size_2d = torch.tensor([self.ins_size, self.ins_size])
        self.sparse_factor = sparse_factor
        if sparse_factor >0:
            self.sparse = True
        else :
            self.sparse = False
                    
    def get_batch(self, batch_size):
        points = torch.rand(size = [batch_size, self.ins_size, 2]).detach()
        
        if not self.sparse :
            real_batch_idx_dummy = torch.zeros(batch_size).detach()
            # ins_size_batch =self.ins_size_2d.repeat(batch_size,1)
            ins_size_batch =self.ins_size_2d.detach()
            gt_tour_dummy = torch.zeros([batch_size, self.ins_size+1]).detach()
            cost_dummy = torch.zeros(batch_size).detach()

            return real_batch_idx_dummy, points, ins_size_batch, gt_tour_dummy, cost_dummy
        else :
            sparse_factor = self.sparse_factor
            graph_data = []
            point_indicator = []
            edge_indicator = []
            for i in range(0,batch_size) :
                
                kdt = KDTree(points[i], leaf_size=30, metric='euclidean')
                _, idx_knn = kdt.query(points[i], k=sparse_factor, return_distance=True)

                edge_index_0 = torch.arange(points[i].shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
                edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

                edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

                graph_data.append(GraphData(x=points[i],
                                                            edge_index=edge_index))
                
                point_indicator.append(np.array([points[i].shape[0]], dtype=np.int64))
                edge_indicator.append(np.array([edge_index.shape[1]], dtype=np.int64))
            
            point_indicator = np.stack(point_indicator)
            edge_indicator = np.stack(edge_indicator)
            graph_data = GraphBatch.from_data_list(graph_data)
            real_batch_idx_dummy = torch.zeros(batch_size).detach()
            gt_tour_dummy = torch.zeros([batch_size, self.ins_size+1]).detach()
            cost_dummy = torch.zeros(batch_size).detach()
            
            return (
                    real_batch_idx_dummy,
                    graph_data,
                    torch.from_numpy(point_indicator).long(),
                    torch.from_numpy(edge_indicator).long(),
                    gt_tour_dummy ,
                    cost_dummy,
            )



class TSPLIBGraphDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = os.listdir(folder_path)
        self.file_names.remove("opt.json")
        self.length = len(self.file_names)
        with open(os.path.join(self.folder_path, "opt.json"), 'r') as json_file:
            self.opt = json.load(json_file)
    def __len__(self):
        return self.length
    
    def get_example(self, idx):
        problem = os.path.join(self.folder_path, self.file_names[idx])
        problem = tsplib95.load(problem)
        coord = np.array(list(problem.as_name_dict()['node_coords'].values()))
        min_value,max_value = np.min(coord), np.max(coord)
        coord_norm = (coord - min_value) / (max_value-min_value)
        # 0,0에 가장 가까운 노드 찾기
        # distances = np.linalg.norm(coord, axis=1)  # 각 좌표에서 (0, 0)까지의 거리 계산
        # closest_node_index = np.argmin(distances)  # 가장 가까운 노드의 인덱스

        # # 좌표를 가장 가까운 노드로 Shift
        # shifted_coord = coord - coord[closest_node_index]

        # # 전체 좌표에서 최소, 최대 값 찾기
        # min_value = np.min(shifted_coord)
        # max_value = np.max(shifted_coord)

        # # Min-Max 정규화 (x와 y를 동일한 기준으로)
        # coord_norm = (shifted_coord - min_value) / (max_value - min_value)
        return coord, coord_norm
    def __getitem__(self, idx):
        
        unnorm_points, norm_points = self.get_example(idx)
        instance_name = self.file_names[idx].replace('.tsp','')
        optimal_cost = self.opt[instance_name]
        adj_matrix = np.zeros((norm_points.shape[0], norm_points.shape[0]))
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            torch.from_numpy(norm_points).float(),
            torch.from_numpy(adj_matrix).float(),
            torch.zeros(1),
            torch.zeros(1),
            torch.from_numpy(unnorm_points).float(),
            torch.tensor([optimal_cost])
            
        )

# if __name__=='__main__':
#         # from ..train import args
#         storage_path = '/lab-di/squads/diff_nco/data/tsp_custom'
#         training_split = 'tsp100_train_datadist_optimal_1600_shortest_78400.txt'
#         # storage_path = '/workspace/pair/squads/diff_nco'

#         import os

#         train_dataset = TSPGraphDataset(
#                     data_file=os.path.join(storage_path, training_split),
#                     sparse_factor=-1, preload= False)
#         # batch_sampler = MyBatchSampler([[1, 2, 3], [5, 6, 7], [4, 2, 1]])
#         print(train_dataset.get_example([1,2]))
#         print(train_dataset.__getitem__(2))

#         data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=2)
#         for batch in data_loader:
#             print(batch[0])
