"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch

from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


"""
[Remind] 
    [Normalizing Issue of PCTSP]
    locs_2d = torch.rand([batch_size, num_locs+1, 2])
    penalty = torch.rand([batch_size, num_locs+1]) * Max_LENGTHS[num_locs] * 3 /num_locs
    prize = torch.rand([batch_size, num_locs+1]) * 4 /num_locs
"""

# Max_LENGTHS = {20:2, 50:3, 100:4}


class PCTSPGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sparse_factor=-1,preload=False,optimality_dropout_rate=1):
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.file_lines = open(data_file).read().splitlines()
        self.cost_mean = float(self.file_lines[-1].split(' ')[1])
        self.cost_std = float(self.file_lines[-1].split(' ')[3])

        if self.file_lines[-1].split(' ')[0]=='mean':
            del self.file_lines[-1]

            print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')
            print(f'cost_mean "{self.cost_mean}" cost_std {self.cost_std}')

        self.preload = preload
        self.length = len(self.file_lines)
        self.optimality_dropout_rate = optimality_dropout_rate
        
    def __len__(self):
        return self.length

    def get_example(self, idx):
        if self.optimality_dropout_rate < 1 :
            raise NotImplementedError
        else:
        # Select sample
            line = self.file_lines[idx]
            # Clear leading/trailing characters
            line = line.strip()
        
            position_xys = line.split(' Penalties ')[0]
            position_xys = position_xys.split(' ')
            position_xys = np.array(position_xys).reshape([-1, 2]).astype(float)
            num_locs = len(position_xys)-1

            # Extract tour
            penalties = np.array(line.split(' Penalties ')[1].split(' Prizes ')[0].split(' ')).astype(float) 
            prizes = np.array(line.split(' Prizes ')[1].split(' Tours ')[0].split(' ')).astype(float)
            
            
            tour = np.array(line.split(' Tours ')[1].split(' Costs ')[0].split(' ')).astype(int)
            
            cost = float(line.split(' Costs ')[1].split(' Distance ')[0])

            distance = float(line.split(' Distance ')[1].split(' Penalty_Unvisited ')[0])
            prize_collected = float(line.split(' Penalty_Unvisited ')[1].split(' Prize_Collected ')[0])
            

            # penalties *= num_locs/(Max_LENGTHS[num_locs] * 3)
            # prizes *= num_locs/4

            instance = np.concatenate((position_xys.reshape([-1,2]), penalties.reshape([-1, 1]), prizes.reshape([-1, 1])), axis=1)

        return instance, tour, cost

    def __getitem__(self, idx):
        
        if self.optimality_dropout_rate < 1:
            raise NotImplementedError
            points, tour, cost, opt  = self.get_example(idx)
        else:
            
            points, tour, cost = self.get_example(idx)
            points = torch.tensor(points)

        if self.sparse_factor <= 0:
            # Return a densely connected graph
            adj_matrix = np.zeros((points.shape[0], points.shape[0]))
            # for i in range(tour.shape[0] - 1):
                # adj_matrix[tour[i], tour[i + 1]] = 1
            adj_matrix[tour[:-1], tour[1:]] = 1 
            adj_matrix[tour[1:], tour[:-1]] = 1 
            
            tour = np.append(tour, np.zeros(len(points)+1 - len(tour)))
            
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
                #         points,
                #         torch.from_numpy(adj_matrix).float(),
                #         torch.from_numpy(tour).long(),
                #         torch.FloatTensor([cost]),
                # )
                return (
                        torch.LongTensor(np.array([idx], dtype=np.int64)),
                        points,
                        torch.from_numpy(adj_matrix).float(),
                        torch.from_numpy(tour).long(),
                        torch.FloatTensor([cost]),
                )

            
        else:
            raise NotImplementedError
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


if __name__=='__main__':
        # from ..train import args
        storage_path = '/lab-di/squads/diff_nco/data/pctsp_custom'
        # training_split = 'pctsp_20_ortools_1sec_200.txt'
        training_split = 'pctsp_20_ortools_1sec_10000.txt'
        # storage_path = '/workspace/pair/squads/diff_nco'

        import os

        train_dataset = PCTSPGraphDataset(
                    data_file=os.path.join(storage_path, training_split),
                    sparse_factor=-1, preload= False, optimality_dropout_rate=1.0,
        )
        
        print(train_dataset.get_example(1))
        print(train_dataset.__getitem__(1))
        data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4)
        for batch in data_loader:
            print(batch)
