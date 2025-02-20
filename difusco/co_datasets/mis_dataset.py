"""MIS (Maximal Independent Set) dataset."""

import glob
import os
import sys
if int(sys.version.split('.')[1])<9:
  import pickle5 as pickle
else:
  import pickle
import numpy as np
import torch
import re
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch as GraphBatch

class MISDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, data_label_dir=None, num_testset=-1, shuffle=True):
    self.data_file = data_file
    self.file_lines = glob.glob(data_file)
    self.shuffle=shuffle
    if num_testset >0 :
      self.file_lines = self.file_lines[:num_testset]
    # self.file_lines = self.file_lines[:100] 

    self.data_label_dir = data_label_dir
    print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    with open(self.file_lines[idx], "rb") as f:
      graph = pickle.load(f)
    num_nodes = graph.number_of_nodes()

    match = re.search(r'_m(\d+)', self.file_lines[idx])
    if match:
      reward = - int(match.group(1))
    else:
      try:
        reward = - sum([node_data['label'] for node_data in graph.nodes.values()])
      except:
        reward = 0
    if self.data_label_dir is None:
      node_labels = [_[1] for _ in graph.nodes(data='label')]
      if node_labels is not None and node_labels[0] is not None:
        node_labels = np.array(node_labels, dtype=np.int64)
      else:
        node_labels = np.zeros(num_nodes, dtype=np.int64)
    else:
      base_label_file = os.path.basename(self.file_lines[idx]).replace('.gpickle', '_unweighted.result')
      node_label_file = os.path.join(self.data_label_dir, base_label_file)
      with open(node_label_file, 'r') as f:
        node_labels = [int(_) for _ in f.read().splitlines()]
      node_labels = np.array(node_labels, dtype=np.int64)
      assert node_labels.shape[0] == num_nodes

    edges = np.array(graph.edges, dtype=np.int64)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    # add self loop
    self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
    edges = np.concatenate([edges, self_loop], axis=0)
    edges = edges.T

    return num_nodes, node_labels, edges, reward

  def __getitem__(self, idx):
    num_nodes, node_labels, edge_index, reward = self.get_example(idx)
    graph_data = GraphData(x=torch.from_numpy(node_labels),
                           edge_index=torch.from_numpy(edge_index))
    graph_data.edge_length = graph_data.edge_index.shape[-1]
    point_indicator = np.array([num_nodes], dtype=np.int64)
    return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        graph_data,
        torch.from_numpy(point_indicator).long(),
        reward
    )


class MIS_ERGraphEnvironment:
    def __init__(self, lower_bound=700, upper_bound=800, p=0.15):
        """
        Args:
            lower_bound (int): 각 그래프의 최소 노드 수 (default: 700)
            upper_bound (int): 각 그래프의 최대 노드 수 (default: 800)
            p (float): ER 그래프 생성 시, 노드 쌍 간 edge 생성 확률 (default: 0.15)
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p = 1-np.sqrt(1-p)

    def get_batch(self, batch_size):
        """
        배치 크기만큼의 ER 그래프를 생성하여 아래 4가지 값을 반환합니다.
          a = None
          b = 배치된 그래프 데이터 (예, DataBatch) 
              - 예시: b.x.shape = [총 노드수, feature_dim] (여기서는 feature_dim=1)
              -     b.edge_index.shape = [2, 총 edge 수]
          c = 각 ER 그래프의 노드 수를 담은 텐서 (예, [num_nodes_graph1, num_nodes_graph2, ...])
          d = None

        Returns:
            tuple: (None, batch_data, num_nodes_tensor, None)
        """
        graph_list = []
        num_nodes_list = []

        for _ in range(batch_size):
            # 700 ~ 800 사이의 노드 수 선택 (양 끝 포함)
            n = torch.randint(low=self.lower_bound, high=self.upper_bound + 1, size=(1,)).item()
            num_nodes_list.append(n)

            # 노드 feature 생성: 여기서는 각 노드에 대해 단순히 1값 (또는 상수)을 부여 (feature dimension = 1)
            x = torch.zeros(n,  dtype=torch.float)

            # ER 그래프 생성:
            # 모든 ordered pair (i, j)에 대해, 
            #   - i == j 인 경우 : self loop이므로 반드시 edge 포함
            #   - i != j 인 경우 : 확률 p (0.15)로 edge 포함
            # 먼저 n x n 크기의 난수 행렬을 생성한 후, 각 원소가 p 미만이면 True로 간주합니다.
            rand_matrix = torch.rand(n, n)
            # self loop를 위한 대각행렬 (모든 대각원소는 True)
            self_loops = torch.eye(n, dtype=torch.bool)
            # 최종 mask: self loop거나, 난수 < p 인 경우
            mask = ((rand_matrix < self.p)+(rand_matrix.T < self.p) + self_loops)>0
          
            # mask에서 True인 인덱스를 추출하면 edge_index (shape: [num_edges, 2])가 됩니다.
            # PyG에서는 edge_index의 shape을 [2, num_edges]로 사용하므로 transpose합니다.
            edge_index = mask.nonzero(as_tuple=False).t().contiguous()

            # 각 그래프 데이터를 Data 객체로 생성 (TSPGraphEnvironment의 GraphData 생성과 유사)
            data = GraphData(x=x, edge_index=edge_index, edge_length=edge_index.shape[-1])
            graph_list.append(data)

        # 여러 Data 객체를 Batch로 묶습니다.
        batch_data = GraphBatch.from_data_list(graph_list)

        # 각 그래프의 노드 수를 담은 텐서 생성
        num_nodes_tensor = torch.tensor(num_nodes_list, dtype=torch.long)

        # 반환 순서는 (a, b, c, d) : a와 d는 None, b는 배치된 Data, c는 각 그래프의 노드 수
        return None, batch_data, num_nodes_tensor, None
    

# 예시: 배치 크기 4인 경우
if __name__ == '__main__':
    env = MIS_ERGraphEnvironment()
    a, batch_data, num_nodes_tensor, d = env.get_batch(batch_size=4)
    
    print("a:", a)
    print("batch_data.x.shape:", batch_data.x.shape)           # 전체 노드 수 (예: torch.Size([3060, 1]))
    print("batch_data.edge_index.shape:", batch_data.edge_index.shape)  # 전체 edge 수 (예: torch.Size([2, 354914]))
    print("num_nodes_tensor:", num_nodes_tensor)                # 각 ER 그래프의 노드 수
    print("d:", d)