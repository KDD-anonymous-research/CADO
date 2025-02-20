import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData
from torch.distributions import Categorical
from difusco.utils.cython_merge.cython_merge import merge_cython
import copy as cp
def make_tour_to_graph(points, tour, sparse_factor):
    
    sparse_factor = sparse_factor
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
    return (
            # torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
            torch.from_numpy(edge_indicator).long(),
            torch.from_numpy(tour).long(),
    )

def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu", batch=False):
    iterator = 0
    tour = tour.copy()
    with torch.inference_mode():
        if batch: 
            cuda_points = torch.from_numpy(points).to(device)
            
            cuda_tour = torch.from_numpy(tour).to(device)
            batch_size = cuda_tour.shape[0]
            
            cuda_points = cuda_points.reshape(batch_size, -1, 2)
            min_change = torch.tensor(-1.0)
            while torch.min(min_change)<0.0:
                points_i = cuda_points.gather(1, cuda_tour[:,:-1,None].repeat(1, 1, 2)).reshape(batch_size, -1, 1, 2)
                points_j = cuda_points.gather(1, cuda_tour[:,:-1,None].repeat(1, 1, 2)).reshape(batch_size, 1, -1, 2)
                points_i_plus_1 = cuda_points.gather(1, cuda_tour[:,1:,None].repeat(1, 1, 2)).reshape(batch_size, -1, 1, 2)
                points_j_plus_1 = cuda_points.gather(1, cuda_tour[:,1:,None].repeat(1, 1, 2)).reshape(batch_size, 1, -1, 2)
                
                A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
                A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
                A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
                A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

                change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
                valid_change = torch.triu(change, diagonal=2)


                min_change,_ = torch.min(valid_change.reshape(batch_size, -1), dim=-1)
                # print(valid_change.reshape(batch_size, -1)[1].topk(10, largest=False))
                flatten_argmin_index = Categorical(probs=torch.exp(-1* valid_change/0.01+1e-6).reshape(batch_size, -1)).sample()
                # print(valid_change.shape)
                # flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
                # print(flatten_argmin_index)
                min_i = torch.div(flatten_argmin_index, cuda_points.shape[1], rounding_mode='floor')
                min_j = torch.remainder(flatten_argmin_index, cuda_points.shape[1])
                for i in range(batch_size):
                    if min_change[i] < -1e-6:
                        # for i in range(batch_size):
                        cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                iterator += 1
                # else:
                #     break
                if torch.min(min_change) >= -1e-6:
                    break
                if iterator >= max_iterations:
                    break
            tour = cuda_tour.cpu().numpy()

        else:
            cuda_points = torch.from_numpy(points).to(device)
            cuda_tour = torch.from_numpy(tour).to(device)
            batch_size = cuda_tour.shape[0]
            min_change = -1.0
            while min_change < 0.0:
                points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
                points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
                points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
                points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

                A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
                A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
                A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
                A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

                change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
                valid_change = torch.triu(change, diagonal=2)
                
                min_change = torch.min(valid_change)
                flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
                min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
                min_j = torch.remainder(flatten_argmin_index, len(points))

                if min_change < -1e-6:
                    for i in range(batch_size):
                        cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                    iterator += 1
                else:
                    break

                if iterator >= max_iterations:
                    break
            tour = cuda_tour.cpu().numpy()
    return tour, iterator

def am_decoding(points, adj_mat):
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    max_val = int(np.max(dists))+1
    num_locs = len(points)

    dists += 2*max_val * np.eye(num_locs)
    init_node = np.random.randint(num_locs)
    # path = []
    # path.append(init_node)
    node = init_node
    adj_mat = - adj_mat/dists
    initial_adj = cp.deepcopy(adj_mat)
    max_adj = 1
    initial_adj += 2*max_adj * np.eye(num_locs)
    real_adj_matrix = np.zeros_like(adj_mat)
    for _ in range(num_locs-1):
        adj_mat[:, node] = max_adj
        cand_node = np.argmin(adj_mat[node])
        # path.append(int(cand_node))
        real_adj_matrix[node, cand_node] = 1
        node = cand_node
    # path.append(init_node)
    real_adj_matrix[node, init_node] = 1
    real_adj_matrix += real_adj_matrix.T
    merge_iterations = -100 # dummy value for consistency from cython_merge
    # print(path, len(path), real_adj_matrix.sum())

    return real_adj_matrix, merge_iterations

# def am_decoding(points, adj_mat):
#     dists = np.linalg.norm(points[:, None] - points, axis=-1)
#     max_val = int(np.max(dists))+1
#     num_locs = len(points)

#     dists += 2*max_val * np.eye(num_locs)
#     node = 0
#     path = []
#     path.append(node)
    
#     adj_mat = - adj_mat/dists
#     # initial_adj = cp.deepcopy(adj_mat)
#     max_adj = 1
#     adj_mat += 2*max_adj * np.eye(num_locs)
#     real_adj_matrix = np.zeros_like(adj_mat)
#     for _ in range(num_locs):
#         adj_mat[:, node] = max_adj
#         cand_node = np.argmin(adj_mat[node])
#         # path.append(int(cand_node))
#         real_adj_matrix[node, cand_node] = 1

#         node = cand_node
#     real_adj_matrix += real_adj_matrix.T
#     merge_iterations = -100 # dummy value for consistency from cython_merge
#     # print(path, len(path))

#     return real_adj_matrix, merge_iterations

def numpy_merge(points, adj_mat):
    dists = np.linalg.norm(points[:, None] - points, axis=-1)

    components = np.zeros((adj_mat.shape[0], 2)).astype(int)
    components[:] = np.arange(adj_mat.shape[0])[..., None]
    real_adj_mat = np.zeros_like(adj_mat)
    merge_iterations = 0
    for edge in (-adj_mat / dists).flatten().argsort():
        merge_iterations += 1
        a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
        if not (a in components and b in components):
            continue
        ca = np.nonzero((components == a).sum(1))[0][0]
        cb = np.nonzero((components == b).sum(1))[0][0]
        if ca == cb:
            continue
        cca = sorted(components[ca], key=lambda x: x == a)
        ccb = sorted(components[cb], key=lambda x: x == b)
        newc = np.array([[cca[0], ccb[0]]])
        m, M = min(ca, cb), max(ca, cb)
        real_adj_mat[a, b] = 1
        components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
        if len(components) == 1:
            break
    real_adj_mat[components[0, 1], components[0, 0]] = 1
    real_adj_mat += real_adj_mat.T
    return real_adj_mat, merge_iterations

def cython_merge(points, adj_mat):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
        real_adj_mat = np.asarray(real_adj_mat)
    return real_adj_mat, merge_iterations


def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot, yroot = find(parent, x), find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def tsp_minimum_weight_optimized(points, adj_mat):
    dist = np.linalg.norm(points[:, None] - points, axis=-1)
    dist[np.diag_indices_from(dist)] += 10
    A = - adj_mat / dist
    print('adj_mat',adj_mat)
    output_A = np.zeros_like(A)
    N = A.shape[0]
    parent = list(range(N))
    rank = [0] * N
    graph = [[] for _ in range(N)]
    
    heap = []
    for i in range(N):
        for j in range(i+1, N):
            heappush(heap, (A[i, j], i, j))
    
    edge_count = 0
    while heap and edge_count < N:
        weight, i, j = heappop(heap)
        x, y = find(parent, i), find(parent, j)
        if x != y or (len(graph[i]) < 2 and len(graph[j]) < 2):
            if x != y:
                union(parent, rank, x, y)
            graph[i].append(j)
            graph[j].append(i)
            output_A[i, j] = 1
            edge_count += 1

    # Iterative DFS
    path = []
    visited = [False] * N
    stack = [0]
    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            path.append(node)
            stack.extend(neighbor for neighbor in reversed(graph[node]) if not visited[neighbor])
    
    path.append(0)  # Return to start

    # Calculate total distance
    total_distance = sum(A[path[i], path[i+1]] for i in range(len(path)-1))
    # print('total_distance',total_distance)
    print('path', path)
    return output_A, total_distance


def farthest_insertion_tsp(points, adj_mat):
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    num_locs = len(points)
    max_val = int(np.max(dists))+1
    dists += 2*max_val * np.eye(num_locs)
    adj_mat = dists - 2*adj_mat
    # 시작 도시 선택 (여기서는 0번 도시)

    unvisited = np.arange(num_locs)
    init_node = np.random.randint(num_locs)
    path = [init_node]
    unvisited = unvisited[unvisited!=init_node]
    
    # curr_city = 0
    while len(unvisited):
        # 현재 경로에서 가장 먼 도시 찾기
        visited_dists = adj_mat[path]
        unvisited_dists = visited_dists[:, unvisited]
        min_dists = np.min(unvisited_dists, axis=0)
        farthest_city_index = np.argmax(min_dists)
        farthest_city = unvisited[farthest_city_index]

        
        # 찾은 도시를 경로에 추가할 최적 위치 찾기
        best_insertion_cost = float('inf')
        best_insertion_idx = None
        for i in range(len(path)):
            insertion_cost = adj_mat[path[i]][farthest_city] + adj_mat[farthest_city][path[(i+1) % len(path)]] - adj_mat[path[i]][path[(i+1) % len(path)]]
            if insertion_cost < best_insertion_cost:
                best_insertion_cost = insertion_cost
                best_insertion_idx = i+1
        
        # 경로에 도시 추가
        path.insert(best_insertion_idx, farthest_city)
        # unvisited.remove(farthest_city)
        unvisited = unvisited[unvisited!=farthest_city]

    # 시작 도시로 돌아오는 Edge 추가
    path.append(init_node)
    real_adj_matrix = np.zeros_like(adj_mat)
    for i in range(len(path)-1):
        real_adj_matrix[path[i], path[i+1]] = 1
    real_adj_matrix += real_adj_matrix.T
    merge_iterations = -100 # dummy value for consistency from cython_merge
    return real_adj_matrix, merge_iterations

def merge_tours(adj_mat, np_points, edge_index_np, sparse_graph=False, batch_size=1, guided=False, tsp_decoder='cython_merge'):
    """
    To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
    procedure.
    • Initialize extracted tour with an empty graph with N vertices.
    • Sort all the possible edges (i, j) in decreasing order of Aij/(vi_k-vj_k) (i.e., the inverse edge weight,
    multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
    • For each edge (i, j) in the list:
        – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
        – If inserting (i, j) results in a graph with cycles (of length < N), continue.
        – Otherwise, insert (i, j) into the tour.
    • Return the extracted tour.
    """
    # import pdb
    # pdb.set_trace()
    # print('np_points', np_points[:5])]
    if not sparse_graph:
        splitted_adj_mat = np.split(adj_mat, batch_size, axis=0)
        splitted_adj_mat = [
                # adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
                adj_mat[0] for adj_mat in splitted_adj_mat
        ]
    else:
        np_points = np_points.reshape([batch_size, -1, 2])
        task_size = np_points.shape[1]
        splitted_adj_mat = adj_mat.reshape(-1)
        Large_mat = scipy.sparse.coo_matrix(
                        (splitted_adj_mat, (edge_index_np[0], edge_index_np[1])),
                ).toarray() 
        # + scipy.sparse.coo_matrix(
                        # (splitted_adj_mat, (edge_index_np[1], edge_index_np[0])),
                # ).toarray()
        
        splitted_adj_mat = []
        for i in range(batch_size):
            splitted_adj_mat.append(Large_mat[i*task_size:(i+1)*task_size,i*task_size:(i+1)*task_size])

    splitted_points = [
            np_point for np_point in np_points
    ]
    if tsp_decoder == 'cython_merge':
        tsp_decoder = cython_merge
    elif tsp_decoder == 'am_decoding':
        tsp_decoder = am_decoding
    elif tsp_decoder == 'farthest':
        tsp_decoder = farthest_insertion_tsp
    else:
        raise ValueError(f"Unknown decoder type: {tsp_decoder}")

    if np_points.shape[1] > 1000 and batch_size > 1:
        with Pool(batch_size) as p:
            results = p.starmap(
                    tsp_decoder,
                    zip(splitted_points, splitted_adj_mat),
            )
    else:
        results = [
                tsp_decoder(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
        ]
    splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

    tours = []
    for i in range(batch_size):
        tour = [0]
        splited_sym = splitted_real_adj_mat[i] + splitted_real_adj_mat[i].T
        while len(tour) < splitted_adj_mat[i].shape[0] + 1:
            n = np.nonzero(splited_sym[tour[-1]])[0]
            if len(tour) > 1:
                n = n[n != tour[-2]]
            tour.append(n.max())
        tours.append(tour)
    merge_iterations = np.mean(splitted_merge_iterations)
    # if not np.array_equal(np.array(tours[0]), np.array(tours[1])):
        # print('tours are not same')
    return tours, merge_iterations, splitted_real_adj_mat


def merge_tours_parallel(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1, tsp_decoder='cython_merge'):
  """
  To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
  procedure.
  • Initialize extracted tour with an empty graph with N vertices.
  • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
  multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
  • For each edge (i, j) in the list:
    – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    – Otherwise, insert (i, j) into the tour.
  • Return the extracted tour.
  """
  splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
  if tsp_decoder == 'cython_merge':
    tsp_decoder = cython_merge
  elif tsp_decoder == 'am_decoding':
    tsp_decoder = am_decoding
  elif tsp_decoder == 'farthest':
    tsp_decoder = farthest_insertion_tsp
  if not sparse_graph:
    splitted_adj_mat = [
        adj_mat[0] 
        # + adj_mat[0].T
          for adj_mat in splitted_adj_mat
    ]
  else:
    splitted_adj_mat = [
        scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[0], edge_index_np[1])),
        ).toarray() 
        # + scipy.sparse.coo_matrix(
        #     (adj_mat, (edge_index_np[1], edge_index_np[0])),
        # ).toarray() 
        for adj_mat in splitted_adj_mat
    ]

  splitted_points = [
      np_points for _ in range(parallel_sampling)
  ]

  if np_points.shape[0] > 1000 and parallel_sampling > 1:
    with Pool(parallel_sampling) as p:
      results = p.starmap(
          cython_merge,
          zip(splitted_points, splitted_adj_mat),
      )
  else:
    results = [
        cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
    ]

  splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

  tours = []
  for i in range(parallel_sampling):
    tour = [0]
    splitted_real_adj_mat_sym = splitted_real_adj_mat[i] + splitted_real_adj_mat[i].T
    while len(tour) < splitted_adj_mat[i].shape[0] + 1:
      n = np.nonzero(splitted_real_adj_mat_sym[tour[-1]])[0]
      if len(tour) > 1:
        n = n[n != tour[-2]]
      tour.append(n.max())
    tours.append(tour)

  merge_iterations = np.mean(splitted_merge_iterations)
  return tours, merge_iterations

# def merge_tours_240129(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1, guided=False):
#     """
#     To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
#     procedure.
#     • Initialize extracted tour with an empty graph with N vertices.
#     • Sort all the possible edges (i, j) in decreasing order of Aij/(vi_k-vj_k) (i.e., the inverse edge weight,
#     multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
#     • For each edge (i, j) in the list:
#         – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
#         – If inserting (i, j) results in a graph with cycles (of length < N), continue.
#         – Otherwise, insert (i, j) into the tour.
#     • Return the extracted tour.
#     """

#     # splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
#     if not sparse_graph:
#         splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
#         # print('before splitted_adj_mat[0]', splitted_adj_mat[0], splitted_adj_mat[0].shape)
#         splitted_adj_mat = [
#                 adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
#         ]

#     else:
#         np_points = np_points.reshape([parallel_sampling, -1, 2])
#         task_size = np_points.shape[1]
#         # print('sparse merg tours occurs')
#         # print('edge_index_np',edge_index_np, edge_index_np.shape)
#         splitted_adj_mat = adj_mat

#         # M = scipy.sparse.coo_matrix((splitted_adj_mat, (edge_index_np[0], edge_index_np[1])),).toarray()
#         # for adj_mat in splitted_adj_mat:
#             # break
#         # print("M", M)

#         Large_mat = scipy.sparse.coo_matrix(
#                         (splitted_adj_mat, (edge_index_np[0], edge_index_np[1])),
#                 ).toarray() + scipy.sparse.coo_matrix(
#                         (splitted_adj_mat, (edge_index_np[1], edge_index_np[0])),
#                 ).toarray()
        
#         splitted_adj_mat = []
#         for i in range(parallel_sampling):
#             splitted_adj_mat.append(Large_mat[i*task_size:(i+1)*task_size,i*task_size:(i+1)*task_size])

#     # print('after splitted_adj_mat[0]', splitted_adj_mat[0], splitted_adj_mat[0].shape)

#     splitted_points = [
#             np_point for np_point in np_points
#     ]
    

#     if np_points.shape[0] > 1000 and parallel_sampling > 1:
#         with Pool(parallel_sampling) as p:
#             results = p.starmap(
#                     cython_merge,
#                     zip(splitted_points, splitted_adj_mat),
#             )
#     else:

#         # print(_)
#         results = [
#                 cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
#         ]

#     splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

#     tours = []
#     for i in range(parallel_sampling):
#         tour = [0]
#         while len(tour) < splitted_adj_mat[i].shape[0] + 1:
#             n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
#             if len(tour) > 1:
#                 n = n[n != tour[-2]]
#             tour.append(n.max())
#         tours.append(tour)

#     merge_iterations = np.mean(splitted_merge_iterations)

#     return tours, merge_iterations

def calculate_distance_matrix(points, edge_index = None):
    """
    주어진 점들의 좌표로부터 거리 행렬을 계산합니다.
    
    Args:
    points (torch.Tensor): 형태가 [B, N, 2]인 텐서. B는 배치 크기, N은 점의 개수, 2는 x, y 좌표를 나타냅니다.
    
    Returns:
    torch.Tensor: 형태가 [B, N, N]인 거리 행렬
    """
    if edge_index == None:
        # 점들을 [B, N, 1, 2] 형태로 확장
        expanded_p1 = points.unsqueeze(2)
        # 점들을 [B, 1, N, 2] 형태로 확장
        expanded_p2 = points.unsqueeze(1)
        
    else:
        expanded_p1 = points[edge_index[0]]
        expanded_p2 = points[edge_index[1]]
    diff = expanded_p1 - expanded_p2
    dist_matrix = torch.sqrt(torch.sum(diff**2, dim=-1))
        


    return dist_matrix

class TSPEvaluator(object):
    def __init__(self, points, batch=False):
        if isinstance(points, torch.Tensor):
            points = points.cpu().to(torch.float64).numpy()
        self.batch = batch
        # print('points.shape',points.shape)
        if self.batch:
            self.dist_mat = torch.cdist(torch.from_numpy(points), torch.from_numpy(points))
        else:
            self.dist_mat = scipy.spatial.distance_matrix(points, points)

        # print('self.dist_mat.dtype',self.dist_mat.dtype)
    def evaluate(self, route):
        if self.batch:
            route_indices = route.to('cpu')
            # Calculate the indices for the start and end cities in the route
            # print(route_indices)
            # print('route_indices',route_indices)
            start_cities = route_indices[:, :-1]
            end_cities = route_indices[:, 1:]
            # Use tensor operations to calculate tour lengths for all batches in parallel
            distances = self.dist_mat[start_cities, end_cities]
            distances = distances.sum(dim=-1)
            return distances
        else:
            total_cost = 0
            for i in range(len(route) - 1):
                total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost
    def evaluate_new(self, route):
        if self.batch:
            if isinstance(route, torch.Tensor):
                route_indices = route.to('cpu')
            elif isinstance(route, np.ndarray):
                route_indices = torch.from_numpy(route).to('cpu')
            else:
                route_indices = torch.tensor(route, dtype=torch.long)
                if len(route_indices.size()) == 1:
                    route_indices = route_indices.unsqueeze(0)

            batch_size = len(route)
            ins_size = self.dist_mat.size()[-1]
            # Calculate the indices for the start and end cities in the route
            # print(route_indices)
            # print('route_indices',route_indices)
            start_cities = route_indices[:, :-1]
            end_cities = route_indices[:, 1:]
            # Use tensor operations to calculate tour lengths for all batches in parallel
            m_indices = torch.arange(batch_size)
            m_indices = m_indices.reshape(-1, 1).repeat(1, ins_size)
            distances = self.dist_mat[m_indices, start_cities, end_cities]

            distances = distances.sum(dim=-1)
            return distances
        else:
            total_cost = 0
            for i in range(len(route) - 1):
                total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost
