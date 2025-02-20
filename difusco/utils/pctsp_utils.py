import os
import warnings
from multiprocessing import Pool
import time

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
import copy as cp


def merge_tours_pctsp(adj_mat, np_points):
    """Merge tours for pctsp, adj_mat = [Batch, num_locs, num_locs], np_points = [Batch, num_locs, 4], prize = [Batch, num_locs]
    """
    # results_test = merge_tours_pctsp_ver3(adj_mat, np_points)
    # results_test = merge_tours_pctsp_ver4(adj_mat, np_points)
    results_test = merge_tours_pctsp_ver5(adj_mat, np_points)

    return results_test


def merge_tours_pctsp_ver3(adj_mat, np_points):
    """Version 3 of the pctsp, doing similar way to loop search then, reduce the useless length
    """
    batch_size, num_locs = np_points.shape[0], np_points.shape[1]
 
    instances = np_points.reshape([batch_size, num_locs, 4])
    splitted_points = instances[:,:,:2]
    splitted_penalties = instances[:,:,2].reshape([batch_size, num_locs])
    splitted_prizes = instances[:,:,3].reshape([batch_size, num_locs])

    splitted_adj_mat = adj_mat.reshape([batch_size, num_locs, num_locs])

    results_test  = []

    for i in range(batch_size):
        result_new = (merge_pctsp_3(splitted_adj_mat[i], splitted_points[i],splitted_penalties[i], splitted_prizes[i]))
        results_test.append(result_new)

    return results_test

def merge_tours_pctsp_ver4(adj_mat, np_points):
    """using more disturbation of adj_mat 
    """

    batch_size, num_locs = np_points.shape[0], np_points.shape[1]
 
    instances = np_points.reshape([batch_size, num_locs, 4])
    splitted_points = instances[:,:,:2]
    splitted_penalties = instances[:,:,2].reshape([batch_size, num_locs])
    splitted_prizes = instances[:,:,3].reshape([batch_size, num_locs])

    splitted_adj_mat = adj_mat.reshape([batch_size, num_locs, num_locs])

    results_test  = []

    for i in range(batch_size):
        result_new = (merge_pctsp_4(splitted_adj_mat[i], splitted_points[i],splitted_penalties[i], splitted_prizes[i]))
        results_test.append(result_new)

    return results_test


def merge_tours_pctsp_ver5(adj_mat, np_points):
    """using more disturbation of adj_mat 
    """
    # print('merge_tours_pctsp_ver4')

    batch_size, num_locs = np_points.shape[0], np_points.shape[1]
 
    instances = np_points.reshape([batch_size, num_locs, 4])
    splitted_points = instances[:,:,:2]
    splitted_penalties = instances[:,:,2].reshape([batch_size, num_locs])
    splitted_prizes = instances[:,:,3].reshape([batch_size, num_locs])

    splitted_adj_mat = adj_mat.reshape([batch_size, num_locs, num_locs])

    results_test  = []

    for i in range(batch_size):
        result_new = (merge_pctsp_5(splitted_adj_mat[i], splitted_points[i],splitted_penalties[i], splitted_prizes[i]))
        results_test.append(result_new)

    return results_test

def merge_pctsp_5(adj_mat, points, penalty, prize):
    """
    """
    if len(points.shape)>2:
        print('points dimension is wrong', points.shape)
        raise ValueError
    num_locs = len(points)
    node = [0, 0]
    paths_directwise = [[0], [0]]
    activated_nodes = [set([0]), set([0])]
    # prize_sum = 0
    prize_total = prize.sum()
    if type(adj_mat)==type(torch.ones([2])):
        check_positive = (torch.sum(adj_mat)>0).to(int)*2 - 1
    else:
        check_positive = (np.sum(adj_mat)>0).astype(int)*2 - 1

    adj_mat =  check_positive*np.transpose(adj_mat) * adj_mat 
    adj_mat_list = [cp.deepcopy(adj_mat), cp.deepcopy(adj_mat)]

    prize_sum_check = False
    prize_sum_list = [[0], [0]]

    for _ in range(num_locs):
        if prize_sum_check:
            break

        sorted_nodes = np.argsort(-np.append(adj_mat_list[0][node[0]], adj_mat_list[1][node[1]]).flatten())
        if type(sorted_nodes)==type(torch.ones([2])):
            sorted_nodes= sorted_nodes.numpy()

        for idx in range(num_locs):
            direction, cand_node = int(sorted_nodes[idx]/num_locs), sorted_nodes[idx]%num_locs
            if cand_node in activated_nodes[1-direction]:
                prize_sum =  prize_sum_list[direction][-1] + prize_sum_list[1-direction][paths_directwise[1-direction].index(cand_node)]
            # if cand_node == 0:
                if prize_sum >= min(1, prize_total):
                    # print('prize sum exceeds',prize_sum)
                    prize_sum_check = True
                    break
                else:
                    continue
            elif cand_node in activated_nodes[direction]:
                continue


            # print('paths_directwise', paths_directwise, 'activated_nodes',activated_nodes, 'prize_sum_list',prize_sum_list)
            node[direction] = cand_node
            paths_directwise[direction].append(int(cand_node))
            activated_nodes[direction].add(cand_node)
            prize_sum_list[direction].append(prize_sum_list[direction][-1]+prize[cand_node])
            # prize_sum += prize[cand_node]
            # print('_ ', _ ,'cand_node', cand_node)
            adj_mat_list[direction][:, cand_node] = - np.inf
            break
    
    path = paths_directwise[direction] + paths_directwise[1-direction][:paths_directwise[1-direction].index(cand_node)+1][::-1]
    if prize[path].sum()<=min(1, prize_total):
        print('wrong')
        exit(-1)
    return pctsp_tour_clean(path, points, prize, penalty, num_locs)


def merge_pctsp_4(adj_mat, points, penalty, prize):
    """
    """
    if len(points.shape)>2:
        print('points dimension is wrong', points.shape)
        raise ValueError
    num_locs = len(points)
    node = [0, 0]
    paths_directwise = [[0], [0]]
    activated_nodes = set()
    prize_sum = 0
    prize_total = prize.sum()
    if type(adj_mat)==type(torch.ones([2])):
        check_positive = (torch.sum(adj_mat)>0).to(int)*2 - 1
    else:
        check_positive = (np.sum(adj_mat)>0).astype(int)*2 - 1

    adj_mat =  check_positive* np.transpose(adj_mat) * adj_mat

    prize_sum_check = False
    for _ in range(num_locs):
        if prize_sum_check:
            break
        sorted_nodes = np.argsort(-adj_mat[node].flatten())
        if type(sorted_nodes)==type(torch.ones([2])):
            sorted_nodes= sorted_nodes.numpy()

        for idx in range(num_locs):
            direction, cand_node = int(sorted_nodes[idx]/num_locs), sorted_nodes[idx]%num_locs
            if cand_node in activated_nodes:
                continue
            else:
                if cand_node == 0:
                    if prize_sum >= min(1, prize_total):
                        # print('prize sum exceeds',prize_sum)
                        pass
                    else:
                        continue
            if cand_node == 0:
                prize_sum_check = True
                break
            node[direction] = cand_node
            paths_directwise[direction].append(int(cand_node))
            activated_nodes.add(cand_node)
            prize_sum += prize[cand_node]
            # print('_ ', _ ,'cand_node', cand_node)
            adj_mat[:, cand_node] = - np.inf
            break
    
    path = paths_directwise[0] + paths_directwise[1][::-1]
    
    return pctsp_tour_clean(path, points, prize, penalty, num_locs)



def merge_pctsp_3(adj_mat, points, penalty, prize):
    """
    """
    if len(points.shape)>2:
        print('points dimension is wrong', points.shape)
        raise ValueError
    num_locs = len(points)
    node = 0
    path = []
    path.append(node)
    activated_nodes = set()
    prize_sum = 0
    prize_total = prize.sum()

    for _ in range(num_locs):
        sorted_nodes = np.argsort(-adj_mat[node])
        if type(sorted_nodes)==type(torch.ones([2])):
            sorted_nodes= sorted_nodes.numpy()

        for idx in range(num_locs):
            cand_node = sorted_nodes[idx]
            if cand_node in activated_nodes:
                continue
            else:
                if cand_node == 0:
                    if prize_sum >= min(1, prize_total):
                        pass
                    else:
                        continue
            node = cand_node
            if node == 0:
                break
            path.append(int(node))
            activated_nodes.add(node)
            prize_sum += prize[node]
            adj_mat[:,node] = - np.inf
            break
    
    path.append(0)
    return pctsp_tour_clean(path, points, prize, penalty, num_locs)


def pctsp_tour_clean(path, points, prize, penalty, num_locs):
    dist = get_distance_matrix(points)

    node_order = dict()
    prize_sum = prize[path].sum()
    if prize_sum<=1:
        pass
    else:
        node_scores = np.ones(num_locs)*np.inf
        node_order[0] = [path[1], path[-2]]
        for path_idx in range(1, len(path)-1):
            node_cur = path[path_idx]
            node_prev = path[path_idx-1]
            node_next = path[path_idx+1]
            node_order[node_cur] = [node_prev, node_next]
            node_score = (penalty[node_cur] - (dist[node_cur, node_prev]+dist[node_cur, node_next]-dist[node_prev, node_next]))/prize[node_cur]
            node_scores[node_cur] = node_score
            
        while True:
            node_cur = np.argmin(node_scores)
            
            if node_scores[node_cur]>0:
                break
            node_prev, node_next = node_order[node_cur][0], node_order[node_cur][1]
            if prize_sum - prize[node_cur]>=1:
                prize_sum -= prize[node_cur]
                path.remove(node_cur)
                node_order.pop(node_cur)
                node_order[node_prev][1] = node_next
                node_order[node_next][0] = node_prev
                
                if node_prev != 0:
                    node_scores[node_prev] = (penalty[node_prev] - (dist[node_order[node_prev][0],node_prev]+dist[node_prev, node_next]-dist[node_order[node_prev][0], node_next]))/prize[node_prev]

                if node_next != 0:
                    node_scores[node_next] = (penalty[node_next] - (dist[node_prev, node_next]+dist[node_next, node_order[node_next][1]]-dist[node_prev, node_order[node_next][1]]))/prize[node_next]
            
            node_scores[node_cur] = np.inf
            
        
        for _ in range(num_locs+2-len(path)):
            path.append(0)
    return path

def merge_pctsp_prize(points, adj_mat, prize):
    """
    """
    num_locs = len(points)
    node = 0
    path = []
    path.append(node)
    activated_nodes = set()
    prize_sum = 0
    prize_total = prize.sum()
    count = 0
    for _ in range(num_locs):
        
        sorted_nodes = np.argsort(-adj_mat[node])
        if type(sorted_nodes)==type(torch.ones([2])):
            sorted_nodes= sorted_nodes.numpy()

        for idx in range(num_locs):
            cand_node = sorted_nodes[idx]
            if cand_node in activated_nodes:
                continue
            else:
                if cand_node == 0:
                    if prize_sum >= min(1, prize_total):
                        pass
                    else:
                        continue
            node = cand_node
            path.append(int(node))
            activated_nodes.add(node)
            prize_sum += prize[node]
            adj_mat[:,node] = np.inf
            count += 1
            break

        if prize_sum>=1: # END
            node=0

        if node == 0:
            for _ in range(num_locs-count):
                path.append(int(node))
            break
        
    
    return path


def get_distance_matrix(locs_2d):
    """Compute the distances between locs_2d
    """

    if type(locs_2d)==type(np.array([])):
        locs_2d = torch.from_numpy(locs_2d)
    dim_locs = len(locs_2d.size())
    if dim_locs==3:
        batch_size = locs_2d.shape[0]
        num_locs = locs_2d.shape[-2]
    elif dim_locs==2:
        batch_size = 1
        num_locs = locs_2d.shape[-2]
    else:
        print('locs_2d.size()', dim_locs)
        raise NotImplementedError
    
    locs_2d_first = locs_2d.view([batch_size, num_locs, 1, 2])
    locs_2d_second = locs_2d.view([batch_size, 1, num_locs, 2])
    distances = locs_2d_first - locs_2d_second
    distances = torch.sqrt(torch.sum(distances * distances, dim = -1))

    if dim_locs==3:
        return distances
    elif dim_locs==2:
        return distances.view(num_locs, num_locs)


def round_prize(prize, digit = 5):
    """Modify the prize
    """
    prize_scale = pow(10, digit)
    prize = np.array(prize)
    prize *= prize_scale
    prize = np.round(prize).astype(int)
    prize_total = prize.sum()
    prize.tolist()

    return prize, prize_total, prize_scale

def compute_cost_from_tour_batch(tour_batch, points_batch, penalty_batch, prize_batch):
    """Compute the cost from tours using batch
    tour = [batch, *], points = [batch, num_locs, 2], prize = [batch, num_locs], penalty = [batch, num_locs]
    """
    results = [compute_cost_from_tour(tour, points, penalty, prize) for tour, points,  penalty, prize in zip(tour_batch, points_batch, penalty_batch, prize_batch)]
    
    return results

def compute_cost_from_tour(path, points, penalty, prize):
    """Compute costs from tours
    tour = [paths], points = [num_locs, 2], prize = [num_locs], penalty = [num_locs]
    """
    distances = get_distance_matrix(points)

    current_node = 0
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    penalty_sum = 0
    prize_total= prize.sum()
    prize_sum = 0
    tours = []
    penalty_total = penalty.sum()

    node_prev = 0

    first_check = True
    for current_node in path:
        if first_check:
            plan_output += f" {current_node} ->"
            first_check = False
            continue

        penalty_sum += penalty[current_node]
        prize_sum += prize[current_node]
        plan_output += f" {current_node} ->"
        route_distance += distances[node_prev][current_node]
        node_prev = current_node

    plan_output += f" {current_node}\n"
    plan_output += f"Distance of the route: {route_distance}m\n"
    plan_output += f"penalty collected: {penalty_sum}/{sum(penalty)}\n"
    plan_output += f"prize collected: {prize_sum}/{prize_total}\n"  

    cost = - (penalty_sum - penalty_total - route_distance)

    row_write = ''
    for coord in points:
       row_write += str(float(coord[0])) + " "
       row_write += str(float(coord[1])) + " "

    row_write += 'Tours '
    for node in tours:
       row_write += str(node) + ' '

    
    row_write += 'Cost ' + str(float(cost)) + ' '
    row_write += 'Penalty_Unreceived ' + str(float(penalty_sum)) + ' '
    row_write += 'Distance ' + str(float(route_distance)) + ' '
    row_write += 'Prize_Collected ' + str(float(prize_sum)) + ' '
    row_write += '\n'

    return row_write


class PCTSPEvaluator(object):
    def __init__(self, instance, batch=None):
        """batch is not used : just in TSPEvaluator
        """
        # self.dist_mat = torch.cdist(points,points)
        if type(instance) == type(np.array([])):
            instance = torch.from_numpy(instance)

        points = instance[:,:2]
        penalty = instance[:,2]
        prize = instance[:,3]

        self.dist_mat = torch.cdist(points, points)
            
        self.points = points
        self.penalty = penalty
        self.prize = prize

    def evaluate(self, tour):
        """Evaluate the sinlge tour for given instance
        """
        dist_sum = 0
        penalty_sum = 0
        # prize_total= prize.sum()
        # tours = []
        penalty_total = self.penalty.sum()


        route_indices = np.array(tour.to('cpu'))
        if len(route_indices.shape)==1:
            route_indices = route_indices.reshape([1, -1])
        start_cities = route_indices[:, :-1]
        end_cities = route_indices[:, 1:]
        distances = self.dist_mat[start_cities, end_cities]
        dist_sum = distances.sum(axis=-1)
        penalty_sum = self.penalty[route_indices].sum(axis=-1)


        cost = - (penalty_sum - penalty_total - dist_sum)


        return cost

    def evaluate_0228(self, tour):
        """Evaluate the sinlge tour for given instance
        """

        current_node = 0
        dist_sum = 0
        penalty_sum = 0
        prize_sum = 0
        penalty_total = self.penalty.sum()

        node_prev = 0

        first_check = True
        for current_node in tour:
            if first_check:
                first_check = False
                continue

            penalty_sum += self.penalty[current_node]
            prize_sum += self.prize[current_node]
            dist_sum += self.dist_mat[node_prev][current_node]
            node_prev = current_node

        cost = - (penalty_sum - penalty_total - dist_sum)

        return cost


def test_merge_pctsp_tours():
    """Test the merge pctsp_tours()
    """
    instances_torch = torch.from_numpy(instances)
    time1 = time.perf_counter()
    # for _ in range(Repeat):

    path = merge_tours_pctsp(adj_mat, instances_torch)
    time2 = time.perf_counter()

    print('path[0]', path[0], 'len(path)', len(path), 'time', time2-time1)

    # print('row_write',row_write)

    row_writes = compute_cost_from_tour_batch(path, instances_torch[:,:,:2], instances_torch[:,:,2], instances_torch[:,:,3])
    # exit(0)
    print('row_writes[0]', row_writes[0])
    # print('row_writes[1]', row_writes[1])

    # print('path', path)

def test_pctsp_evalutor():

    time1 = time.perf_counter()

    # for _ in range(Repeat):
    # path, path_2 = merge_tours_pctsp(adj_mat, instances)
    path = merge_tours_pctsp(adj_mat, instances)
    
    time2 = time.perf_counter()
    path = torch.tensor(path)
    cost_sum = 0
    num_instances = len(instances)
    for i in range(num_instances):
        pctsp_solver = PCTSPEvaluator(instances[i])
    # pctsp_solver = PCTSPEvaluator(points[0], prizes[0], penalties[0])
        cost = pctsp_solver.evaluate(path[i])
    # print('[New] path[0]', path[0], 'prize[0]', prizes[0], 'penalty[0]', penalties[0])
        cost_sum += cost

        # row_write = compute_cost_from_tour(path[i], instances[i,:,:2], instances[i,:,2].squeeze(), instances[i,:,3].squeeze())
        # print(cost, row_write)

    # exit(0)

    cost = pctsp_solver.evaluate(path[i])
    cost_old = pctsp_solver.evaluate_0228(path[i])
    # print('[old] path[0]', path[0], 'prize[0]', prizes[0], 'penalty[0]', penalties[0])
    cost_avg = cost_sum/num_instances

    print('cost', cost, 'cost_old', cost_old, 'cost_avg', cost_avg)
    if np.abs(cost-cost_old)>1e-5:
        raise Exception("cost values are different", cost, cost_old)



def test_pctsp_evaluator_load_dataset():

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
                
                tour = np.append(tour, np.zeros(len(points) - len(tour)))
                
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


    load_dir = '/lab-di/squads/diff_nco/data/pctsp_custom/pctsp_20_ortools_1sec_200.txt'

    train_dataset = PCTSPGraphDataset(
            data_file=load_dir,
            sparse_factor=-1, preload= False, optimality_dropout_rate=1.0,
)
    instance = train_dataset.__getitem__(1) 
    idx, points, adj_matrix, path, cost = instance

    pctsp_solver = PCTSPEvaluator(points)
    # pctsp_solver = PCTSPEvaluator(points[0], prizes[0], penalties[0])



# PCTSPEvaluator(np_points_single)
if __name__=='__main__':
    # np.random.seed(0)
    # torch.manual_seed(0)
    num_locs = 50
    batch_size = 500
    Repeat = 1
    Max_LENGTHS = {20:2.0, 50:3.0, 100:4.0}

    points = np.random.uniform(size=[batch_size, num_locs+1, 2])
    penalties = np.random.uniform(size=[batch_size, num_locs+1])/num_locs*Max_LENGTHS[num_locs]*3
    prizes = np.random.uniform(size=[batch_size, num_locs+1])/num_locs*4
    
    # adj_mat = torch.rand(size=[batch_size, num_locs+1,num_locs+1])

    adj_mat = - get_distance_matrix(points.reshape([batch_size, num_locs+1, 2]))
    
    # adj_mat = - torch.from_numpy(np.sqrt(np.sum(np.power(points.reshape([batch_size, num_locs+1, 1, 2]) - points.reshape([batch_size, 1, num_locs+1, 2]), 2), axis=-1))) ## This is minus cause we want shortest path (In Difusco, it is reverse)

    prizes[:, 0] = 0
    penalties[:, 0] = 0

    instances = np.concatenate((points.reshape([batch_size, -1, 2]), penalties.reshape([batch_size, -1, 1]), prizes.reshape([batch_size, -1, 1])), axis=-1)
    
    time1 = time.perf_counter()
    test_merge_pctsp_tours()
    time2 = time.perf_counter()

    test_pctsp_evalutor()
    time3 = time.perf_counter()

    test_pctsp_evaluator_load_dataset()
    time4 = time.perf_counter()
    print('time merge', time2-time1, 'time eval', time3-time2, 'laod data', time4-time3)