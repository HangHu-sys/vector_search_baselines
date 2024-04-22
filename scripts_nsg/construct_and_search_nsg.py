"""
This script is used to (a) construct NSG using the C++ binaries, and (b) conduct search.

Example Usage:

python construct_and_search_nsg.py \
    --DATA_PATH /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_base.fvecs \
    --QUERY_PATH /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_query.fvecs \
    --GT_PATH /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_groundtruth.ivecs \
    --KNNG_PATH ./sift_200nn.graph \
    --NSG_PATH ./sift1m.nsg
"""


import numpy as np
import random
import time
import bisect
import argparse
import os
import glob

class Neighbor:
    def __init__(self, id, distance, flag):
        self.id = id
        self.distance = distance
        self.flag = flag

    def __lt__(self, other: 'Neighbor'):
        return self.distance < other.distance
    
    def printNeighbor(self):
        print("id: ", self.id, "distance: ", self.distance, "flag: ", self.flag)


def compare(query_data, db_vec, dimension: int):
    return np.sum((query_data - db_vec) ** 2)


def print_retset(retset):
    for i in range(len(retset)):
        retset[i].printNeighbor()
        
        
class IndexNSG():
    def __init__(self, dimension: int, n: int):
        self.final_graph = []
        self.dimension = dimension
        self.width = 0
        self.ep_ = 0
        self.nd_ = n
    
    def Load(self, filename: str):
        with open(filename, "rb") as f:
            self.width = int.from_bytes(f.read(4), byteorder='little', signed=False)
            self.ep_ = int.from_bytes(f.read(4), byteorder='little', signed=False)
            cc = 0
            while True:
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k = int.from_bytes(k_bytes, byteorder='little', signed=False)
                cc += k
                tmp = list(np.frombuffer(f.read(k * 4), dtype=np.uint32))
                self.final_graph.append(tmp)
            cc //= self.nd_
            # print(cc)


    def search_with_base_graph(self, query, x, K, parameters):
        L = parameters['L_search']
        data_ = x
        retset:list[Neighbor] = []
        init_ids = []
        flags = [0] * self.nd_
        node_counter = 0
        
        for tmp_l in range(min(L, len(self.final_graph[self.ep_]))):
            init_ids.append(self.final_graph[self.ep_][tmp_l])
            flags[init_ids[tmp_l]] = 1
        
        tmp_l += 1
    
        while tmp_l < L:
            id = random.randint(0, self.nd_ - 1)
            if flags[id] == 1:
                continue
            flags[id] = 1
            init_ids.append(id)
            tmp_l += 1

        for i in range(len(init_ids)):
            id = init_ids[i]
            dist = compare(data_[id], query, self.dimension)
            retset.append(Neighbor(id, dist, 1))
        
        node_counter += len(init_ids)
        
        retset.sort()
        
        k = 0
        while k < L:
            nk = L
            if retset[k].flag:
                retset[k].flag = 0
                n = retset[k].id
        
                for m in range(len(self.final_graph[n])):
                    id = self.final_graph[n][m]
                    if flags[id] == 1:
                        continue
                    flags[id] = 1
                    dist = compare(query, data_[id], self.dimension)
                    node_counter += 1
                    if dist >= retset[L - 1].distance:
                        continue
                    nn = Neighbor(id, dist, 1)
                    bisect.insort_left(retset, nn)
                    r = retset.index(nn)
                    if len(retset) > L + 1:
                        retset.pop()
                    if r < nk:
                        nk = r
                    # nk: the index of the smallest dist
            
            if nk <= k:
                k = nk
            else:  
                k += 1
                
        # print_retset(retset)
        
        indices = [0] * K
        for i in range(K):
            indices[i] = retset[i].id
        
        return (indices, node_counter)


    def search_with_base_graph_2queue(self, query, x, K, parameters):
        L = parameters['L_search']
        data_ = x
        candidate_set:list[Neighbor] = []
        top_candidates:list[Neighbor] = []
        node_counter = 0
        
        init_ids = []
        flags = [0] * self.nd_
        
        for tmp_l in range(min(L, len(self.final_graph[self.ep_]))):
            init_ids.append(self.final_graph[self.ep_][tmp_l])
            flags[init_ids[tmp_l]] = 1
    
        while tmp_l < L:
            id = random.randint(0, self.nd_ - 1)
            if flags[id] == 1:
                continue
            flags[id] = 1
            init_ids.append(id)
            tmp_l += 1

        for i in range(len(init_ids)):
            id = init_ids[i]
            dist = compare(data_[id], query, self.dimension)
            candidate_set.append(Neighbor(id, dist, 1))
            top_candidates.append(Neighbor(id, dist, 1))
        
        node_counter += len(init_ids)
        
        candidate_set.sort()
        top_candidates.sort()
        
        while len(candidate_set) > 0:
            cur_node = candidate_set.pop(0)
            
            if cur_node.distance > top_candidates[L - 1].distance:  # here we assume both queues has infinite capacity
                break
            
            cur_id = cur_node.id

            for m in range(len(self.final_graph[cur_id])):
                id = self.final_graph[cur_id][m]
                if flags[id] == 1:
                    continue
                flags[id] = 1
                dist = compare(query, data_[id], self.dimension)
                node_counter += 1
                if dist >= top_candidates[L - 1].distance:  # here L-1 is a relaxed condition ???
                    continue
                nn = Neighbor(id, dist, 1)
                bisect.insort_left(candidate_set, nn)
                bisect.insort_left(top_candidates, nn)
        
        indices = [0] * K
        for i in range(K):
            indices[i] = top_candidates[i].id
        
        return (indices, node_counter)


def load_data(filename):
    with open(filename, "rb") as file:
        dim_bytes = file.read(4)  # Read 4 bytes for dimension
        dim = int.from_bytes(dim_bytes, byteorder='little')  # Convert bytes to integer for dimension

        file.seek(0, 2)  # Move the file pointer to the end
        fsize = file.tell()  # Get the file size
        num = fsize // ((dim + 1) * 4)  # Calculate the number of data points

        file.seek(0)  # Move the file pointer back to the beginning
        data = np.empty((num, dim), dtype=np.float32)  # Create an empty numpy array to store data

        for i in range(num):
            file.seek(4, 1)  # Move the file pointer forward by 4 bytes to skip index
            data[i] = np.fromfile(file, dtype=np.float32, count=dim)  # Read dim number of float values

    return data, num, dim


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Wenqi: Format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_PATH', type=str, default="/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_base.fvecs", help='Path to the base data file in fvecs format')
    parser.add_argument('--QUERY_PATH', type=str, default="/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_query.fvecs", help='Path to the query data file in fvecs format')
    parser.add_argument('--GT_PATH', type=str, default="/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_groundtruth.ivecs", help='Path to the groundtruth file in ivecs format')
    parser.add_argument('--KNNG_PATH', type=str, default="./sift_200nn.graph", help='Path to the KNNG graph file')
    parser.add_argument('--NSG_PATH', type=str, default="./sift1m.nsg", help='Path to the NSG index file')

    # TODO 1: Add parameters for the search algorithm
    # NSG construction parameters
    L = 40
    R = 50
    C = 500
    # Search parameters
    K = 100
    L_search = 100
    assert(L_search >= K)

    args = parser.parse_args()


    '''
    Part 1: Load data and build the NSG index
    '''
    print("Start loading data")
    data_load, points_num, dim = load_data(args.DATA_PATH)
    query_load, query_num, query_dim = load_data(args.QUERY_PATH)
    assert(dim == query_dim)
    gt_load = ivecs_read(args.GT_PATH)

    print("data_load: ", np.shape(data_load))
    print("query_load: ", np.shape(query_load))
    print("gt_load: ", np.shape(gt_load))

    index = IndexNSG(dim, points_num)

    if glob.glob(args.NSG_PATH):
        print("NSG index file already exists")
    else:
        nsg_construct_path = "../nsg/build/tests/test_nsg_index"
        if not glob.glob(nsg_construct_path):
            print(f"NSG construction code: {nsg_construct_path} not found")
            exit()
        # assert path exists
        assert os.path.exists(args.DATA_PATH)
        assert os.path.exists(args.KNNG_PATH)
        
        cmd_build_nsg = f"{nsg_construct_path} {args.DATA_PATH} {args.KNNG_PATH} {L} {R} {C} {args.NSG_PATH}"
        print(f"Building NSG by running command:\n{cmd_build_nsg}")
        os.system(cmd_build_nsg)

        if not glob.glob(args.NSG_PATH):
            print("NSG index file construction failed")
            exit()
        print("NSG index file construction succeeded")


    '''
    Part 2: Perform the search
    '''
    index.Load(args.NSG_PATH)

    qsize = 100
    paras = {'L_search': L_search}

    total = 0
    correct = 0
    total_counter = 0

    print("Start searching (single-threaded)")
    for i in range(qsize):
        indices, node_counter = index.search_with_base_graph(query_load[i], data_load, K, paras)
        total_counter += node_counter
        gt = gt_load[i]
        # print(len(gt))
        g = set(gt)
        total += len(gt)
        
        for item in indices:
            if item in g:
                correct += 1

    recall = 1.0 * correct / total
    avg_counter = 1.0 * total_counter / qsize
    print("recall: ", recall, "avg_counter: ", avg_counter)