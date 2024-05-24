"""
To compare the search performance of the full graph approach and the sub-graph approach, we need to construct:
    
    1. The full graph index (e.g. SIFT1M_index_MD64_par1_0.bin)
    
    2. The sub-graph indexes (e.g. SIFT1M_index_MD64_par4_x.bin)
        
        The first two must share the same efConstruction, efSearch, and M parameters.
        Both of them are used for the c++ search and get_items() operation in conversion.
    
    3. The verification graph index (e.g. SIFT1M_index_ver.bin)
    
        The verification graph index is used to convert the sub-graph search results (ids) to full graph search results (ids).
        So this index should be much more accurate using higher efConstruction, efSearch, and M parameters.

Example Usage:
    python construct_subgraph_hnsw.py \
        --dbname SIFT1M \
        --ef_construction 128 \
        --MD 64 \
        --ef_full 64 \
        --ef_sub 24 \
        --hnsw_path /mnt/scratch/hanghu/CPU_hnsw_index \
        --subgraph_result_path /mnt/scratch/hanghu/sub_graph_results \
        --sub_graph_num 4
"""
import argparse
import sys
import os
import time 

import hnswlib
import numpy as np

from utils import mmap_fvecs, mmap_bvecs, ivecs_read, fvecs_read, mmap_bvecs_SBERT, \
    read_deep_ibin, read_deep_fbin, print_recall, calculate_recall


def read_from_log(log_fname:str):
    """
    Read recall and node_count from the log file
    """
    node_counter = 0
    recall_1 = None
    recall_10 = None

    with open(log_fname, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if "recall_1:" in line:
                recall_1 = float(line.split(" ")[1])
            elif "recall_10:" in line:
                recall_10 = float(line.split(" ")[1])
            elif "node counter per query" in line:
                node_counter = int(line.split(" ")[4])

    return recall_1, recall_10, node_counter


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', type=str, default="SIFT1M", help='name of the database, e.g., SIFT10M, Deep10M, GLOVE')
    parser.add_argument('--ef_construction', type=int, default=128, help='ef construction parameter')
    parser.add_argument('--MD', type=int, default=64, help='Max degree of base layer, M * 2 === M0 == MD')
    parser.add_argument('--ef_full', type=int, default=64, help='ef search parameter on full graph')
    parser.add_argument('--ef_sub', type=int, default=64, help='ef search parameter on each sub-graph')
    parser.add_argument('--hnsw_path', type=str, default="../data/CPU_hnsw_indexes", help='Path to the HNSW index file')
    parser.add_argument('--subgraph_result_path', type=str, default="", help='Path to the subgraph search result')
    parser.add_argument('--sub_graph_num', type=int, default=1, help='Number of partitions')
    args = parser.parse_args()
    
    dbname = args.dbname
    ef_construction = args.ef_construction
    ef_full = args.ef_full
    ef_sub = args.ef_sub
    hnsw_path = args.hnsw_path
    subgraph_result_path = args.subgraph_result_path
    sub_graph_num = args.sub_graph_num
    M = int(args.MD / 2)
    print("MD: {} Derived M in HNSW: {}".format(args.MD, M))
    
    log_perf_test = 'hnsw_{}.log'.format(np.random.randint(1000000))

    if dbname.startswith('SIFT'):
        # SIFT1M to SIFT1000M
        dbsize = int(dbname[4:-1])
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/bigann'
        xb = mmap_bvecs(os.path.join(dataset_dir, 'bigann_base.bvecs'))
        xq = mmap_bvecs(os.path.join(dataset_dir, 'bigann_query.bvecs'))
        gt = ivecs_read(os.path.join(dataset_dir, 'gnd/idx_%dM.ivecs' % dbsize))

        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        # print(xb[0])
        # print(xb[250000])
        

        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)
        gt = np.array(gt, dtype=np.int32)

    elif dbname.startswith('Deep'):
        # Deep1M to Deep1000M
        dbsize = int(dbname[4:-1])
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/deep1b'

        xb = read_deep_fbin(os.path.join(dataset_dir, 'base.1B.fbin'))
        xq = read_deep_fbin(os.path.join(dataset_dir, 'query.public.10K.fbin'))
        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

    elif dbname.startswith('GLOVE'):
        dbsize = 2
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/GLOVE_840B_300d'
    
        xb = read_deep_fbin(os.path.join(dataset_dir, 'glove.840B.300d.fbin'))
        xq = read_deep_fbin(os.path.join(dataset_dir, 'query_10K.fbin'))
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

    elif dbname.startswith('SBERT1M'):
        dbsize = 1
        # assert dbname[:5] == 'SBERT' 
        # assert dbname[-1] == 'M'
        # dbsize = int(dbname[5:-1]) # in million
        
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/sbert'
        xb = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'sbert1M.fvecs'), num_vec=int(dbsize * 1e6))
        # xb = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'sbert3B.fvecs'), num_vec=int(dbsize * 1e6))
        xq = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'query_10K.fvecs'), num_vec=10 * 1000)
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

        # trim to correct size
        xb = xb[:dbsize * 1000 * 1000]
        
    else:
        print('unknown dataset', dbname, file=sys.stderr)
        sys.exit(1)

    print("Vector shapes:")
    print("Base vector xb: ", xb.shape)
    print("Query vector xq: ", xq.shape)
    print("Ground truth gt: ", gt.shape)


    # Construct the full graph index
    index_path_full = os.path.join(hnsw_path, '{}_index_MD{}_par1_0.bin'.format(dbname, args.MD))
    if not os.path.exists(index_path_full):
        # Load the entire graph and build full index
        print(f"Start full index construction for [{dbname}]")
        p_full = hnswlib.Index(space='l2', dim=xb.shape[1])  # possible options are l2, cosine or ip
        p_full.init_index(max_elements=xb.shape[0], ef_construction=ef_construction, M=M)
        p_full.add_items(xb)
        p_full.save_index(index_path_full)
    else:
        # Load the index
        print(f"Loading full index for [{dbname}]")
        p_full = hnswlib.Index(space='l2', dim=xb.shape[1])
        p_full.load_index(index_path_full)
        
    # Construct the sub-graph indexes
    p = []
    for i in range(sub_graph_num):

        N_VEC = int(dbsize * 1000 * 1000 / sub_graph_num)
        xb_sub = xb[i * N_VEC : (i+1) * N_VEC]
        # print(xb_sub[0])
        dim = xb_sub.shape[1] # should be 128
        nq = xq.shape[0]

        index_path_sub = os.path.join(hnsw_path, '{}_index_MD{}_par{}_{}.bin'.format(dbname, args.MD, sub_graph_num, i))
        
        # train index if not exist
        if not os.path.exists(index_path_sub):
            print(f"Start subgraph index construction for [{dbname}], subgraph [{i}]")
            # Declaring index
            p.append(hnswlib.Index(space='l2', dim=dim))  # possible options are l2, cosine or ip

            # Initing index
            # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
            # during insertion of an element.
            # The capacity can be increased by saving/loading the index, see below.
            #
            # ef_construction - controls index search speed/build speed tradeoff
            #
            # M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
            # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
            p[i].init_index(max_elements=N_VEC, ef_construction=ef_construction, M=M)

            # Set number of threads used during batch search/construction
            # By default using all available cores
            # p.set_num_threads(16)

            batch_size = 10000
            batch_num = int(np.ceil(N_VEC / batch_size))
            for j in range(batch_num):
                # print("Adding {} th batch of {} elements".format(i, batch_size))
                xbatch = xb_sub[j * batch_size: (j + 1) * batch_size]
                p[i].add_items(xbatch)
            
            # Serializing and deleting the index:
            print("Saving index to '%s'" % index_path_sub)
            p[i].save_index(index_path_sub)
        
        else:
            # Load the index
            print(f"Loading subgraph index for [{dbname}], subgraph [{i}]")
            p.append(hnswlib.Index(space='l2', dim=dim))
            p[i].load_index(index_path_sub)


    # Run c++ binary for subgraph search
    # SIFT1M /mnt/scratch/hanghu/CPU_hnsw_index/SIFT1M_index_MD64_par4 32 0 1 10000
    hnsw_index_prefix = os.path.join(hnsw_path, '{}_index_MD{}_par{}'.format(dbname, args.MD, sub_graph_num))
    # There are some fixed search parameters
    cmd_hnsw_search = f"../hnswlib/build/main {dbname} {hnsw_index_prefix} {sub_graph_num} {ef_sub} 0 1 10000 {subgraph_result_path} > {log_perf_test}"
    print(f"Running sub-graph search command: {cmd_hnsw_search}")
    os.system(cmd_hnsw_search)
    _, _, node_counter_sub = read_from_log(log_perf_test)
    
    # Run c++ binary for full graph search: sub_graph_num = 1
    hnsw_index_prefix = os.path.join(hnsw_path, '{}_index_MD{}_par1'.format(dbname, args.MD))
    cmd_hnsw_search = f"../hnswlib/build/main {dbname} {hnsw_index_prefix} 1 {ef_full} 0 1 10000 {subgraph_result_path} > {log_perf_test}"
    print(f"Running full-graph search command: {cmd_hnsw_search}")
    os.system(cmd_hnsw_search)
    recall_1_full_ori, recall_10_full_ori, node_counter_full = read_from_log(log_perf_test)
    
    # os.system(f"rm {log_perf_test}")
    
    
    # Build the verification graph index
    index_path_ver = os.path.join(hnsw_path, '{}_index_ver.bin'.format(dbname))
    if not os.path.exists(index_path_ver):
        # build ver index
        print(f"Start verification index construction for [{dbname}]")
        p_ver = hnswlib.Index(space='l2', dim=xb.shape[1])  # possible options are l2, cosine or ip
        p_ver.init_index(max_elements=xb.shape[0], ef_construction=256, M=128)
        p_ver.add_items(xb)
        p_ver.save_index(index_path_ver)
        p_ver.set_ef(256)
    else:
        # Load the index
        print(f"Loading verification index for [{dbname}]")
        p_ver = hnswlib.Index(space='l2', dim=xb.shape[1])
        p_ver.load_index(index_path_ver)
        p_ver.set_ef(256)


    # Convert sub-graph result ids to full graph ids
    # and calculate recall
    recall_1_sub = 0
    recall_10_sub = 0
    for i in range(sub_graph_num):
        # read search result
        subgraph_result_file = os.path.join(subgraph_result_path, f'{dbname}_{sub_graph_num}_{i}.txt')
        res_id = np.loadtxt(subgraph_result_file)
        
        res_id = res_id.flatten()
        res_items = p[i].get_items(res_id)
        
        # search in the full index
        print("Searching in the verification index...")
        I, _ = p_ver.knn_query(res_items, k=1)
        # print(I[:10])
        I = I.reshape(10000, 10)
        # print(I[0])

        recall_1_sub += calculate_recall(I, gt, 1)
        recall_10_sub += calculate_recall(I, gt, 10)
    
    
    # Measure the recall loss due to the conversion
    # by comparing the recall_x_full_ori with recall_x_full_con
    subgraph_result_file = os.path.join(subgraph_result_path, f'{dbname}_1_0.txt')
    res_id = np.loadtxt(subgraph_result_file)
    res_id = res_id.flatten()
    res_items = p_full.get_items(res_id)
    I, _ = p_ver.knn_query(res_items, k=1)
    I = I.reshape(10000, 10)
    recall_1_full_con = calculate_recall(I, gt, 1)
    recall_10_full_con = calculate_recall(I, gt, 10)
    
        
        
    print("Sub-graph search recall@1: ", recall_1_sub)
    print("Full graph search (ori) recall@1: ", recall_1_full_ori)
    print("Full graph search (con) recall@1: ", recall_1_full_con)
    print("Sub-graph search recall@10: ", recall_10_sub)
    print("Full graph search (ori) recall@10: ", recall_10_full_ori)
    print("Full graph search (con) recall@10: ", recall_10_full_con)
    
    print("==============================================")
    
    print("Sub-graph search node counter (per query): ", node_counter_sub)
    print("Full graph search node counter (per query): ", node_counter_full)    
    