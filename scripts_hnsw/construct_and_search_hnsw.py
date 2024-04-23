"""
Construct hnsw-graph & test  performance, assuming python version of hnswlib is installed
    If the graph is not constructed, construct the graph and save it.
    If the graph is already constructed, load the graph and test the performance (various ef, thread numbers, etc.)

Example Usage:
python construct_and_search_hnsw.py \
    --dbname SIFT1M \
    --M 32 \
    --hnsw_path ../data/CPU_hnsw_indexes
"""
import argparse
import sys
import os
import time 

import hnswlib
import numpy as np

from utils import mmap_fvecs, mmap_bvecs, ivecs_read, fvecs_read, read_deep_ibin, read_deep_fbin

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', type=str, default="SIFT1M", help='name of the database, e.g., SIFT10M, Deep10M, GLOVE')
    parser.add_argument('--ef_construction', type=int, default=128, help='ef construction parameter')
    parser.add_argument('--M', type=int, default=32, help='M0 is for base layer, M === M0 * 2')
    parser.add_argument('--hnsw_path', type=str, default="../data/CPU_hnsw_indexes", help='Path to the NSG index file')
    args = parser.parse_args()
    
    dbname = args.dbname
    ef_construction = args.ef_construction
    hnsw_path = args.hnsw_path
    M = args.M
    
    index_path = os.path.join(hnsw_path, '{}_index_M_{}.bin'.format(dbname, M))

    if dbname.startswith('SIFT'):
        # SIFT1M to SIFT1000M
        dbsize = int(dbname[4:-1])
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/bigann'
        xb = mmap_bvecs(os.path.join(dataset_dir, 'bigann_base.bvecs'))
        xq = mmap_bvecs(os.path.join(dataset_dir, 'bigann_query.bvecs'))
        gt = ivecs_read(os.path.join(dataset_dir, 'gnd/idx_%dM.ivecs' % dbsize))

        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]

        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)
        gt = np.array(gt, dtype=np.int32)

    elif dbname.startswith('Deep'):
        # Deep1M to Deep1000M
        dbsize = int(dbname[4:-1])
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/deep1b'

        xb = mmap_fvecs(os.path.join(dataset_dir, 'base.fvecs'))
        xq = mmap_fvecs(os.path.join(dataset_dir, 'deep1B_queries.fvecs'))
        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

    elif dbname.startswith('GLOVE'):
        dbsize = 2
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/GLOVE_840B_300d'
    
        xb = read_deep_fbin(os.path.join(dataset_dir, 'glove.840B.300d.fbin'))
        xq = read_deep_fbin(query_fbin_dir = os.path.join(dataset_dir, 'query_10K.fbin'))
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

    else:
        print('unknown dataset', dbname, file=sys.stderr)
        sys.exit(1)

    N_VEC = int(dbsize * 1000 * 1000)
    print("Vector shapes:")
    print("Base vector xb: ", xb.shape)
    print("Query vector xq: ", xq.shape)
    print("Ground truth gt: ", gt.shape)

    dim = xb.shape[1] # should be 128
    nq = xq.shape[0]


    # train index if not exist
    if not os.path.exists(index_path):
        # Declaring index
        p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

        # Initing index
        # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
        # during insertion of an element.
        # The capacity can be increased by saving/loading the index, see below.
        #
        # ef_construction - controls index search speed/build speed tradeoff
        #
        # M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
        # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
        p.init_index(max_elements=N_VEC, ef_construction=ef_construction, M=M)

        # Set number of threads used during batch search/construction
        # By default using all available cores
        # p.set_num_threads(16)

        batch_size = 10000
        batch_num = int(np.ceil(N_VEC / batch_size))
        for i in range(batch_num):
            print("Adding {} th batch of {} elements".format(i, batch_size))
            xbatch = xb[i * batch_size: (i + 1) * batch_size]
            p.add_items(xbatch)


        # Serializing and deleting the index:
        print("Saving index to '%s'" % index_path)
        p.save_index(index_path)

        # p.set_num_threads(1)
        # Query the elements for themselves and measure recall:
        start = time.time()
        I, D = p.knn_query(xq, k=100)
        end = time.time()
        t_consume = end - start

        print("Searching...")
        print(' ' * 4, '\t', 'R@1    R@10   R@100')
        for rank in 1, 10, 100:
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            print("{:.4f}".format(n_ok / float(nq)), end=' ')
        print("Search {} vectors in {} sec\tQPS={}".format(nq, t_consume, nq / t_consume))


    # If index exists, load the index
    else:
        p = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.

        print("\nLoading index from {}\n".format(index_path))

        # Increase the total capacity (max_elements), so that it will handle the new data
        p.load_index(index_path, max_elements=N_VEC)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        ef_set = [64]
        # ef_set = [8, 16, 32, 64, 128, 256]
        batch_size_set = [1, 4, 16, 64, 256, 1024, 10000]
        num_threads_set = [1, 2, 4, 8, 16, 32, 0]
        k_set = [10]
        # k_set = [1, 10]

        for ef in ef_set:
            for k in k_set:
                for num_threads in num_threads_set:
                    for batch_size in batch_size_set:

                        print("\nef = {} k = {} \tnum_threads = {}\tbatch_size = {}".format(ef, k, num_threads, batch_size))
                        p.set_ef(ef)
                        p.set_num_threads(num_threads)

                        I = np.zeros((nq, k), dtype=np.int64)
                        D = np.zeros((nq, k), dtype=np.float32)

                        # Query the elements for themselves and measure recall:
                        start = time.time()
                        batch_num = int(np.ceil(nq / batch_size))
                        for bid in range(0, nq, batch_size):
                            I[bid:bid+batch_size,:], D[bid:bid+batch_size,:] = p.knn_query(xq[bid:bid+batch_size], k=k, num_threads=num_threads)
                        # I, D = p.knn_query(xq, k=k)
                        end = time.time()
                        t_consume = end - start

                        # print("Searching...")
                        print(' ' * 4, '\t', 'R@1    R@10   R@100')
                        for rank in 1, 10, 100:
                            if rank > k:
                                break
                            n_ok = (I[:, :rank] == gt[:, :1]).sum()
                            print("{:.4f}".format(n_ok / float(nq)), end=' ')
                        print("\nSearch {} vectors in {:.2f} sec\tQPS={:.2f}\tPer-batch latency: {:.2f} ms".format(
                            nq, t_consume, nq / t_consume, t_consume / nq * 1000 * batch_size), flush=True)
