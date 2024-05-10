"""
Example Usage:
    python construct_faiss_knn.py --dataset SIFT1M --construct_K 200 --output_path ../data/CPU_knn_graphs
    python construct_faiss_knn.py --dataset SIFT10M --construct_K 200 --output_path ../data/CPU_knn_graphs
    python construct_faiss_knn.py --dataset SBERT1M --construct_K 200 --output_path ../data/CPU_knn_graphs
"""


import faiss
import os
import numpy as np
import struct
import argparse
from utils import mmap_bvecs, mmap_bvecs_SBERT

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sift1m', help='dataset to use: sift1m')
    parser.add_argument('--construct_K', type=int, default=200, help='K value for KNN graph construction')
    parser.add_argument('--output_path', type=str, default='../data/CPU_knn_graphs', help='Path to save the KNN graph')
    parser.add_argument('--batch_size', type=int, default=1000 * 1000, help='Batch size for KNN graph construction')

    args = parser.parse_args()
    dataset:str = args.dataset
    construct_K:int = args.construct_K
    
    print("Load data...")
    
    if dataset.startswith('SIFT'):
        # sift1m to sift1000m
        dbsize = int(dataset[4:-1])
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/bigann'
        xb = mmap_bvecs(os.path.join(dataset_dir, 'bigann_base.bvecs'))
        
        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        
    elif dataset.startswith('SBERT1M'):
        dbsize = 1
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/sbert'
        xb = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'sbert1M.fvecs'))
        
        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        
    print("Data loaded: ", xb.shape)
    
    X = xb
    d = X.shape[1]
    N = X.shape[0]

    res = faiss.StandardGpuResources()  # use a single GPU

    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)

    print("Train index...")

    index.add(X)

    print("Search index...")

    # D, I = index.search(X, construct_K)
    # search results in batch
    D = np.zeros((N, construct_K), dtype=np.float32)
    I = np.zeros((N, construct_K), dtype=np.int32)
    for i in range(0, N, args.batch_size):
        print("Batch: ", i)
        X_batch = X[i:i + args.batch_size]
        D_batch, I_batch = index.search(X_batch, construct_K)
        D[i:i + args.batch_size] = D_batch
        I[i:i + args.batch_size] = I_batch

    print("Save knn graph...")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    filename = os.path.join(args.output_path, f"{dataset}_{construct_K}NN.graph")
    with open(filename, "wb") as f:
        for i in range(N):
            f.write(struct.pack('i', construct_K))
            
            for j in range(construct_K):
                f.write(struct.pack('i', I[i, j]))