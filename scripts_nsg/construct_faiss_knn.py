import faiss
import os
import numpy as np
import struct
import argparse
from utils import mmap_bvecs, mmap_bvecs_SBERT

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sift1m', help='dataset to use: sift1m')
    parser.add_argument('--construct_K', type=int, default=10, help='K value for KNN graph construction')

    args = parser.parse_args()
    dataset:str = args.dataset
    construct_K:int = args.construct_K
    
    print("Load data...")
    
    if dataset.startswith('sift'):
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

    D, I = index.search(X, construct_K)

    print("Save knn graph...")

    filename = f"/mnt/scratch/hanghu/nsg_experiments/KNN_graphs/{dataset}_{construct_K}NN.graph"
    with open(filename, "wb") as f:
        for i in range(N):
            f.write(struct.pack('i', construct_K))
            
            for j in range(construct_K):
                f.write(struct.pack('i', I[i, j]))