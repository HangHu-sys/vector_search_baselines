# NSG scripts

## We applied some changes to the NSG folder

@Hang

1. 

2. 

## Construct NSG Indexes

First: construct the exact kNN graph using GPU faiss:

```
python construct_faiss_knn.py --dbname SIFT1M --construct_K 200 --output_path ../data/CPU_knn_graphs
python construct_faiss_knn.py --dbname SIFT10M --construct_K 200 --output_path ../data/CPU_knn_graphs
python construct_faiss_knn.py --dbname Deep1M --construct_K 200 --output_path ../data/CPU_knn_graphs
python construct_faiss_knn.py --dbname Deep10M --construct_K 200 --output_path ../data/CPU_knn_graphs
python construct_faiss_knn.py --dbname SBERT1M --construct_K 200 --output_path ../data/CPU_knn_graphs
```

Second: construct NSG

```
python run_all_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SIFT1M

python run_all_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SIFT10M

python run_all_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset Deep1M

python run_all_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset Deep10M

# python run_all_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SBERT1M
```


Third: convert to FPGA format

```
# SIFT1M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT1M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT1M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT1M --max_degree 16

# SIFT10M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT10M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT10M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT10M --max_degree 16

# Deep1M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep1M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep1M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep1M --max_degree 16

# Deep10M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep10M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep10M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep10M --max_degree 16

# SBERT1M
# python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SBERT1M --max_degree 64
# python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SBERT1M --max_degree 32
# python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SBERT1M --max_degree 16
```

## Evaluate CPU latency & throughput

Before running the following commands, make sure the configurations in the scripts are properly set:, e.g., 
```
	search_L_list = [64, 80, 100]
	omp_list = [1] # 1 = enable; 0 = disable
	batch_size_list = [1, 2, 4, 8, 16, 10000]
	construct_R_list = [64] # R means max degree
```

Now run (make sure `max_cores` is properly set):
```
python run_all_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset SIFT1M --perf_df_path perf_df.pickle --max_cores 16
python run_all_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset SIFT10M --perf_df_path perf_df.pickle --max_cores 16
python run_all_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset Deep1M --perf_df_path perf_df.pickle --max_cores 16
python run_all_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset Deep10M --perf_df_path perf_df.pickle --max_cores 16
```
