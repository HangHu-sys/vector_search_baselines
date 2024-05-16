# HNSW scripts

## Construct HNSW Indexes

First, construct HNSW:

```
# SIFT1M
python construct_and_search_hnsw.py --dbname SIFT1M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SIFT1M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SIFT1M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# SIFT10M
python construct_and_search_hnsw.py --dbname SIFT10M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SIFT10M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SIFT10M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# Deep1M
python construct_and_search_hnsw.py --dbname Deep1M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname Deep1M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname Deep1M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# Deep10M
python construct_and_search_hnsw.py --dbname Deep10M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname Deep10M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname Deep10M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# SBERT1M
python construct_and_search_hnsw.py --dbname SBERT1M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SBERT1M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SBERT1M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes
```

Second: convert to FPGA format:

```
# SIFT1M
python hnsw_to_FPGA.py --dbname SIFT1M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT1M_MD64
python hnsw_to_FPGA.py --dbname SIFT1M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT1M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT1M_MD32
python hnsw_to_FPGA.py --dbname SIFT1M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT1M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT1M_MD16

# SIFT10M
python hnsw_to_FPGA.py --dbname SIFT10M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT10M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT10M_MD64
python hnsw_to_FPGA.py --dbname SIFT10M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT10M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT10M_MD32
python hnsw_to_FPGA.py --dbname SIFT10M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT10M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT10M_MD16

# Deep1M
python hnsw_to_FPGA.py --dbname Deep1M --CPU_index_path ../data/CPU_hnsw_indexes/Deep1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/Deep1M_MD64
python hnsw_to_FPGA.py --dbname Deep1M --CPU_index_path ../data/CPU_hnsw_indexes/Deep1M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/Deep1M_MD32
python hnsw_to_FPGA.py --dbname Deep1M --CPU_index_path ../data/CPU_hnsw_indexes/Deep1M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/Deep1M_MD16

# Deep10M
python hnsw_to_FPGA.py --dbname Deep10M --CPU_index_path ../data/CPU_hnsw_indexes/Deep10M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/Deep10M_MD64
python hnsw_to_FPGA.py --dbname Deep10M --CPU_index_path ../data/CPU_hnsw_indexes/Deep10M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/Deep10M_MD32
python hnsw_to_FPGA.py --dbname Deep10M --CPU_index_path ../data/CPU_hnsw_indexes/Deep10M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/Deep10M_MD16

# SBERT1M
python hnsw_to_FPGA.py --dbname SBERT1M --CPU_index_path ../data/CPU_hnsw_indexes/SBERT1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SBERT1M_MD64
python hnsw_to_FPGA.py --dbname SBERT1M --CPU_index_path ../data/CPU_hnsw_indexes/SBERT1M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/SBERT1M_MD32
python hnsw_to_FPGA.py --dbname SBERT1M --CPU_index_path ../data/CPU_hnsw_indexes/SBERT1M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/SBERT1M_MD16
```

## Evaluate CPU latency & throughput

Before running the following commands, make sure the configurations in the scripts are properly set:, e.g., 
```
    max_degree_list = [64]
    ef_list = [16, 32, 48, 64, 80, 96]
    omp_list = [1] # 1 = enable; 0 = disable
    batch_size_list = [1, 2, 4, 8, 16, 10000]
```

Now run (make sure `max_cores` is properly set):
```
python run_all_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df.pickle --dataset SIFT1M --max_cores 16
python run_all_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df.pickle --dataset SIFT10M --max_cores 16
python run_all_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df.pickle --dataset Deep1M --max_cores 16
python run_all_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df.pickle --dataset Deep10M --max_cores 16
```