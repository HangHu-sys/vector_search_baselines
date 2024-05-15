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