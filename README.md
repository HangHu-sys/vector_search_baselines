# FPGA graph search


## NSG emulation

_This is a temporary guidance of compilation._

This is the python emulation of NSG vector search algorithm, including NSG index construction and NSG search.


### Prerequisites

* OpenBLAS (for efanna_graph)

### How to compile
1. **Clone the repository**
```shell
git clone --recursive https://github.com/PomeloTea0726/hnsw_experiments.git
cd hnsw_experiments
```
2. **Compile efanna_graph**

A knn graph is required for nsg index construction. We suggest you use [efanna\_graph](https://github.com/PomeloTea0726/efanna_graph) to build this knn graph. Currently you can directly fetch examples from [here](https://github.com/PomeloTea0726/nsg/#pre-built-knn-graph-and-nsg-index).

3. **Compile nsg**
```shell
cd nsg/
mkdir build/ && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

**IMPORTANT NOTE: Currently we only support the construction of nsg index from any given KNN-graph. Thus, you should ignore the compilation of efanna_graph, and provide your own KNN-graph.**

### Usage
```shell
cd nsg_emu/
python nsg.py \
--DATA_PATH [DATA_PATH] \
--QUERY_PATH [QUERY_PATH] \
--GT_PATH [GT_PATH] \
--KNNG_PATH [KNNG_PATH] \
--NSG_PATH [NSG_PATH]
```

* `DATA_PATH` is the path of the base data in `fvecs` format.
* `QUERY_PATH` is the path of the query data in `fvecs` format.
* `GT_PATH` is the path of the groundtruth data in `ivecs` format.
* `KNNG_PATH` is the path of the pre-built kNN graph.
* `NSG_PATH` is the path of the generated NSG index. If not found, then the script will construct one at this path.

The construction parameters `L`, `R`, `C` and search parameters `K`, `L_search` of NSG can be modified in `nsg.py`.

Here is one example of the above execution command on sift1m:
```shell
python nsg.py \
--DATA_PATH ../../dataset/sift/sift_base.fvecs \
--QUERY_PATH ../../dataset/sift/sift_query.fvecs \
--GT_PATH ../../dataset/sift/sift_groundtruth.ivecs \
--KNNG_PATH ../../dataset/sift_200nn.graph \
--NSG_PATH ../../dataset/sift/sift.nsg
```



## TODO

- [ ] Add efanna_graph support. (Require (1) OpenBLAS support, (2) Modification of the script `nsg.py`.)
- [ ] Add parameters `L`, `R`, `C`, `K`, `L_search` for the algorithm as input.
