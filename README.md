# HNSW and NSG baselines

This repo contains all the baseline experiments for HNSW and NSG, as well as the index conversion scripts for FPGAs.


**Clone the repository**

```shell
git clone --recursive https://github.com/PomeloTea0726/hnsw_experiments.git
cd hnsw_experiments
```

To update:
```
git pull --recurse-submodules
```

### NSG

**Compile nsg**
```shell
mkdir nsg/build
cd nsg/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16
```

See `scripts_nsg/README.md`

### HNSW

**Compile hnswlib**

```
mkdir hnswlib/build
cd hnswlib/build
cmake ..
make 
```

See `scripts_hnsw/README.md`

## TODO

Wenqi added:

- [ ] @Hang, add the description to the changes in NSG in the REAMDE.md in scripts_nsg