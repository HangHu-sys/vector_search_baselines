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

See `scripts_hnsw/README.md`

## TODO

Wenqi added:

- [ ] Recall fluctuates? can we fix those randomness at least by fixing seeds? This would be a major problem later on for debugging.   