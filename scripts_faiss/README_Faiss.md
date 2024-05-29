# Faiss scripts

## CPU Faiss

### Verify that with PQ cannot reach the same level of recall as graph: 

For SIFT, using PQ=128 (same as dim), barely achieves good recall.

```
python cpu_faiss.py --faiss_index_path ../data/CPU_Faiss_indexes --dbname SIFT1M --index_key IVF1,PQ128 --mode run_one --topK 100 --qbs 100 --parametersets 'nprobe=1'

nprobe=1 	      	 R@1    R@10   R@100
0.9913 0.9928 0.9944 
QPS = 53.080952381910585
```

using PQ=64, already cannot achieve the same level of recall as graphs (only 87%).

```
python cpu_faiss.py --faiss_index_path ../data/CPU_Faiss_indexes --dbname SIFT1M --index_key IVF1,PQ64 --mode run_one --topK 100 --qbs 100 --parametersets 'nprobe=1'

nprobe=1 	      	 R@1    R@10   R@100
0.8343 0.8764 0.9074 
```

### Evaluate CPU latency and throughput

Running the following commands will automatically construct indexes if they are not yet.

We set `nlist=1024` for 1M datasets and `nlist=4096`, approximately being the square root of number of database vectors. 

First check the parameters set in the scripts:

```
batch_size_list = [1, 2, 4, 8, 16, 32, 10000]
```

Run performance tests:

```
python cpu_faiss.py --dbname SIFT1M --index_key IVF1024,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 64 --perf_df_path perf_df_faiss_cpu.pickle
python cpu_faiss.py --dbname Deep1M --index_key IVF1024,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 64 --perf_df_path perf_df_faiss_cpu.pickle
python cpu_faiss.py --dbname SPACEV1M --index_key IVF1024,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 64 --perf_df_path perf_df_faiss_cpu.pickle

python cpu_faiss.py --dbname SIFT10M --index_key IVF4096,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 128 --perf_df_path perf_df_faiss_cpu.pickle
python cpu_faiss.py --dbname Deep10M --index_key IVF4096,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 128 --perf_df_path perf_df_faiss_cpu.pickle
python cpu_faiss.py --dbname SPACEV10M --index_key IVF4096,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 128 --perf_df_path perf_df_faiss_cpu.pickle
```

## GPU Faiss

### Evaluate CPU latency and throughput

Running the following commands will automatically construct indexes if they are not yet.

We set `nlist=1024` for 1M datasets and `nlist=4096`, approximately being the square root of number of database vectors. 

First check the parameters set in the scripts:

```
batch_size_list = [1, 2, 4, 8, 16, 32, 10000]
```

Run performance tests:

```
# Optionally specify which GPU to use
export CUDA_VISIBLE_DEVICES=1

python cpu_faiss.py --dbname SIFT1M --index_key IVF1024,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 64 --use_gpu 1 --perf_df_path perf_df_faiss_gpu.pickle
python cpu_faiss.py --dbname Deep1M --index_key IVF1024,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 64 --use_gpu 1 --perf_df_path perf_df_faiss_gpu.pickle
python cpu_faiss.py --dbname SPACEV1M --index_key IVF1024,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 64 --use_gpu 1 --perf_df_path perf_df_faiss_gpu.pickle

python cpu_faiss.py --dbname SIFT10M --index_key IVF4096,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 128 --use_gpu 1 --perf_df_path perf_df_faiss_gpu.pickle
python cpu_faiss.py --dbname Deep10M --index_key IVF4096,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 128 --use_gpu 1 --perf_df_path perf_df_faiss_gpu.pickle
python cpu_faiss.py --dbname SPACEV10M --index_key IVF4096,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 128 --use_gpu 1 --perf_df_path perf_df_faiss_gpu.pickle
```
