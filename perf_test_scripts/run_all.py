'''

'''


import os
import numpy as np
import argparse
import json

def config_exist_in_dict(json_dict:dict,
        dataset:str, algo:str, M:str, ef_construction:str, ef:str, k:str, omp:str, interq_multithread:str, batch_size:str):
    """
    Given the input dict and config, check if the entry is in the dict

    Note: keys should be in string format

    Return: True (already exist) or False (not exist)
    """
    if dataset not in json_dict:
        return False
    if algo not in json_dict[dataset]:
        return False
    if M not in json_dict[dataset][algo]:
        return False
    if ef_construction not in json_dict[dataset][algo][M]:
        return False
    if ef not in json_dict[dataset][algo][M][ef_construction]:
        return False
    if k not in json_dict[dataset][algo][M][ef_construction][ef]:
        return False
    if omp not in json_dict[dataset][algo][M][ef_construction][ef][k]:
        return False
    if interq_multithread not in json_dict[dataset][algo][M][ef_construction][ef][k][omp]:
        return False
    if batch_size not in json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread]:
        return False
    if "accuracy" not in json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread][batch_size] or \
        "node_count" not in json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread][batch_size]:
        return False
    return True


def read_from_log(log_fname:str):
    """
    Read accuracy and node_count from the log file
    """
    node_counter = None
    accuracy = None

    with open(log_fname, 'r') as file:
        for line in file:
            if "node counter:" in line:
                node_counter = [int(x) for x in line.split(':')[1].split()]
            elif "accuracy:" in line:
                accuracy = float(line.split(':')[1])
                break

    return node_counter, accuracy


def write_to_json(out_json_fname:str, json_dict:dict, overwrite:int,
        dataset:str, algo:str, M:str, ef_construction:str, ef:str, k:str, omp:str, interq_multithread:str, batch_size:str,
        accuracy:float, node_count:int):
    """
    Write accuracy and node_count to the json file
    """
    if dataset not in json_dict:
        json_dict[dataset] = dict()
    if algo not in json_dict[dataset]:
        json_dict[dataset][algo] = dict()
    if M not in json_dict[dataset][algo]:
        json_dict[dataset][algo][M] = dict()
    if ef_construction not in json_dict[dataset][algo][M]:
        json_dict[dataset][algo][M][ef_construction] = dict()
    if ef not in json_dict[dataset][algo][M][ef_construction]:
        json_dict[dataset][algo][M][ef_construction][ef] = dict()
    if k not in json_dict[dataset][algo][M][ef_construction][ef]:
        json_dict[dataset][algo][M][ef_construction][ef][k] = dict()
    if omp not in json_dict[dataset][algo][M][ef_construction][ef][k]:
        json_dict[dataset][algo][M][ef_construction][ef][k][omp] = dict()
    if interq_multithread not in json_dict[dataset][algo][M][ef_construction][ef][k][omp]:
        json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread] = dict()
    if batch_size not in json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread]:
        json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread][batch_size] = dict()
    
    if not overwrite and \
        "accuracy" in json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread][batch_size] and \
        "node_count" in json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread][batch_size]:
        return
    else:
        json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread][batch_size]['accuracy'] = accuracy
        json_dict[dataset][algo][M][ef_construction][ef][k][omp][interq_multithread][batch_size]['node_count'] = node_count
        with open(out_json_fname, 'w') as f:
            json.dump(json_dict, f, indent=2)
        return


parser = argparse.ArgumentParser()
parser.add_argument('--out_json_fname', type=str, default='results.json')
parser.add_argument('--overwrite', type=int, default=0, help='0: continuously write, 1: overwrite')
parser.add_argument('--algo', type=str, default='hnsw', help='algorithm to use: hnsw, nsg')
parser.add_argument('--algo_path', type=str, default=None, help='path to algorithm')
parser.add_argument('--dataset', type=str, default='sift1m', help='dataset to use: sift1m')
parser.add_argument('--data_path', type=str, default=None, help='path to dataset')


args = parser.parse_args()

out_json_fname:str = args.out_json_fname
overwrite:int = args.overwrite
algo:str = args.algo
algo_path:str = args.algo_path
dataset:str = args.dataset
data_path:str = args.data_path

if not os.path.exists(algo_path):
    raise ValueError(f"Path to algorithm does not exist: {algo_path}")
if not os.path.exists(data_path):
    raise ValueError(f"Path to dataset does not exist: {data_path}")

if os.path.exists(out_json_fname):
    with open(out_json_fname, 'r') as f:
        json_dict = json.load(f)
else:
    json_dict = dict()

M_list = [4, 8, 16, 32, 64]
ef_construction_list = [40, 60, 80, 100, 120, 140, 160]
ef_list = [40, 60, 80, 100, 120, 140, 160]
k_list = [10]   # any ef should be larger than k
omp_list = [1]
interq_multithread_list = [16]
batch_size_list = [10, 50, 100, 500, 1000]

# TODO 1: add nsg
# TODO 2: add latency

if dataset == 'sift1m':
    if algo == 'hnsw':
        for M in M_list:
            for ef_construction in ef_construction_list:
                for ef in ef_list:
                    for k in k_list:
                        for omp in omp_list:
                            for interq_multithread in interq_multithread_list:
                                for batch_size in batch_size_list:
                                    
                                    print(f"Config: dataset={dataset}"
                                            f", algo={algo}"
                                            f", M={M}"
                                            f", ef_construction={ef_construction}"
                                            f", ef={ef}"
                                            f", k={k}"
                                            f", omp={omp}"
                                            f", interq_multithread={interq_multithread}"
                                            f", batch_size={batch_size}")
                                    if not overwrite and \
                                        config_exist_in_dict(json_dict,
                                                            dataset, algo, str(M), str(ef_construction), str(ef), str(k), str(omp), str(interq_multithread), str(batch_size)):
                                        print("Already exist in json, skip")
                                        continue
                                    
                                    log_perf_test = 'perf_test.log'
                                    cmd_perf_test = f"{algo_path} " + \
                                                    f"{M} " + \
                                                    f"{ef_construction} " + \
                                                    f"{ef} " + \
                                                    f"{k} " + \
                                                    f"{omp} " + \
                                                    f"{interq_multithread} " + \
                                                    f"{batch_size} " + \
                                                    f"{data_path}" + \
                                                    f" > {log_perf_test}"
                                    
                                    print(f"Running command:\n{cmd_perf_test}")
                                    os.system(cmd_perf_test)
                                    
                                    node_counter, accuracy = read_from_log(log_perf_test)
                                    print(f"Accuracy: {accuracy}")
                                    write_to_json(out_json_fname, json_dict, overwrite, 
                                                    dataset, algo, str(M), str(ef_construction), str(ef), str(k), str(omp), str(interq_multithread), str(batch_size),
                                                    accuracy, node_counter)
            
                cmd_rm_bin = f"rm ./sift1m_ef_{ef_construction}_M_{M}.bin"
                os.system(cmd_rm_bin)                        
                                    
                                    



