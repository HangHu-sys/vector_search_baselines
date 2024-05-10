'''
    This script runs c++ nsg to measure the recall and search time.
    Example Usage:
        python run_all.py --perf_df_path perf_df.pickle \
            --nsg_con_bin_path ../nsg/build/tests/test_nsg_index \
            --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search \
            --dataset SIFT1M
    
    The latest result can be viewed in temp log: "./nsg.log".
    Currently we support SIFT1M, and SBERT1M.
'''
import os
import argparse
import glob
import pandas as pd

def read_from_log(log_fname:str):
    """
    Read recall and time_batch from the log file
    """
    num_batch = 0
    time_batch = []
    qps = 0.0
    recall = 0.0

    with open(log_fname, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if "qsize divided into" in line:
                num_batch = int(line.split(" ")[3])
            elif "recall" in line:
                recall = float(line.split(" ")[1])
            elif "time for each batch" in line:
                for i in range(num_batch):
                    values = lines[idx + 1 + i].split(" ")
                    time_batch.append(float(values[0]))
            elif "qps" in line:
                qps = float(line.split(" ")[1])

    return recall, time_batch, qps


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_knng_path', type=str, default="../data/CPU_knn_graphs", help='path to input knn graph for index construction')
    parser.add_argument('--output_nsg_path', type=str, default="../data/CPU_NSG_index", help='path to output constructed nsg index')
    parser.add_argument('--perf_df_path', type=str, default='perf_df.pickle', help='path to save the performance dataframe')
    parser.add_argument('--nsg_con_bin_path', type=str, default=None, help='path to nsg construction algorithm')
    parser.add_argument('--nsg_search_bin_path', type=str, default=None, help='path to nsg search algorithm')
    parser.add_argument('--dataset', type=str, default='SIFT1M', help='dataset to use: SIFT1M')
    parser.add_argument('--mode', type=str, default='search', help='mode to use: either construct or search', choices=['construct', 'search'])

    args = parser.parse_args()

    perf_df_path:str = args.perf_df_path
    nsg_con_bin_path:str = args.nsg_con_bin_path
    nsg_search_bin_path:str = args.nsg_search_bin_path
    dataset:str = args.dataset

    if not os.path.exists(nsg_con_bin_path):
        raise ValueError(f"Path to algorithm does not exist: {nsg_con_bin_path}")
    if not os.path.exists(nsg_search_bin_path):
        raise ValueError(f"Path to algorithm does not exist: {nsg_search_bin_path}")

    key_columns = ['dataset', 'max_degree', 'search_L', 'omp', 'interq_multithread', 'batch_size']
    result_columns = ['recall_1', 'recall_10' 'time_batch', 'qps']
    columns = key_columns + result_columns

    if os.path.exists(perf_df_path): # load existing
        df = pd.read_pickle(perf_df_path)
        assert len(df.columns.values) == len(columns)
        for col in columns:
            assert col in df.columns.values
    else:
        df = pd.DataFrame(columns=columns)
    pd.set_option('display.expand_frame_repr', False)

    construct_K_list_const = [200] # knn graph
    construct_L_list_const = [50] # a magic number that controls graph construction quality
    construct_C_list_const = [500] # another magic number that controls graph construction quality
    construct_R_list = [16, 32, 64] # R means max degree
    
    search_L_list = [100]
    search_K_list = [10]
    omp_list = [0]
    omp_interq_multithread_list = [1]
    batch_size_list = [2000]
    
    log_nsg = 'nsg.log'  
    
    for construct_K in construct_K_list:        
        # Check if KNN graph file exists
        knng_path = os.path.join(args.input_knng_path, f"{dataset}_{construct_K}NN.graph")
        if not glob.glob(knng_path):
            print(f"KNN graph file does not exist: {knng_path}, use command:")
            print(f"    python construct_faiss_knn.py --dataset {dataset} --construct_K {construct_K}")
            exit()
        print(f"KNN graph file exists: {knng_path}")
        
        for construct_L, R, C in construct_L_R_C_list:
            # Construct NSG index
            nsg_file = os.path.join(args.output_nsg_path, f"{dataset}_{construct_L}_{R}_{C}.nsg")
            if glob.glob(nsg_file):
                print(f"NSG index file already exists: {nsg_file}")
            else:
                print(f"Generating NSG index file: {nsg_file}")
                assert os.path.exists(knng_path)
                cmd_build_nsg = f"{nsg_con_bin_path} {dataset} {knng_path} {construct_L} {R} {C} {nsg_file}"
                os.system(cmd_build_nsg)
                
                if not glob.glob(nsg_file):
                    print("NSG index file construction failed")
                    exit()
                print("NSG index file construction succeeded")
            
            # Perform the search
            for search_L in search_L_list:
                for search_K in search_K_list:
                    for omp in omp_list:
                        
                        if omp == 0:
                            interq_multithread_list = [1]
                        else:
                            interq_multithread_list = omp_interq_multithread_list
                            
                        for interq_multithread in interq_multithread_list:
                            for batch_size in batch_size_list:
                                
                                cmd_search_nsg = f"{nsg_search_bin_path} " + \
                                                f"{dataset} " + \
                                                f"{nsg_file} " + \
                                                f"{search_L} " + \
                                                f"{search_K} " + \
                                                f"{omp} " + \
                                                f"{interq_multithread} " + \
                                                f"{batch_size}" + \
                                                f" > {log_nsg}"
                                print(f"Searching NSG by running command:\n{cmd_search_nsg}")
                                os.system(cmd_search_nsg)

                                recall, time_batch, qps = read_from_log(log_nsg)
                                # print(f"Recall: {recall}, Time Batch: {time_batch}, QPS: {qps}")
                                # if already in the df, delete the old row first
                                key_values = {
                                    'dataset': dataset,
                                    'construct_K': construct_K,
                                    'construct_L_R_C': (construct_L, R, C),
                                    'search_L': search_L,
                                    'search_K': search_K,
                                    'omp': omp,
                                    'interq_multithread': interq_multithread,
                                    'batch_size': batch_size
                                }
                                if len(df) > 0:
                                    idx = df.index[(df['dataset'] == dataset) & \
                                                    (df['construct_K'] == construct_K) & \
                                                    (df['construct_L_R_C'] == (construct_L, R, C)) & \
                                                    (df['search_L'] == search_L) & \
                                                    (df['search_K'] == search_K) & \
                                                    (df['omp'] == omp) & \
                                                    (df['interq_multithread'] == interq_multithread) & \
                                                    (df['batch_size'] == batch_size)]
                                    if len(idx) > 0:
                                        df = df.drop(idx)
                                    df = df.append({**key_values, 'recall': recall, 'time_batch': time_batch, 'qps': qps}, ignore_index=True)
    
    if args.perf_df_path is not None:
        df.to_pickle(args.perf_df_path, protocol=4)