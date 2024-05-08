'''
    This script runs c++ nsg to measure the recall and search time.
    Example Usage:
        python run_all.py --save_df perf_df.pickle \
            --nsg_con_path ../nsg/build/tests/test_nsg_index \
            --nsg_search_path ../nsg/build/tests/test_nsg_optimized_search \
            --dataset sift1m
    
    The latest result can be viewed in temp log: "./nsg.log".
    Currently we support sift1m, and SBERT1M.
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
    parser.add_argument('--save_df', type=str, default='perf_df.pickle', help='path to save the performance dataframe')
    parser.add_argument('--nsg_con_path', type=str, default=None, help='path to nsg construction algorithm')
    parser.add_argument('--nsg_search_path', type=str, default=None, help='path to nsg search algorithm')
    parser.add_argument('--dataset', type=str, default='sift1m', help='dataset to use: sift1m')

    args = parser.parse_args()

    save_df:str = args.save_df
    nsg_con_path:str = args.nsg_con_path
    nsg_search_path:str = args.nsg_search_path
    dataset:str = args.dataset

    if not os.path.exists(nsg_con_path):
        raise ValueError(f"Path to algorithm does not exist: {nsg_con_path}")
    if not os.path.exists(nsg_search_path):
        raise ValueError(f"Path to algorithm does not exist: {nsg_search_path}")


    key_columns = ['dataset', 'construct_K', 'construct_L_R_C', 'search_L', 'search_K', 'omp', 'interq_multithread', 'batch_size']
    result_columns = ['recall', 'time_batch', 'qps']
    columns = key_columns + result_columns

    if os.path.exists(save_df): # load existing
        df = pd.read_pickle(save_df)
        assert len(df.columns.values) == len(columns)
        for col in columns:
            assert col in df.columns.values
    else:
        df = pd.DataFrame(columns=columns)
    pd.set_option('display.expand_frame_repr', False)

    construct_K_list = [200]
    construct_L_R_C_list = [(40, 50, 500)]
    search_L_list = [100]
    search_K_list = [10]
    omp_list = [0]
    omp_interq_multithread_list = [1]
    batch_size_list = [2000]
    
    log_nsg = 'nsg.log'  
    
    for construct_K in construct_K_list:        
        # Check if KNN graph file exists
        knng_path = f"/mnt/scratch/hanghu/nsg_experiments/KNN_graphs/{dataset}_{construct_K}NN.graph"
        if not glob.glob(knng_path):
            print(f"KNN graph file does not exist: {knng_path}, use command:")
            print(f"    python construct_faiss_knn.py --dataset {dataset} --construct_K {construct_K}")
            exit()
        print(f"KNN graph file exists: {knng_path}")
        
        for construct_L, R, C in construct_L_R_C_list:
            # Construct NSG index
            nsg_file = f"/mnt/scratch/hanghu/nsg_experiments/NSG_index/{dataset}_{construct_L}_{R}_{C}.nsg"
            if glob.glob(nsg_file):
                print(f"NSG index file already exists: {nsg_file}")
            else:
                print(f"Generating NSG index file: {nsg_file}")
                assert os.path.exists(knng_path)
                cmd_build_nsg = f"{nsg_con_path} {dataset} {knng_path} {construct_L} {R} {C} {nsg_file}"
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
                                
                                cmd_search_nsg = f"{nsg_search_path} " + \
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
    
    if args.save_df is not None:
        df.to_pickle(args.save_df, protocol=4)