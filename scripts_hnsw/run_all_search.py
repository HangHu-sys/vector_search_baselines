'''
Sample command to run the script:
    python run_all_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main \
        --perf_df_path perf_df.pickle --dataset SIFT1M --max_cores 16
'''


import os
import argparse
import json
import glob
import pandas as pd

def read_from_log(log_fname:str):
    """
    Read recall and node_count from the log file
    """
    num_batch = 0
    node_count = []
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
            elif "node counter and time for each batch" in line:
                for i in range(num_batch):
                    values = lines[idx + 1 + i].split(" ")
                    node_count.append(int(values[0]))
                    time_batch.append(float(values[1]))
            elif "qps" in line:
                qps = float(line.split(" ")[1])

    return node_count, time_batch, qps, recall


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--perf_df_path', type=str, default='perf_df.pickle')
    parser.add_argument('--hnsw_search_bin_path', type=str, default=None, help='path to algorithm bin')
    parser.add_argument('--hnsw_index_path', type=str, default="../data/CPU_hnsw_indexes", help='path to output constructed nsg index')
    parser.add_argument('--dataset', type=str, default='SIFT1M', help='dataset to use: SIFT1M')
    parser.add_argument('--max_cores', type=int, default=16, help='maximum number of cores used for search')

    args = parser.parse_args()

    perf_df_path:str = args.perf_df_path
    hnsw_search_bin_path:str = args.hnsw_search_bin_path
    hnsw_index_path:str = args.hnsw_index_path
    dataset:str = args.dataset
    max_cores:int = args.max_cores

    hnsw_index_path = os.path.abspath(hnsw_index_path)

    # random num log
    log_perf_test = 'hnsw_{}.log'.format(np.random.randint(1000000))

    if not os.path.exists(hnsw_search_bin_path):
        raise ValueError(f"Path to algorithm does not exist: {hnsw_search_bin_path}")

    key_columns = ['dataset', 'max_degree', 'ef', 'omp_enable', 'max_cores', 'batch_size']
    result_columns = ['recall_1', 'recall_10', 'time_batch', 'qps']
    columns = key_columns + result_columns

    if os.path.exists(perf_df_path): # load existing
        df = pd.read_pickle(perf_df_path)
        assert len(df.columns.values) == len(columns)
        for col in columns:
            assert col in df.columns.values
        print("Performance dataframe loaded")
    else:
        print(f"Performance dataframe does not exist, create a new one: {perf_df_path}")
        df = pd.DataFrame(columns=columns)
    pd.set_option('display.expand_frame_repr', False)

    # Full grid search
    max_degree_list = [64]
    ef_list = [16, 32, 48, 64, 80, 96]
    k_list = [10]  # any ef should be larger than k
    omp_list = [1] # 1 = enable; 0 = disable
    batch_size_list = [1, 2, 4, 8, 16, 10000]

    for max_degree in max_degree_list:

        hnsw_index_file = os.path.join(hnsw_index_path, f"{dataset}_index_MD{max_degree}.nsg")
        assert os.path.exists(hnsw_index_file)

        for ef in ef_list:
            for k in k_list:
                for omp in omp_list:
                    if omp == 0:
                        interq_multithread = 1
                    else:
                        if batch_size > max_cores:
                            interq_multithread = max_cores
                        else:
                            interq_multithread = batch_size
                        
                    for batch_size in batch_size_list:
                        
						#### TODO: @Hang refer to scripts_nsg/run_all_construct_and_search.py
                        # Use df to save the performance
                        # Use the changed binary commands (for search only)
                        # Support various datasets in the hnsw binary
                        # In README, add the command to build HNSW (like those for NSG)

                        # print(f"Config: dataset={dataset}"
                        #         f", max_degree={max_degree}"
                        #         f", ef={ef}"
                        #         f", k={k}"
                        #         f", omp={omp}"
                        #         f", interq_multithread={interq_multithread}"
                        #         f", batch_size={batch_size}")
                        
                        # cmd_perf_test = f"{hnsw_search_bin_path} " + \
                        #                 f"{max_degree} " + \
                        #                 f"{ef_construction} " + \
                        #                 f"{ef} " + \
                        #                 f"{k} " + \
                        #                 f"{omp} " + \
                        #                 f"{interq_multithread} " + \
                        #                 f"{batch_size} " + \
                        #                 f"{data_path}" + \
                        #                 f" > {log_perf_test}"
                        
                        # print(f"Running command:\n{cmd_perf_test}")
                        # # Pre-running to avoid cold start
                        # if not glob.glob("*.bin"):
                        #     os.system(cmd_perf_test)
                        # # Running the actual test
                        # os.system(cmd_perf_test)
                        
                        # node_count, time_batch, qps, recall = read_from_log(log_perf_test)
                        # write_to_json(perf_df_path, json_dict, 
                        #                 dataset, algo, str(max_degree), str(ef_construction), str(ef), str(k), str(omp), str(interq_multithread), str(batch_size),
                        #                 recall, node_count, time_batch, qps)
    
                                        
