"""
Convert saved NSG index to FPGA format

Example Usage:
python nsg_to_FPGA.py \
    --data_path /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_base.fvecs \
    --nsg_path ./sift1m.nsg \
    --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/SIFT1M_M64

"""


import numpy as np
import argparse
import os
import struct

from utils import load_data

class NSGConverter():
    
    def __init__(self, dimension: int, num_nodes: int):
        self.final_graph = []
        self.dim = dimension
        self.width = 0 # M in HNSW, max edges
        self.ep_ = 0
        self.cur_element_count = num_nodes
        self.db_vec = None
    
    def load_index(self, filename: str):
        print("Loading NSG index")
        with open(filename, "rb") as f:
            self.width = int.from_bytes(f.read(4), byteorder='little', signed=False)
            self.ep_ = int.from_bytes(f.read(4), byteorder='little', signed=False)
            cc = 0
            while True:
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k = int.from_bytes(k_bytes, byteorder='little', signed=False)
                cc += k
                tmp = list(np.frombuffer(f.read(k * 4), dtype=np.uint32))
                self.final_graph.append(tmp)
            cc //= self.cur_element_count
            # print(cc)
        assert self.cur_element_count == len(self.final_graph)

    def load_vectors(self, db_vec):
        """
        db_vec: numpy array of shape (n, dimension)
        """
        self.db_vec = db_vec
        assert self.cur_element_count == self.db_vec.shape[0], "The number of vertices in the index and data do not match"
    
    def convert_to_FPGA_format(self, out_dir, num_channels=[1]):
        """
        Save the data as FPGA format, must make sure `load_index` and `load_vectors` are already invoked.
        """
        print("Converting NSG index to FPGA format")
        assert self.db_vec is not None, "Please load the data first"
        assert len(self.final_graph) > 0, "Please load the index first"
        assert len(self.final_graph) == self.db_vec.shape[0], "The number of vertices in the index and data do not match"

        # save meta data
        byte_array_meta = bytearray() 
        byte_array_meta += self.cur_element_count.to_bytes(4, byteorder='little', signed=False)
        byte_array_meta += self.ep_.to_bytes(4, byteorder='little', signed=False) # entry point ID
        byte_array_meta += self.width.to_bytes(4, byteorder='little', signed=False) # max edges

        # save to files
        with open(os.path.join(out_dir, 'meta.bin'), 'wb') as f:
            f.write(byte_array_meta)
        
        # save ground level links
        # format of each node:
        #   [64 B header = num_links (4B int) + 60 byte paddings] + N [64B actual links] + paddings (to 64 B)
        if self.width % 16 == 0:
            num_bytes_per_ground_links = int(self.width / 16) * 64 + 64
        else:
            num_bytes_per_ground_links = (int(self.width / 16) + 1) * 64 + 64
        byte_array_ground_links = bytearray()
        for i in range(self.cur_element_count):
            link_count = len(self.final_graph[i])
            links = self.final_graph[i]
            byte_array_ground_links += int(link_count).to_bytes(4, byteorder='little', signed=False)
            byte_array_ground_links += b'\x00' * 60
            for j in range(self.width):
                if j < link_count:
                    byte_array_ground_links += int(links[j]).to_bytes(4, byteorder='little', signed=False)
                else:
                    byte_array_ground_links += b'\x00' * 4
            if len(byte_array_ground_links) % 64 != 0:
                byte_array_ground_links += b'\x00' * (64 - len(byte_array_ground_links) % 64)
        assert len(byte_array_ground_links) == self.cur_element_count * num_bytes_per_ground_links
            
        # with open(os.path.join(out_dir, 'ground_links.bin'), 'wb') as f:
        #     f.write(byte_array_ground_links)

        # save ground level vectors
        # format of each node:
        # [vector (4B float) + padding] + [visited (4B int, init as -1) + padding]
        if self.dim % 16 == 0:
            num_bytes_per_vector_with_padding = int(self.dim / 16) * 64 + 64
        else:
            num_bytes_per_vector_with_padding = (int(self.dim / 16) + 1) * 64 + 64
        byte_array_ground_vectors = bytearray()
        for i in range(self.cur_element_count):
            vector = self.db_vec[i]
            for j in range(self.dim):
                byte_array_ground_vectors += struct.pack('f', vector[j])
            if len(byte_array_ground_vectors) % 64 != 0:
                byte_array_ground_vectors += b'\x00' * (64 - len(byte_array_ground_vectors) % 64)
            byte_array_ground_vectors += b'\xff\xff\xff\xff' # -1 in int32
            byte_array_ground_vectors += b'\x00' * 60
        assert len(byte_array_ground_vectors) == self.cur_element_count * num_bytes_per_vector_with_padding

        # with open(os.path.join(out_dir, 'ground_vectors.bin'), 'wb') as f:
        #     f.write(byte_array_ground_vectors)
        
        # partition to N channels in round robin fashion
        for nc in num_channels:
            byte_array_ground_vectors_per_channel = [bytearray() for _ in range(nc)]
            byte_array_ground_links_per_channel = [bytearray() for _ in range(nc)]
            for i in range(self.cur_element_count):
                byte_array_ground_vectors_per_channel[i % nc] += byte_array_ground_vectors[i * num_bytes_per_vector_with_padding: (i + 1) * num_bytes_per_vector_with_padding]
                byte_array_ground_links_per_channel[i % nc] += byte_array_ground_links[i * num_bytes_per_ground_links: (i + 1) * num_bytes_per_ground_links]
            for i in range(nc):
                with open(os.path.join(out_dir, 'ground_vectors_{}_chan_{}.bin'.format(nc, i)), 'wb') as f:
                    f.write(byte_array_ground_vectors_per_channel[i])
                with open(os.path.join(out_dir, 'ground_links_{}_chan_{}.bin'.format(nc, i)), 'wb') as f:
                    f.write(byte_array_ground_links_per_channel[i])

        return byte_array_meta, byte_array_ground_links, byte_array_ground_vectors

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_base.fvecs", help='Path to the base data file in fvecs format')
    parser.add_argument('--nsg_path', type=str, default="./sift1m.nsg", help='Path to the NSG index file')
    parser.add_argument('--FPGA_index_path', type=str, default="/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/SIFT1M_M64", help='Path to the NSG index file')

    args = parser.parse_args()

    assert os.path.exists(args.data_path), "Data file does not exist"
    assert os.path.exists(args.nsg_path), "NSG index file does not exist"
    assert os.path.exists(args.FPGA_index_path), "FPGA index directory does not exist"

    # load data
    print("Start loading data")
    data_load, points_num, dim = load_data(args.data_path)
    print("data_load: ", np.shape(data_load))

    # load NSG index
    nsg_converter = NSGConverter(dim, points_num)

    nsg_converter.load_index(args.nsg_path)
    nsg_converter.load_vectors(data_load)

    nsg_converter.convert_to_FPGA_format(args.FPGA_index_path, num_channels=[1, 2, 4])
