import numpy as np

def load_data_fvecs(filename):
    with open(filename, "rb") as file:
        dim_bytes = file.read(4)  # Read 4 bytes for dimension
        dim = int.from_bytes(dim_bytes, byteorder='little')  # Convert bytes to integer for dimension

        file.seek(0, 2)  # Move the file pointer to the end
        fsize = file.tell()  # Get the file size
        num = fsize // ((dim + 1) * 4)  # Calculate the number of data points

        file.seek(0)  # Move the file pointer back to the beginning
        data = np.empty((num, dim), dtype=np.float32)  # Create an empty numpy array to store data

        for i in range(num):
            file.seek(4, 1)  # Move the file pointer forward by 4 bytes to skip index
            data[i] = np.fromfile(file, dtype=np.float32, count=dim)  # Read dim number of float values

    return data, num, dim


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Wenqi: Format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def mmap_bvecs_SBERT(fname, num_vec=int(1e6)):
    """
    SBERT, 384 dim, no header
    """
    d = 384
    x = np.memmap(fname, dtype='float32', mode='r')
    return x.reshape(-1, d)