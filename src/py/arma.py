import numpy as np
import scipy.sparse

def fromfile(path, dtype=np.float64):
    with open(path, 'rb') as f:
        line = f.readline()
        # TODO: this first line also contains information about the dtype
        if line[:4] != b'ARMA':
            raise Exception("file %s isn't in Armadillo binary format" % path)
        dims = tuple(map(int, f.readline().split()))
        return np.fromfile(f, dtype=dtype).reshape(*dims)

def loadcoo(path, dtype=np.float64, index_dtype=np.uint64):
    rows, cols, vals = [], [], []
    with open(path, 'r') as f:
        for line in f:
            row_str, col_str, val_str = line.split()
            rows.append(index_dtype(row_str))
            cols.append(index_dtype(col_str))
            vals.append(dtype(val_str))
    return scipy.sparse.coo_matrix((np.array(vals), (np.array(rows), np.array(cols))))
