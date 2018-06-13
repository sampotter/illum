import h5py
import scipy.sparse

def dok_from_coo_file(path, M, N, dtype, sep=None):
    dok = scipy.sparse.dok_matrix((M, N), dtype)
    def parse_line(line):
        return tuple(map(int, line.split(sep)))
    with open(path) as f:
        for line in f:
            i, j, v = parse_line(line)
            dok[i, j] = v
    return dok
        
def csc_from_h5_file(path, M, N, dtype):
    with h5py.File(path, 'r') as f:
        data = np.array(f['values'][:]).flatten()
        indices = np.array(f['rowind'][:]).flatten()
        indptr = np.array(f['indptr'][:]).flatten()
    return scipy.sparse.csc_matrix(
        (data, indices, indptr), shape=(M, N), dtype=dtype)
