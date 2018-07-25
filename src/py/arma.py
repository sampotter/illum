import numpy as np

def fromfile(path, dtype=np.float64):
    with open(path, 'rb') as f:
        line = f.readline()
        # TODO: this first line also contains information about the dtype
        if line[:4] != b'ARMA':
            raise Exception("file %s isn't in Armadillo binary format" % path)
        dims = tuple(map(int, f.readline().split()))
        return np.fromfile(f, dtype=dtype).reshape(*dims)
