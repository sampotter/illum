import scipy.sparse

def dok_from_coo_file(path, M, N, dtype):
    dok = scipy.sparse.dok_matrix((M, N), dtype)
    def parse_line(line):
        return tuple(map(int, line.split()))
    with open(path) as f:
        for line in f:
            i, j, v = parse_line(line)
            dok[i, j] = v
    return dok
        
