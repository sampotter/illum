import numpy as np

def readobj(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    rows = [line.split() for line in lines]

    vrows = [r[1:] for r in rows if r[0] == 'v']
    v = np.array([[float(x) for x in r] for r in vrows])

    frows = [r[1:] for r in rows if r[0] == 'f']
    f = np.array([[int(i) - 1 for i in r] for r in frows], dtype=np.uint16)
    
    va = np.array([float(r[1]) for r in rows if r[0] == 'va'])

    vnrows = [r[1:] for r in rows if r[0] == 'vn']
    vn = np.array([[float(x) for x in r] for r in vnrows])

    return v, f, vn, va
