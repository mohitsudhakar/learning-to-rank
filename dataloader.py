import numpy as np

def sample_data():
    # generate query data
    x = np.array([[0.2, 0.3, 0.4],
                  [0.1, 0.7, 0.4],
                  [0.3, 0.4, 0.1],
                  [0.8, 0.4, 0.3],
                  [0.9, 0.35, 0.25]])
    y = np.array([0, 1, 0, 0, 2])
    q = np.array([1, 1, 1, 2, 2])
    return x, y, q

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        x, y, q = [], [], []
        info = []
        for line in lines:
            # print(line)
            feat, comment = line.split(' #')
            feat = feat.split()
            yi = feat[0]
            qi = feat[1].split(':')[-1]
            xi = feat[2:]
            xi = [f.split(':')[-1] for f in xi]
            # print(yi, qi, xi, comment)
            x.append(xi)
            y.append(yi)
            q.append(qi)
            info.append(comment)
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.int), np.array(q, dtype=np.int)
