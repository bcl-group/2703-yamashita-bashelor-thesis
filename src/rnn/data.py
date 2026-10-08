import numpy as np


def create_dataset(f, maxlen):

    length_of_sequences = len(f)

    x = []
    t = []

    for i in range(length_of_sequences - maxlen):
        x.append(f[i:i + maxlen])
        t.append(f[i + maxlen])

    x = np.array(x).reshape(-1, maxlen, 1)
    t = np.array(t).reshape(-1, 1)

    return x, t


def split_in_order(x, t, test_size=0.2):

    n_train = int(len(x) * (1.0 - test_size))
    return x[:n_train], x[n_train:], t[:n_train], t[n_train:]
