import numpy as np

def best_individuals_mean_performance(c):
    a = []
    for i in c :
        a.append(np.max(i))

    return np.mean(a)


def mean_number_of_gens(c):
    a = []
    for i in c:
        a.append(len(i))

    return np.floor(np.mean(a))

