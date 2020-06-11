import numpy as np


def repair(population,user_under_consideration):

    for i in range(len(population)-1):

        for j in range(len(population[0])-1):
            if user_under_consideration[j] > 0:
                population[i][j] = int(user_under_consideration[j])


    return population