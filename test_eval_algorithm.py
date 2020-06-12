import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

def testing(best_mean_individuals,user_under_consideration_index_number) :
    pwd = os.getcwd() + '\\' + 'ua.test'
    data = pd.read_csv( pwd , sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None, encoding='utf-8')
    data = data.drop('timestamp', axis=1)
    # obtaning user-movie matrix from the movielens data
    matrix = data.pivot(index='user', columns='item', values='rating')

    new_index_column = np.arange(1682) + 1
    matrix = matrix.reindex(new_index_column, axis=1)

    matrix = matrix.fillna(0)
    test_matrix = matrix.to_numpy()

    user_under_cons = test_matrix[user_under_consideration_index_number]
    nonz_ind = np.nonzero(user_under_cons)
    the_10_ratings = user_under_cons[nonz_ind]

    #obtaining the RMSE and MAE for each gen
    #these two metrics are calculated between the 10 ratings(true data) and the corresponding 10 ratings of the best inds per gen (predicted)
    mae_per_gen = []
    rmse_per_gen = []
    for i in range(np.shape(best_mean_individuals)[0]):
        pred = best_mean_individuals[i,:]
        rmse_per_gen.append(np.sqrt(mean_squared_error( the_10_ratings , pred[nonz_ind] )))
        mae_per_gen.append(mean_absolute_error( the_10_ratings , pred[nonz_ind] ))


    #plotting the rmse,mae per gen
    plt.figure()
    plt.plot(np.arange(150 + 1), rmse_per_gen)
    plt.xlabel('generation')
    plt.ylabel('RMSE')
    plt.show()

    plt.figure()
    plt.plot(np.arange(150 + 1), mae_per_gen)
    plt.xlabel('generation')
    plt.ylabel('MAE')
    plt.show()




