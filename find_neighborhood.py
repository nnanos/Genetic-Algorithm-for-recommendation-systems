import numpy as np
import numpy.matlib
import pandas as pd
import os

def get_neighborhood(user_movie_matrix,user_under_consideration):


    #finding the neighbor of that user------------------------------

    #centering the user movie matrix -------------------------------
    b = np.mean(user_movie_matrix,axis = 1)
    mean_matrix = np.tile(b.T,(1682,1)).T

    mask = np.zeros(np.shape(user_movie_matrix))
    mask[np.nonzero(user_movie_matrix)] = 1
    mean_matrix = np.multiply(mean_matrix,mask)
    user_movie_centered = user_movie_matrix - mean_matrix
    #-----------------------------------------------------------------

    #normalizing the user_movie_centered matrix-----------------------
    user_movie_centered_normalized = np.zeros(np.shape(user_movie_matrix))
    for i in range(np.shape(user_movie_matrix)[0]):
        norm2 = np.sqrt(np.dot(user_movie_centered[i, :], user_movie_centered[i, :]))
        user_movie_centered_normalized[i,:] = user_movie_centered[i,:]/norm2
    #-----------------------------------------------------------------

    #obtaining all the pearson correlations by projecting the user_under_consideration to the
    #space defined by the user_movie_centered_normalized
    correlations = user_movie_centered_normalized.dot(user_movie_centered_normalized[user_under_consideration,:])

    I = np.argsort(correlations)
    top_10_similar_user_ind = I[(len(I)-11):(len(I)-1)]
    top_10_similar_user_ind = np.flip(top_10_similar_user_ind)

    neighborhood = user_movie_matrix[top_10_similar_user_ind,:]
    neighborhood_centered_normalized = user_movie_centered_normalized[top_10_similar_user_ind,:]

    return neighborhood_centered_normalized