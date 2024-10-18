from math import ceil
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pyinform import transferentropy, mutualinfo




###################################################################################################
# Cross correlation
###################################################################################################

def get_activity_cross_correlation(activity, tau=1):
    neuron_num = activity.shape[0]
    cross_corr = np.zeros((neuron_num, neuron_num))

    for i in range(neuron_num):
        for j in range(neuron_num):
            postsynaptic = activity[i, tau:]
            if tau == 0:
                presynaptic = activity[j, :]
            else:
                presynaptic = activity[j, :-tau]
            # normalize
            # presynaptic = (presynaptic - np.mean(presynaptic)) / np.std(presynaptic)
            # postsynaptic = (postsynaptic - np.mean(postsynaptic)) / np.std(postsynaptic)
            cross_corr[i, j] = np.corrcoef(presynaptic, postsynaptic)[0, 1]
    return cross_corr



###################################################################################################
# Mutual Information
###################################################################################################

def get_activity_mutual_information(activity, tau=1, bins=10):
    neuron_num = activity.shape[0]
    mutual_information = np.zeros((neuron_num, neuron_num))

    bin_edges = np.linspace(np.min(activity), np.max(activity), bins + 1)

    for i in range(neuron_num):
        for j in range(i, neuron_num):
            if tau == 0:
                presynaptic = activity[j]
                postsynaptic = activity[i]
            else:
                presynaptic = activity[j, :-tau]
                postsynaptic = activity[i, tau:]

            digitized_presynaptic = np.digitize(presynaptic, bins=bin_edges)
            digitized_postsynaptic = np.digitize(postsynaptic, bins=bin_edges)

            # Calculate mutual information
            score = mutualinfo.mutual_info(digitized_presynaptic, digitized_postsynaptic)

            mutual_information[i, j] = score
            mutual_information[j, i] = score

    return mutual_information



###################################################################################################
# Transfer Entropy
###################################################################################################

def get_activity_transfer_entropy(activity, history=1, bins=10):
    """
    Compute the NxN connectivity matrix using transfer entropy.

    Parameters:
    activity (numpy.ndarray): NxT matrix of neural activity.
    history (int): The number of past time steps to consider for each neuron.
    future (int): The number of future time steps to predict.

    Returns:
    numpy.ndarray: NxN matrix of transfer entropy values.
    """
    N, T = activity.shape
    connectivity_matrix = np.zeros((N, N))

    bin_edges = np.linspace(np.min(activity), np.max(activity), bins + 1)

    # Loop over all pairs of neurons
    for source in range(N):
        for target in range(N):
            
            # Extract the time series for the source and target
            src_series = activity[source, :]
            tgt_series = activity[target, :]

            digitized_presynaptic = np.digitize(src_series, bins=bin_edges)
            digitized_postsynaptic = np.digitize(tgt_series, bins=bin_edges)

            # Calculate transfer entropy from source to target
            te = transferentropy.transfer_entropy(digitized_presynaptic, digitized_postsynaptic, k=history)

            # Store the transfer entropy in the matrix
            connectivity_matrix[target, source] = te

    return connectivity_matrix