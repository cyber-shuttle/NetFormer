import numpy as np
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy import stats, signal


def linear_transform(matrix, GT):
    flatten_matrix = matrix.flatten()
    flatten_GT = GT.flatten()

    # linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(flatten_matrix, flatten_GT)
    # positive transformation only
    if slope < 0:
        slope = -slope
        intercept = -intercept
    result = matrix * slope + intercept
    return result




############################################################################################################
# Connectivty-constrained simulation
############################################################################################################

def construct_weight_matrix_cell_type(neuron_num):
    """
    Construct weight matrix that simulates the real mouse data with cell type information

    
    Parameters
    ----------
    neuron_num: int
        Number of neurons in the network
    
        
    Returns
    -------
    weight_matrix: torch.Tensor
        Weight matrix used in the simulation

    cell_type_order: list
        Order of cell types in the weight matrix

    cell_type_ids: np.array
        Cell type ids that encode the cell type information for each neuron

    cell_type_count: dict
        Number of neurons in each cell type
    """

    cell_type_order = ['EC', 'Pvalb', 'Sst', 'Vip']
    # Let the first 76% neurons be EC, 8% neurons be Pvalb, 8% neurons be Sst, 8% neurons be Vip
    cell_type_ids = np.zeros(neuron_num, dtype=int)
    cell_type_ids[:int(neuron_num*0.76)] = 0
    cell_type_ids[int(neuron_num*0.76):int(neuron_num*0.84)] = 1
    cell_type_ids[int(neuron_num*0.84):int(neuron_num*0.92)] = 2
    cell_type_ids[int(neuron_num*0.92):] = 3
    
    cell_type_count = {'EC':int(neuron_num*0.76), 'Pvalb':int(neuron_num*0.08), 'Sst':int(neuron_num*0.08), 'Vip':int(neuron_num*0.08)}
    
    # construct cutoff matrix using probability matrix from mouse patch clamp data
    cutoff_matrix = np.zeros((4, 4))
    cutoff_matrix[0, 0] = 13/229
    cutoff_matrix[1, 0] = 22/53
    cutoff_matrix[2, 0] = 20/67
    cutoff_matrix[3, 0] = 11/68

    cutoff_matrix[0, 1] = 18/52
    cutoff_matrix[1, 1] = 45/114
    cutoff_matrix[2, 1] = 8/88
    cutoff_matrix[3, 1] = 0/54

    cutoff_matrix[0, 2] = 13/56
    cutoff_matrix[1, 2] = 15/84
    cutoff_matrix[2, 2] = 8/154
    cutoff_matrix[3, 2] = 25/84

    cutoff_matrix[0, 3] = 3/62
    cutoff_matrix[1, 3] = 1/54
    cutoff_matrix[2, 3] = 12/87
    cutoff_matrix[3, 3] = 2/209

    # construct strength matrix using post-synaptic potential matrix from mouse patch clamp data
    strength_matrix = np.zeros((4, 4))
    strength_matrix[0, 0] = 0.11
    strength_matrix[1, 0] = 0.27
    strength_matrix[2, 0] = 0.1
    strength_matrix[3, 0] = 0.45

    strength_matrix[0, 1] = -0.44
    strength_matrix[1, 1] = -0.47
    strength_matrix[2, 1] = -0.44
    strength_matrix[3, 1] = -0.23

    strength_matrix[0, 2] = -0.16
    strength_matrix[1, 2] = -0.18
    strength_matrix[2, 2] = -0.19
    strength_matrix[3, 2] = -0.17

    strength_matrix[0, 3] = -0.06
    strength_matrix[1, 3] = -0.10
    strength_matrix[2, 3] = -0.17
    strength_matrix[3, 3] = -0.10

    # uniformly initialize weight matrix with uniform distribution from 0 to 1
    weight_matrix = torch.rand(neuron_num, neuron_num)

    # set elements over cutoff to 0
    for i in range(neuron_num):
        for j in range(neuron_num):
            cell_type_i = cell_type_ids[i]
            cell_type_j = cell_type_ids[j]
            if weight_matrix[i, j] > cutoff_matrix[cell_type_i, cell_type_j]:
                weight_matrix[i, j] = 0
            else:
                mean = strength_matrix[cell_type_i, cell_type_j]
                std = 0.1
                weight_matrix[i, j] = torch.normal(mean, std, size=(1,))

    return weight_matrix, cell_type_order, cell_type_ids, cell_type_count



############################################################################################################
# Ring circuit simulation
############################################################################################################

def construct_weight_matrix_ring_circuit(N):
    # Weight parameters
    sig_1 = 6.98
    sig_2 = 7
    a1 = 1
    a2 = 1.0005

    # Create weight matrix W
    indices = np.arange(N)
    i_indices = indices.reshape(N, 1)
    j_indices = indices.reshape(1, N)
    x = np.minimum(np.abs(i_indices - j_indices), N - np.abs(i_indices - j_indices))
    W = a1 * (np.exp(-x**2 / (2 * sig_1**2)) - a2 * np.exp(-x**2 / (2 * sig_2**2)))

    return W



############################################################################################################
# Multimodal mouse data preprocessing with multiple sessions
############################################################################################################

def assign_unique_neuron_ids(
    all_sessions_original_UniqueID: list,
    num_neurons_per_session: list,
):
    """
    Assign unique neuron IDs to all neurons in all sessions, 
    since same neuron can be recorded in multiple sessions in a animal

    
    Parameters
    ----------
    all_sessions_original_UniqueID: list
        a concatenated list of the original UniqueID from all sessions

    num_neurons_per_session: list
        a list of the number of neurons in each session
        

    Returns
    -------
    all_sessions_new_UniqueID: list
        a list of sessions new UniqueID, each session is a 1D array of shape num_neurons
    """

    # first reassign ID starting from 0 to those non-NaN neurons
    # same IDs should be assigned to neurons that have the same original UniqueID
    non_nan_values = all_sessions_original_UniqueID[~np.isnan(all_sessions_original_UniqueID)]
    unique_non_nan_values = np.unique(non_nan_values)
    id_mapping = {unique_non_nan_values[i]: i for i in range(len(unique_non_nan_values))}

    new_ids = [id_mapping[non_nan_values[i]] for i in range(len(non_nan_values))]
    all_sessions_new_UniqueID = np.copy(all_sessions_original_UniqueID)
    all_sessions_new_UniqueID[~np.isnan(all_sessions_new_UniqueID)] = new_ids

    # assign new IDs to those NaN neurons
    num_unique_non_nan = unique_non_nan_values.shape[0]     # new IDs start from num_unqiue_non_nan
    num_nan = np.sum(np.isnan(all_sessions_original_UniqueID))           # new IDs end with num_non_nan + num_nan -1

    new_ids = np.arange(num_unique_non_nan, num_unique_non_nan + num_nan)
    all_sessions_new_UniqueID[np.isnan(all_sessions_new_UniqueID)] = new_ids

    # segment all_sessions_new_UniqueID into sessions
    all_sessions_new_UniqueID = np.split(all_sessions_new_UniqueID, np.cumsum(num_neurons_per_session)[:-1])

    num_unique_neurons = num_unique_non_nan + num_nan

    return all_sessions_new_UniqueID, num_unique_neurons    # all_sessions_new_UniqueID - a list of 1D array: (num_sessions, num_neurons_per_session)



def assign_unique_cell_type_ids(
    all_sessions_original_cell_type: list,
    num_neurons_per_session: list,
):
    """

    Parameters
    ----------
    all_sessions_original_cell_type: list
        a concatenated list of the original cell_type from all sessions
  
        
    Returns
    -------
    all_sessions_new_cell_type: list
        a list of sessions new cell_type, each session is a 1D array of shape num_neurons
    """

    unique_cell_types = list(set(all_sessions_original_cell_type))
    unique_cell_types.sort()
    
    # assign IDs to cell types
    cell_type_order = [unique_cell_types[i] for i in range(len(unique_cell_types))]
    print('cell_type_order:', cell_type_order) 

    # get new cell type IDs
    all_sessions_new_cell_type_id = np.zeros(len(all_sessions_original_cell_type)).astype(int)
    for i in range(len(all_sessions_original_cell_type)):
        all_sessions_new_cell_type_id[i] = cell_type_order.index(all_sessions_original_cell_type[i])

    # segment all_sessions_new_cell_type_id into sessions
    all_sessions_new_cell_type_id = np.split(all_sessions_new_cell_type_id, np.cumsum(num_neurons_per_session)[:-1])

    return all_sessions_new_cell_type_id, cell_type_order   # all_sessions_new_cell_type_id - a list of 1D array: (num_sessions, num_neurons_per_session)



def sliding_windows(
    all_sessions_acitvity: list,
    all_sessions_new_UniqueID: list,
    all_sessions_new_cell_type_id: list,
    all_sessions_state: list,
    window_size: int,
):
    """

    Parameters
    ----------
    (can be from TRAIN or VAL set)

    all_sessions_acitvity: list
        a list of sessions activity, each session is a 2D array of shape (num_neurons, num_frames)

    all_sessions_new_UniqueID: list
        a list of sessions new UniqueID, each session is a 1D array of shape num_neurons

    all_sessions_new_cell_type_id: list
        a list of sessions new cell type id, each session is a 1D array of shape num_neurons

    all_sessions_state: list
        a list of sessions state, each session is a 1D array of shape num_frames

    window_size: int
        size of sliding window

        
    Returns
    -------
    all_sessions_activity_windows: list
        a list of sessions activity windows, each session is a 3D array of shape (num_windows, num_neurons, window_size)

    all_sessions_new_UniqueID_windows: list
        a list of sessions new UniqueID windows, each session is a 2D array of shape (num_windows, num_neurons) (each row should be the same)

    all_sessions_new_cell_type_id_windows: list
        a list of sessions new cell type id windows, each session is a 2D array of shape (num_windows, num_neurons) (each row should be the same)
    """

    all_sessions_activity_windows = []
    all_sessions_new_UniqueID_windows = []
    all_sessions_new_cell_type_id_windows = []
    all_sessions_state_windows = []

    for i in range(len(all_sessions_acitvity)):
        num_neurons = all_sessions_acitvity[i].shape[0]
        num_frames = all_sessions_acitvity[i].shape[1]
        num_windows = num_frames - window_size + 1

        # activity
        activity_windows = np.zeros((num_windows, num_neurons, window_size))
        for j in range(num_windows):
            activity_windows[j] = all_sessions_acitvity[i][:, j:j+window_size]
        all_sessions_activity_windows.append(activity_windows)

        # UniqueID
        UniqueID_windows = np.zeros((num_windows, num_neurons))
        for j in range(num_windows):
            UniqueID_windows[j] = all_sessions_new_UniqueID[i]
        all_sessions_new_UniqueID_windows.append(UniqueID_windows)

        # cell type id
        cell_type_id_windows = np.zeros((num_windows, num_neurons))
        for j in range(num_windows):
            cell_type_id_windows[j] = all_sessions_new_cell_type_id[i]
        all_sessions_new_cell_type_id_windows.append(cell_type_id_windows)

        # state
        state_windows = np.zeros((num_windows, window_size))
        for j in range(num_windows):
            state_windows[j] = all_sessions_state[i][j:j+window_size]
        all_sessions_state_windows.append(state_windows)

    return all_sessions_activity_windows, all_sessions_new_UniqueID_windows, all_sessions_new_cell_type_id_windows, all_sessions_state_windows





############################################################################################################
# Tools for evaluation on connectivity matrix inferred from NetFormer
############################################################################################################

def NN2KK_remove_no_connection_sim(
    connectivity_matrix_new, 
    connectivity_matrix_GT, 
    cell_type_order, 
    cell_type_count
):
    connectivity_matrix_KK = np.zeros((len(cell_type_order), len(cell_type_order)))

    accumulated_num_cells_i = 0
    for i in range(len(cell_type_order)):
        old_i_start = accumulated_num_cells_i
        accumulated_num_cells_i += cell_type_count[cell_type_order[i]]

        accumulated_num_cells_j = 0
        for j in range(len(cell_type_order)):
            old_j_start = accumulated_num_cells_j
            accumulated_num_cells_j += cell_type_count[cell_type_order[j]]

            # count the number of non-zeros in the ground truth matrix
            mask_non_zeros = connectivity_matrix_GT[old_i_start : accumulated_num_cells_i, old_j_start : accumulated_num_cells_j] != 0
            total_num_non_zeros_elements = np.sum(mask_non_zeros)

            if total_num_non_zeros_elements == 0:
                connectivity_matrix_KK[i, j] = 0
            else:
                connectivity_matrix_KK[i, j] = np.sum(connectivity_matrix_new[old_i_start : accumulated_num_cells_i, old_j_start : accumulated_num_cells_j][mask_non_zeros]) / total_num_non_zeros_elements

    return connectivity_matrix_KK



def multisession_NN_to_KK(
    multisession_NN_list: list,
    cell_type_order: list,
    multisession_cell_type_id_list: list,
):
    """
    This function is used for mouse data to compute K*K connectivity strength matrix from multiple sessions N*N results.
    It computes one K*K matrix directly from N*N matrices from multiple sessions.


    Parameters
    ----------
    multisession_NN_list: list
        a list of N*N connectivity matrices from multiple sessions (N can be different in different sessions)
    
    cell_type_order: list
        This should contain all cell types that are in the data. (a union of all cell types from all sessions). 
        The order should be decided when processing the data. E.g. ['EC', 'Pvalb', 'Sst', 'Vip']

    multisession_cell_type_id_list: list
        a list of cell type ids from multiple sessions, each session is a 1D array of shape num_neurons

    
    Returns
    -------
    result: np.array
        K*K connectivity strength, with cell_type_order as the order of cell types
    """

    KK_result = np.zeros((len(cell_type_order), len(cell_type_order)))
    # count the number of sessions that have connection between cell type i and cell type j
    KK_count = np.zeros((len(cell_type_order), len(cell_type_order)))

    for i in range(len(multisession_NN_list)):
        current_session_NN = multisession_NN_list[i]
        current_session_cell_type_id = multisession_cell_type_id_list[i]

        for j in range(len(current_session_cell_type_id)):
            for k in range(len(current_session_cell_type_id)):

                # Ignore the diagonal elements
                if j == k:
                    continue

                cell_type_j = current_session_cell_type_id[j]
                cell_type_k = current_session_cell_type_id[k]

                KK_result[cell_type_j, cell_type_k] += current_session_NN[j, k]
                KK_count[cell_type_j, cell_type_k] += 1

    # make all 0 entries in KK_count to be 1, so that we don't divide by 0
    KK_count[KK_count == 0] = 1
    return KK_result / KK_count



def experiment_KK_to_eval_KK(
    experiment_KK: np.ndarray,
    experiment_cell_type_order: list,
    eval_cell_type_order: list,
):
    """
    This function is used to convert experiment_KK to eval_KK, since eval cell types are a subset of experiment cell types.
    The output KK will be a matrix with eval cell-type order.
    """

    eval_KK = np.zeros((len(eval_cell_type_order), len(eval_cell_type_order)))
    for eval_cell_type_i in eval_cell_type_order:

        if eval_cell_type_i in experiment_cell_type_order:
            experiment_index_i = experiment_cell_type_order.index(eval_cell_type_i)
            eval_index_i = eval_cell_type_order.index(eval_cell_type_i)

            for eval_cell_type_j in eval_cell_type_order:

                if eval_cell_type_j in experiment_cell_type_order:
                    experiment_index_j = experiment_cell_type_order.index(eval_cell_type_j)
                    eval_index_j = eval_cell_type_order.index(eval_cell_type_j)

                    eval_KK[eval_index_i, eval_index_j] = experiment_KK[experiment_index_i, experiment_index_j]

    return eval_KK