import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

""" 
TODO: Get the training data in a readable format. 
"""
training_data = ['H','e','l', 'l', 'o']
char_batch = training_data[:1000]  # Taking the characters for the entire dataset is excessive

# Encode/Decode characters
chars = list(set(char_batch))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Set hyperparameters
learning_rate = 1e-3
sequence_length = 25
hidden_size = 100


def one_hot_encode(arr, n_labels):
    """
    :param arr: Input characters
    :param n_labels: (int) Number of individual characters
    :return: (List) List of vectors with one-hot encoding for each of the chars in arr
    """
    encoded = []
    for char in arr:
        vector = np.zeros(n_labels)
        vector[char_to_idx[char]] = 1
        encoded.append(vector)

    return encoded


def get_batches(arr, batch_size, seq_length):
    """
    :param arr: encoded input characters
    :param batch_size: length of the batch requested
    :param seq_length: hyperparameter
    :return: matrix of char encodings and corresponding array of ground truth targets
    """
    training_seq_length = arr.shape[1]
    number_of_batches = training_seq_length // (batch_size * seq_length)
    arr = arr[(batch_size * seq_length) * number_of_batches]  # Only keep enough to make full batches
    for i in range(0, number_of_batches, seq_length):
        chars = arr[i:i + seq_length]
        targets = np.zeros_like(chars)
        yield chars, targets
