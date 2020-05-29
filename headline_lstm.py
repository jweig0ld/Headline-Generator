import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

""" 
TODO: Get the training data in a readable format. 
"""
training_data = ['H', 'e', 'l', 'l', 'o']
char_batch = training_data[:1000]  # Taking the characters for the entire dataset is excessive

# Encode/Decode characters
chars = list(set(char_batch))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Set hyperparameters
learning_rate = 1e-3
sequence_length = 25
hidden_size = 100


def one_hot_encode(arr, n_chars):
    """
    :param arr: Input characters
    :param n_labels: (int) Number of individual characters
    :return: Matrix of one-hot encodings (each row is an input character, each column is the encoding)
    """
    arr_length = len(arr)
    encoded = np.zeros((n_chars, arr_length))
    for i in range(arr_length):
        encoded[char_to_idx[arr[i]], i] = 1
    return encoded


def one_hot_decode(arr):
    """
    :param arr: Matrix of encoded characters
    :return: List of decoded characters
    """
    decoded = []
    for i in range(arr.shape[1]):
        idx = np.argmax(arr[:, i])
        decoded.append(idx_to_char[idx])
    return decoded


def get_batches(arr, batch_size, seq_length):
    return "You're gonna nail this"


class LSTM(nn.Module):
    def __init__(self, chars, n_hidden, n_layers, dropout_probability, learning_rate):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_probability = dropout_probability
        self.leaning_rate = learning_rate
        self.chars = chars

        self.lstm = nn.LSTM(input_size=len(self.chars), hidden_size=n_hidden, num_layers=n_layers, bias=True,
                            batch_first=True, dropout=dropout_probability)
        self.dropout = nn.Dropout(dropout_probability)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, h):
        r_out, h = self.lstm(x, h)
        out = self.dropout(r_out)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, h

    def init_h(self, batch_size):
        weight = next(self.parameters())
        h = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
             weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return h

def train(model, data, epochs, batch_size, lr, clip, val_frac, seq_length, print_every):
    """
    :param model: LSTM Network
    :param data: Training data (encoded vectors)
    :param epochs: Number of epochs (int)
    :param batch_size: The number of sequences per batch
    :param lr: Learning Rate
    :param clip: Gradient clipping
    :param val_frac: The proportion of the dataset that we want for validation
    :param seq_length: Length of the sequence that we want to process at once
    :param print_every: Number of iterations after which we want to print
    """
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr)
    loss_type = nn.CrossEntropyLoss()

    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    """ This is where I left off - still need to deal with the batches and the input data """