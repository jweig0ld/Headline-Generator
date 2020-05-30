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
n_hidden = 512
n_layers = 3
batch_size = 128
seq_length = 100
n_epochs = 20


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
    """
    :param arr: The encoded array that we want to make batches from
    :param batch_size: The number of sequences that we want in a batch
    :param seq_length: The length of each sequence that we want
    :return: Yielding batches
    """

    total_size = batch_size * seq_length
    """ Total number of batches that we can make: """
    n_batches = len(arr) // total_size

    """ We only want to keep enough characters to make full batches """
    arr = arr[:n_batches * total_size]
    """ Reshape it into rows"""
    arr = arr.reshape((batch_size, -1))

    for i in range(0, arr.shape[1], seq_length):
        x = arr[:, i:i + seq_length]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, i + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


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
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    counter = 0
    n_chars = len(model.chars)

    for epoch in range(epochs):

        h = model.init_h(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            """ One hot encode the input and make them torch tensors """
            x = one_hot_decode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            """ Need to create new variables for the hidden state, 
            otherwise we backprop through the entire training history """
            h = tuple([each.data for each in h])

            model.zero_grad()

            output, h = model(inputs, h)

            loss = criterion(output, targets.view(batch_size * seq_length))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()

            """ Loss statistics """
            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    """ One hot encode the data and make them Tensors """
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    """ Creating new variables for the hidden state, otherwise
                    we would backpropagate through the entire training history """
                    val_h = tuple([each.data for each in val_h])

                    outputs, val_h = model(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length))
                    val_losses.append(val_loss.item())

                """ Reset to train mode after iterating through the validation data """
                model.train()

                print("Epoch: {}/{}...".format(epoch + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


model = LSTM(chars, n_hidden, n_layers)
print(model)