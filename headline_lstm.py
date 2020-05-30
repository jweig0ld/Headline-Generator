import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


with open('tales.txt', 'r') as f:
    text = f.read()

text[:100]

chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

encoded = np.array([char2int[ch] for ch in text])
encoded[:100]

# Set hyperparameters
lr = 1e-3
n_hidden = 512
n_layers = 3
batch_size = 128
seq_length = 100
n_epochs = 20


def one_hot_encode(arr, n_labels):
    """
    :param arr: Input characters
    :param n_labels: (int) Number of individual characters
    :return: Matrix of one-hot encodings (each row is an input character, each column is the encoding)
    """
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


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


batches = get_batches(encoded, 8, 50)
x, y = next(batches)


class LSTM(nn.Module):
    def __init__(self, chars, n_hidden, n_layers, dropout_probability, learning_rate):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_probability = dropout_probability
        self.leaning_rate = learning_rate
        self.chars = chars
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

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
            x = one_hot_encode(x, n_chars)
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

train(model, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=lr, print_every=10)

model_name = 'rnn_20_epoch.net'

checkpoint = {'n_hidden': model.n_hidden,
              'n_layers': model.n_layers,
              'state_dict': model.state_dict(),
              'tokens': model.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)


def predict(model, char, h=None, top_k=None):
    """ Given a character, return the next character """

    x = np.array([[model.char2int[char]]])
    x = one_hot_encode(x, len(model.chars))
    inputs = torch.from_numpy(x)

    h = tuple([each.data for each in h])
    out, h = model(inputs, h)

    p = F.softmax(out, dim=1).data

    if top_k is None:
        top_ch = np.arange(len(model.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    """ Select the next likely character with some element of randomness """
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    return model.int2char[char], h


def sample(model, size, prime='Once upon a time', top_k=None):
    """ Change everything back to eval mode """
    model.eval()

    chars = [ch for ch in prime]
    h = model.init_h(1)
    for ch in prime:
        char, h = predict(model, ch, h, top_k=top_k)

    chars.append(char)

    """ Pass in previous character to get a new one """
    for ii in range(size):
        char, h = predict(model, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


print(sample(model, 2000, prime='Once upon a time', top_k=5))

""" Here we have loaded in a model that has trained over 20 epochs 'rnn_20_epoch.net' """
with open('rnn_20_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)

loaded = LSTM(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

""" Sample using a loaded model """
print(sample(loaded, 2000, top_k=5, prime="The beautiful"))

