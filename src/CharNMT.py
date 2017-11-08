import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder of the character-level NMT model. First embed each character into 
    d_c-dimensional vector, then use 8 single convolutional layer and max 
    pooling with stride to capture localities of unigrams up to 8-grams. Then 
    we get segments with width stride and feed them into a 4-layer highway 
    network. Finally, use single-layer bi-GRU to get the representation of 
    the whole sentence sequence.
    """

    def __init__(self, 
            input_dim, 
            hid_dim, 
            vocab_size, 
            dropout, 
            s, 
            n_highway, 
            n_rnn_layers):
        super(CharNMT, self).__init__()
        self.s = s

        # Embedding
        self.embedding = nn.Embedding(vocab_size, input_dim)

        # half convolution with 8-grams
        self.n_filters = [10] * 8
        for i in range(len(n_filters)):
            conv2d = nn.Conv2d(1, n_filters[i], (i+1, input_dim), padding=(i, 0))
            setattr(self, "conv_layer{}".format(i+1), conv2d)

        # number of layers in highway network
        self.n_highway = n_highway
        highway_dim = input_len / s
        for i in range(n_highway):
            setattr(self, "highway_gate{}".format(i+1), nn.Linear(highway_dim, highway_dim))
            setattr(self, "highway_layer{}".format(i+1), nn.Linear(highway_dim, highway_dim))

        # recurrent layer
        self.n_rnn_layers = n_rnn_layers
        self.rnn_encoder = nn.GRU(input_dim, hid_dim, self.n_rnn_layers, batch_first=True, 
                dropout=dropout, bidirectional=True)

        self._init_weights()


    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)


    def init_gru(self, batch_size):
        return Variable(torch.zeros(self.n_rnn_layers*2, batch_size, self.hid_dim))


    def forward(self, x, h):
        # Embedding
        x = self.embedding(x)

        # Convoluation
        x = self._convolution(x)

        # Max pooling with stride
        x = self._max_pooling(x)

        # Highway network
        x = self._highway_net(x)

        # Recurrent layer
        x = self._recurrent(x, h)

        return x


    def _convolution(self, x):
        """
        Single-layer convolution over kernel with width 1 ... 8 + ReLU

        ----------
        @params
            x: tensor, with dimension (batch_size, seq_len, input_dim)

        @return
            Y: tensor, with dimension (batch_size, seq_len, N), where 
               N = \sum self.n_filters as feature
        ----------
        """
        batch_size = x.size(0)
        seq_len = x.size(2)
        Y = []
        for i in range(self.n_filters):
            y = getattr(self."conv_layer{}".format(i+1))(x)
            y = F.relu(y[:,:,:seq_len])
            Y.append(y.view(batch_size, self.n_filters[i], seq_len))

        Y = torch.cat(Y, 1)
        return Y


    def _max_pooling(self, x):
        """
        Max pooling with stride self.s

        ----------
        @params
            x: tensor, with dimension (batch_size, seq_len, feature)

        @return
            y: tensor, with dimension (batch_size, seq_len/stride, feature)
        ----------
        """
        return F.max_pool2d(x, (self.s, 1))

    
    def _highway_net(self, x):
        """
        multi-layer highway network

        ----------
        @params
            x: tensor, with dimension (batch_size, seq_len, feature)

        @return
            y: tensor, same dimension as input x
        ----------
        """
        y = x
        for i in range(self.n_highway):
            g = F.sigmoid(getattr(self, "highway_gate{}".format(i+1))(y))
            relu = F.relu(getattr(self, "highway_layer{}".format(i+1)(y)))
            y = g * relu + (1-g) * y

        return y


    def _recurrent(self, x, h):
        """
        Bi-RNN, using GRU in this setting

        ----------
        @params
            x: tensor, input of sequences, with dimension (batch_size, seq_len, feature)

        @return
            y: tensor, output of RNN hidden state with two directions concatenated, 
               with dimension (batch_size, seq_len, feature)
        ----------
        """
        output, h = self.rnn(x, h)
        return output


class Decoder(nn.Module):
    """
    Decoder of the character-level NMT model, with attention mechanism. Predict 
    next character given previous hidden state of decoder, last decoded character 
    and context information, denoted as s_{t-1}, y_{t-1}, c_t, respectively.
    """

    def __init__(self):
        super(Decoder, self).__init__()


    def forward(self):
        #TODO
        pass


class CharNMT(nn.Module):
    """
    char2char neural machine translation

    ----------
    @params
        vocab_size: int, number of unique characters
        input_dim: int, dimension of each input character representation
        hid_dim: int, dimension of hidden state in RNN
        dropout: float, dropout used in NMT neural network
        s: int, stride used in convolutional neural network
        n_highway: int, number of layers in highway network
        n_rnn_encoder_layers: int, number of stacked layers in RNN encoder
    ----------
    """

    def __init__(self, 
            vocab_size, 
            input_dim=300, 
            hid_dim=300, 
            dropout=0.5, 
            s=5, 
            n_highway=4, 
            n_rnn_encoder_layers=1):
        super(CharNMT, self).__init__()
        self.encoder = Encoder(input_dim, hid_dim, vocab_size, dropout, 
                s, n_highway, n_rnn_encoder_layers)
        self.decoder = Decoder()


    def forward(self, x):
        c = self.encoder(x)
