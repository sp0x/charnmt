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
            src_emb, 
            hid_dim, 
            vocab_size, 
            dropout, 
            s, 
            n_highway, 
            n_rnn_layers):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.s = s

        # Embedding
        self.embedding = nn.Embedding(vocab_size, src_emb)

        # half convolution with 8-grams
        self.n_filters = [200, 200, 250, 250, 300, 300, 300, 300]
        for i in range(len(self.n_filters)):
            conv2d = nn.Conv2d(1, self.n_filters[i], (i+1, src_emb), padding=(i, 0))
            setattr(self, "conv_layer{}".format(i+1), conv2d)

        # number of layers in highway network
        self.n_highway = n_highway
        highway_dim = sum(self.n_filters)
        for i in range(n_highway):
            setattr(self, "highway_gate{}".format(i+1), nn.Linear(highway_dim, highway_dim))
            setattr(self, "highway_layer{}".format(i+1), nn.Linear(highway_dim, highway_dim))

        # recurrent layer
        self.n_rnn_layers = n_rnn_layers
        self.rnn_encoder = nn.GRU(highway_dim, hid_dim, self.n_rnn_layers, 
                batch_first=True, dropout=dropout, bidirectional=True)

        self._init_weights()


    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)


    def init_gru(self, batch_size):
        return Variable(torch.zeros(self.n_rnn_layers*2, batch_size, self.hid_dim))


    def forward(self, x, h):
        # Embedding
        x = self.embedding(x.long())

        # Convoluation
        x = torch.unsqueeze(x, 1)
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
            x: tensor, with dimension (batch_size, seq_len, src_emb)

        @return
            Y: tensor, with dimension (batch_size, seq_len, N), where 
               N = \sum self.n_filters as feature
        ----------
        """
        batch_size = x.size(0)
        seq_len = x.size(2)
        Y = []
        for i in range(len(self.n_filters)):
            y = getattr(self, "conv_layer{}".format(i+1))(x)
            y = F.relu(y[:,:,:seq_len,:])
            y = y.view(batch_size, self.n_filters[i], seq_len)
            Y.append(y)

        Y = torch.cat(Y, 1)
        return Y.transpose(1, 2)


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
            relu = F.relu(getattr(self, "highway_layer{}".format(i+1))(y))
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
        output, h = self.rnn_encoder(x, h)
        return output


class Decoder(nn.Module):
    """
    Decoder of the character-level NMT model, with attention mechanism. Predict 
    next character given previous hidden state of decoder, last decoded character 
    and context information, denoted as s_{t-1}, y_{t-1}, z_i, respectively.
    """

    def __init__(self, 
            tar_emb, 
            hid_dim, 
            vocab_size, 
            context_dim, 
            dropout, 
            n_rnn_layers, 
            decoder_layers):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, tar_emb)
        
        # single-layer attention score network
        attention_feature_dim = hid_dim * 2 + tar_emb + context_dim
        self._align = nn.Linear(attention_feature_dim, 1)

        # decoder network from attention
        self.decode_layers = [context_dim] + decoder_layers + [vocab_size]
        for i in range(len(self.decode_layers)-1):
            fc = nn.Linear(self.decode_layers[i], self.decode_layers[i+1])
            setattr(self, "decoder_layer{}".format(i+1), fc)

        # rnn decoder
        self.n_rnn_layers = n_rnn_layers
        input_dim = tar_emb + context_dim
        self.rnn_decoder = nn.GRU(input_dim, hid_dim, self.n_rnn_layers, 
                batch_first=True, dropout=dropout, bidirectional=True)

        self._init_weights()


    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)


    def init_gru(self, batch_size):
        return Variable(torch.zeros(self.n_rnn_layers*2, batch_size, self.hid_dim))


    def forward(self, prev_s, prev_y, z):
        """
        Decoding with attention, and update decoding RNN memory

        @params
            prev_s: tensor, hidden state at time t-1 in the decoding procedure, 
                    with dimention (n_layer*n_direction, batch_size, hid_dim)
            prev_y: list of int, indices of previous characters at time t-1
            z: tensor, context-dependent vectors, with dimension (batch_size, 
               seq_len, feature)

        @return
            next_char: tensor, a distribution of each character
            h: updated memory state of RNN decoder
        """
        y = self.embedding(prev_y.long())

        # attention mechanism
        batch_size = y.size(0)
        prev_h = prev_s.transpose(0,1).contiguous().view(batch_size, -1)
        context = self._attention(prev_h, y, z)

        # update memory cell
        cat_input = torch.cat((y, context), 1).unsqueeze(1)
        output, h = self.rnn_decoder(cat_input, prev_s)

        # decode
        next_char = self._decode(context)

        return next_char, h


    def _attention(self, prev_s, prev_y, z):
        """
        Attention score function over each z_i w.r.t. previous hidden state and 
        previous generated token (character in this model) in the decoding 
        procedure. It computes a weighted average of the source sequence (z).

        ----------
        @params
            prev_s: tensor, hidden state at time t-1 in the decoding procedure, 
                    with dimention (batch_size, hid_dim)
            prev_y: tensor, character embeddings in decoder, with dimension 
                    (batch_size, tar_emb)
            z     : tensor, input sequence with dimention (batch_size, seq_len, 
                    feature)

        @return
            c: tensor, weighted average of all input tokens, with dimension 
               (batch_size, context_dim)
        ----------
        """
        batch_size, seq_len, dim = z.size()
        c = 0
        betas = []
        for i in range(seq_len):
            z_i = z[:,i,:]
            x = torch.cat((prev_s, prev_y, z_i), 1)
            beta = F.tanh(self._align(x))
            betas.append(beta)

        betas = torch.stack(betas, 1)
        alpha = F.softmax(betas)
        c = alpha * z

        return c.sum(1)


    def _decode(self, c):
        x = c
        for i in range(len(self.decode_layers)-1):
            x = getattr(self, "decoder_layer{}".format(i+1))(x)
        return x


class CharNMT(nn.Module):
    """
    char2char neural machine translation

    ----------
    @params
        vocab_size: int, number of unique characters
        max_len: int max. sequence length
        src_emb: int, embedding dimension of character in source language
        tar_emb: int, embedding dimension of character in target language
        hid_dim: int, dimension of hidden state in RNN
        dropout: float, dropout used in NMT neural network
        s: int, stride used in convolutional neural network
        n_highway: int, number of layers in highway network
        n_rnn_encoder_layers: int, number of stacked layers in RNN encoder
        n_rnn_decoder_layers: int, number of stacked layers in RNN decoder
        decoder_layers: list of int, number of neurons in each decoding layer
    ----------
    """

    def __init__(self, 
            vocab_size, 
            max_len=450,
            src_emb=128, 
            tar_emb=512,
            hid_dim=300, 
            dropout=0.5, 
            s=5, 
            n_highway=4, 
            n_rnn_encoder_layers=1, 
            n_rnn_decoder_layers=1, 
            decoder_layers=[1024]):
        super(CharNMT, self).__init__()
        self.name = "CharNMT"

        self.encoder = Encoder(src_emb, hid_dim, vocab_size, dropout, 
                s, n_highway, n_rnn_encoder_layers)

        self.decoder = Decoder(tar_emb, hid_dim, vocab_size, hid_dim * 2, 
                dropout, n_rnn_decoder_layers, decoder_layers)


    def init_hidden(self, batch_size):
        enc_h = self.encoder.init_gru(batch_size)
        dec_h = self.decoder.init_gru(batch_size)
        return enc_h, dec_h


    def compute_context(self, x, enc_h):
        return self.encoder(x, enc_h)


    def forward(self, x, c, dec_h):
        '''
        encode, attention, decode
        ----------
        @params
            x    :  previously generated characters
            c    :  tensor, context-dependent vectors, with dimension 
                    (batch_size, seq_len, feature)
            dec_h:  tensor, hidden state at time t-1 in the decoding procedure, 
                    with dimention (batch_size, hid_dim)
            
        @return
            char at time t and hidden state in decoder at time t
        ----------
        '''
        next_char, dec_h = self.decoder(dec_h, x, c)
        return next_char, dec_h


    def zero_grads(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
