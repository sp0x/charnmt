import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import os
import copy

import config
import utils
from WordAttnNMT import Decoder, train, evaluate
import eval


class ResNet(nn.Module):

    def __init__(self, in_size, hid_size):
        super(ResNet, self).__init__()

        self.fc1 = nn.Linear(in_size, hid_size)
        self.fc2 = nn.Linear(hid_size, in_size)
        self.bn = nn.BatchNorm2d(hid_size)

    def forward(self, x, training):
        if training:
            noise = 0.2
        else:
            noise = 0

        std = Variable(torch.randn(x.size())) * noise
        if x.is_cuda:
            std = std.cuda()

        residue = self.fc1(F.relu(x + std))
        residue = F.relu(residue).transpose(1,2).contiguous()
        residue = self.bn(residue).transpose(1,2)
        return x + self.fc2(residue)


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
            n_highway=4, 
            n_rnn_layers=1):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.s = s

        # Embedding
        self.embedding = nn.Embedding(vocab_size, src_emb)

        # half convolution with 8-grams
        self.n_filters = [200, 200, 250, 250, 300, 300, 300, 300]
        for i in range(len(self.n_filters)):
            conv2d = nn.Conv2d(1, self.n_filters[i], (i+1, src_emb), padding=(i, 0), bias=True)
            setattr(self, "conv_layer{}".format(i+1), conv2d)

        # number of layers in highway network
        self.n_highway = n_highway
        highway_dim = sum(self.n_filters)
        #for i in range(n_highway):
        #    setattr(self, "highway_gate{}".format(i+1), nn.Linear(highway_dim, highway_dim))
        #    setattr(self, "highway_layer{}".format(i+1), nn.Linear(highway_dim, highway_dim))

        # ResNet
        for i in range(n_highway):
            setattr(self, "resnet{}".format(i+1), ResNet(highway_dim, 400))

        # recurrent layer
        self.n_rnn_layers = n_rnn_layers
        self.rnn_encoder = nn.GRU(highway_dim, hid_dim, self.n_rnn_layers, 
                batch_first=True, dropout=dropout, bidirectional=True)

        self._init_weights()


    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_rnn_layers*2, batch_size, self.hid_dim))


    def forward(self, x, h, seq_len=None):
        # Embedding
        x = self.embedding(x)

        # Convoluation
        x = torch.unsqueeze(x, 1)
        x = self._convolution(x)

        # Max pooling with stride
        x = self._max_pooling(x)

        # Highway network
        #x = self._highway_net(x)

        # ResNet
        x = self._resnet(x)

        # Recurrent layer
        x, h = self._recurrent(x, h)

        return x, h


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
        """@depreciated
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


    def _resnet(self, x):
        """
        Multi-layer residual network

        @parameters
            same as highway network
        """
        y = x

        for i in range(self.n_highway):
            y = getattr(self, "resnet{}".format(i+1))(y, self.training)

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
        self.rnn_encoder.flatten_parameters()
        return self.rnn_encoder(x, h)


class DecoderC(nn.Module):
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
        input_dim = tar_emb# + context_dim
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
            alpha: tensor, attention weights of each segments, with dimension 
                (batch_size, seq_len)
        """
        y = self.embedding(prev_y.long())

        # attention mechanism
        batch_size = y.size(0)
        prev_h = prev_s.transpose(0,1).contiguous().view(batch_size, -1)
        context, attn = self._attention(prev_h, y, z)

        # update memory cell
        #cat_input = torch.cat((y, context), 1).unsqueeze(1)
        cat_input = y.unsqueeze(1)
        output, h = self.rnn_decoder(cat_input, prev_s)

        # decode
        next_char = self._decode(context)

        return next_char, h, attn


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
            alpha: tensor, attention weights of each segments, with dimension 
                (batch_size, seq_len)
        ----------
        """
        batch_size, seq_len, dim = z.size()
        c = 0
        betas = []
        for i in range(seq_len):
            z_i = z[:,i,:]
            x = torch.cat((prev_s, prev_y, z_i), 1)
            beta = self._align(x)
            betas.append(beta)

        betas = torch.stack(betas, 1)
        alpha = F.softmax(betas)
        c = alpha * z

        return c.sum(1), alpha.squeeze()


    def _decode(self, c):
        x = c
        for i in range(len(self.decode_layers)-1):
            x = getattr(self, "decoder_layer{}".format(i+1))(x)
        return x


def lr_schedule(t):
    """
    learning rate schedule, use piecewise
    """
    if t < 10:
        return 0.001

    if t < 50:
        return 0.0005

    return 0.0001


def main():
    conf = config.Config()
    conf.word_level = False

    # Load data
    vocab, idx2char = utils.build_vocab(
            [conf.train_path, conf.dev_path, conf.test_path], 
            False)
    print("vocabulary size = {}".format(len(vocab)))

    train_source_seqs, train_target_seqs = utils.load_data(
            conf.train_path, 
            vocab, 
            conf.train_pickle,
            conf.max_seq_len, 
            conf.reverse_source)

    dev_source_seqs, dev_target_seqs = utils.load_data(
            conf.dev_path, 
            vocab, 
            conf.dev_pickle,
            conf.max_seq_len, 
            conf.reverse_source)

    print("Training set = {}\nValidation set = {}".format(
        len(train_source_seqs), len(dev_source_seqs)))

    # Define/Load models
    if os.path.exists(conf.save_path+"/encoderC") and not conf.debug_mode:
        encoder = torch.load(conf.save_path+"/encoderC")
        decoder = torch.load(conf.save_path+"/decoderC")
    else:
        encoder = Encoder(
                conf.source_emb, 
                conf.hid_dim, 
                len(vocab), 
                conf.dropout, 
                conf.stride)
        decoder = Decoder(
                conf.target_emb, 
                conf.hid_dim, 
                len(vocab), 
                conf.dropout)

    if conf.debug_mode:
        train_source_seqs = train_source_seqs[:10]
        train_target_seqs = train_target_seqs[:10]
        dev_source_seqs = train_source_seqs
        dev_target_seqs = train_target_seqs
        test_source_seqs = train_source_seqs
        test_target_seqs = train_target_seqs
    else:
        # split large corpus to fit memory
        size = 70000
        n = 0 % (int(np.ceil(len(train_source_seqs)/size)))
        start = n * size
        train_source_seqs = train_source_seqs[start:start+size]
        train_target_seqs = train_target_seqs[start:start+size]

    if conf.cuda:
        encoder.cuda()
        decoder.cuda()

    lr = conf.lr
    bleu_n = conf.bleu_n
    best_bleu = -1
    best_encoder = None
    best_decoder = None

    for epoch in range(conf.epochs):
        lr = lr_schedule(epoch)
        print("*** Epoch [{:5d}] lr = {} ***".format(epoch, lr))

        train_loss = train(train_source_seqs, train_target_seqs, 
                encoder, decoder, lr, conf)
        print("Training loss: {:5.6f}".format(train_loss))

        ## BLEU score
        bleu_epoch = []
        for j, (src_trn, ref_trn, out_trn) in enumerate(evaluate(dev_source_seqs, dev_target_seqs,
                                                       encoder, decoder, conf)):
            for i in range(len(src_trn)):

                out_seq = utils.convert2sequence(out_trn[i], idx2char, "")[: -6]
                ref_seq = utils.convert2sequence(ref_trn[i], idx2char, "")[6:-6]

                bleu_trn = eval.BLEU(out_seq, ref_seq, bleu_n)

                bleu_epoch.append(bleu_trn)

        bleu = sum(bleu_epoch) / len(bleu_epoch)
        print('Bleu_' + str(bleu_n) + ' Score =', bleu)

        if bleu > best_bleu:
            best_bleu = bleu
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)
            torch.save(encoder, conf.save_path+"/encoderC")
            torch.save(decoder, conf.save_path+"/decoderC")

    if not conf.debug_mode:
        test_source_seqs, test_target_seqs = utils.load_data(
                conf.test_path, 
                vocab, 
                conf.test_pickle,
                conf.max_seq_len, 
                conf.reverse_source)

        del train_source_seqs, train_target_seqs
        del dev_source_seqs, dev_target_seqs

    bleus = []
    for _, (src, ref_trn, out_trn) in enumerate(evaluate(
        test_source_seqs, test_target_seqs, best_encoder, best_decoder, conf)):
        for i in range(len(src)):
            ## BLEU score
            out_seq = utils.convert2sequence(out_trn[i], idx2char, "")[: -6]
            ref_seq = utils.convert2sequence(ref_trn[i], idx2char, "")[6:-6]

            bleu = eval.BLEU(out_seq, ref_seq, bleu_n)

            print("Source\t{}".format(
                utils.convert2sequence(src[i], idx2char, "")))
            print("Ref\t{}".format(ref_seq))
            print("Output\t{}\n".format(out_seq))
            print('Bleu_' + str(bleu_n) + ' Score =', bleu)

            bleus.append(bleu)

    print('average BLEU score on test set = {:2.6f}'.format(
        sum(bleus) / len(bleus)))

    #with open(conf.save_path+"/encoderC", "wb") as f:
    #    torch.save(encoder, f)
    #with open(conf.save_path+"/decoderC", "wb") as f:
    #    torch.save(decoder, f)


if __name__ == "__main__":
    main()
