import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import random
import os
import pickle
import numpy as np
import copy

import utils
import eval
import config


class Encoder(nn.Module):

    def __init__(self, 
            src_emb, 
            hid_dim, 
            vocab_size, 
            dropout, 
            n_layers=1):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, src_emb, padding_idx=0)
        self.gru = nn.GRU(src_emb, hid_dim, n_layers, 
                batch_first=True, dropout=dropout, bidirectional=True)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers*2, batch_size, self.hid_dim))

    def forward(self, x, h, seq_len):
        x = self.embedding(x)

        packed_seq = pack_padded_sequence(x, seq_len, batch_first=True)
        output, h = self.gru(packed_seq, h)
        output, out_len = pad_packed_sequence(output, True)

        return output, h

class Decoder(nn.Module):

    def __init__(self, 
            tar_emb, 
            hid_dim, 
            vocab_size, 
            dropout, 
            n_layers=1):
        super(Decoder, self).__init__()
        
        self.n_layers = n_layers
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, tar_emb, padding_idx=0)
        self.attn = nn.Linear(hid_dim * 3, 1)
        self.gru = nn.GRU(hid_dim*2+tar_emb, hid_dim, n_layers, 
                batch_first=True, dropout=dropout)
        self.out = nn.Linear(hid_dim * 3, vocab_size)

    def forward(self, x, h, encoder_out, use_cuda):
        x = self.embedding(x)

        batch_size, encoder_len, dim = encoder_out.size()
        last_hidden = h[-1]
        betas = Variable(torch.zeros(batch_size, encoder_len))
        if use_cuda:
            betas = betas.cuda()

        for i in range(encoder_len):
            betas[:,i] = F.softmax(self.attn(
                torch.cat((last_hidden, encoder_out[:,i,:]), dim=1)))
        attn_weights = F.softmax(betas).unsqueeze(dim=1)

        context = attn_weights.bmm(encoder_out)

        rnn_input = torch.cat((x, context), dim=2)
        self.gru.flatten_parameters()
        output, hidden = self.gru(rnn_input, h)

        output = F.log_softmax(self.out(
            torch.cat((output.squeeze(dim=1), context.squeeze(dim=1)), dim=1)))

        return output, hidden, attn_weights


def train(source, target, encoder, decoder, lr, conf):
    """
    ----------
    @params
        source: list of list, sequences of source language
        target: list of list, sequences of target language
        encoder: Encoder, object of encoder in NMT
        decoder: Decoder, object of decoder in NMT
        lr: float, learning rate
        conf: Config, wraps anything needed
    ----------
    """
    encoder.train()
    decoder.train()
    enc_opt = optim.Adam(encoder.parameters(), lr=lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    total_loss = 0
    for batch, (x, x_len, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, conf.batch_size, True)):
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        loss = 0

        x = x[:,1:] # skip <SOS>
        batch_size, src_len = x.shape
        x = Variable(torch.LongTensor(x.tolist()), volatile=False)
        y = Variable(torch.LongTensor(y.tolist()))

        enc_h = encoder.init_hidden(batch_size)

        if conf.cuda:
            x = x.cuda()
            y = y.cuda()
            enc_h = enc_h.cuda()
        print(x.size())
        encoder_out, enc_h = encoder(x, enc_h, x_len-1)
        # use last forward hidden state in encoder
        dec_h = enc_h[:decoder.n_layers]
        #dec_h = decoder.init_hidden(enc_h)

        target_len = y.size(1)
        decoder_input = y[:, 0:1]

        # Scheduled sampling
        use_teacher_forcing = random.random() < conf.teaching_ratio

        if use_teacher_forcing:
            for i in range(1, target_len):
                decoder_out, dec_h, ATT = decoder(decoder_input, dec_h, encoder_out, conf.cuda)
                loss += utils.loss_in_batch(decoder_out, y[:,i], mask[:,i], loss_fn)
                decoder_input = y[:, i:i+1]

        else:
            for i in range(1, target_len):
                decoder_out, dec_h, ATT = decoder(decoder_input, dec_h, encoder_out, conf.cuda)
                loss += utils.loss_in_batch(decoder_out, y[:,i], mask[:,i], loss_fn)

                topv, topi = decoder_out.data.topk(1)
                ni = topi[:,:1]
                decoder_input = Variable(torch.LongTensor(ni.tolist()))
                if conf.cuda:
                    decoder_input = decoder_input.cuda()

        total_loss += loss.data[0]
        loss /= batch_size
        loss.backward()

        enc_opt.step()
        dec_opt.step()

    return total_loss / len(source)


def evaluate(source, target, encoder, decoder, conf, max_len=50):
    """
    ----------
    @params
        source: list of list, sequences of source language
        target: list of list, sequences of target language
        encoder: Encoder, object of encoder in NMT
        decoder: Decoder, object of decoder in NMT
        conf: Config, wraps anything needed
        max_len: int, max length of generated translation
    ----------
    """
    if not conf.word_level:
        max_len *= 6

    encoder.eval()
    decoder.eval()
    loss_fn = nn.NLLLoss()

    for batch, (x, x_len, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, conf.batch_size, False)):

        src = x[:,1:] # skip <SOS>
        batch_size, src_len = src.shape
        translations = np.zeros([batch_size, max_len], dtype=int)
        src = Variable(torch.LongTensor(src.tolist()), volatile=True)
        y = Variable(torch.LongTensor(y.tolist()))

        enc_h = encoder.init_hidden(batch_size)

        if conf.cuda:
            src = src.cuda()
            y = y.cuda()
            enc_h = enc_h.cuda()

        encoder_out, enc_h = encoder(src, enc_h, x_len-1)
        # use last forward hidden state in encoder
        dec_h = enc_h[:decoder.n_layers]
        #dec_h = decoder.init_hidden(enc_h)

        word = x[:,:1]
        for i in range(1, max_len+1):
            word = Variable(torch.LongTensor(word.tolist()))
            if conf.cuda:
                word = word.cuda()

            decoder_out, dec_h, ATT = decoder(word, dec_h, encoder_out, conf.cuda)

            word = decoder_out.data.topk(1)[1]
            start = batch * conf.batch_size
            translations[:,i-1:i] = word.cpu().numpy()

        yield x, y.data.cpu().numpy(), translations


def main():
    conf = config.Config()

    # Load data
    if os.path.exists(conf.data_path + "/src_vocab.p"):
        src_vocab = pickle.load(open(conf.data_path + "/src_vocab.p", "rb"))
        tar_vocab = pickle.load(open(conf.data_path + "/tar_vocab.p", "rb"))
        src_idx2token = pickle.load(open(conf.data_path + "/src_idx2token.p", "rb"))
        tar_idx2token = pickle.load(open(conf.data_path + "/tar_idx2token.p", "rb"))
    else:
        src_vocab, src_idx2token, tar_vocab, tar_idx2token = utils.build_vocab(
                [conf.train_path, conf.dev_path, conf.test_path], True)
        pickle.dump(src_vocab, open(conf.data_path + "/src_vocab.p", "wb"))
        pickle.dump(tar_vocab, open(conf.data_path + "/tar_vocab.p", "wb"))
        pickle.dump(src_idx2token, open(conf.data_path + "/src_idx2token.p", "wb"))
        pickle.dump(tar_idx2token, open(conf.data_path + "/tar_idx2token.p", "wb"))
    
    print("src vocab = {}\ntar vocab = {}".format(len(src_vocab), len(tar_vocab)))

    train_source_seqs, train_target_seqs = utils.load_data(
            conf.train_path, 
            [src_vocab, tar_vocab], 
            conf.train_pickle,
            conf.max_seq_len, 
            conf.reverse_source)

    dev_source_seqs, dev_target_seqs = utils.load_data(
            conf.dev_path, 
            [src_vocab, tar_vocab], 
            conf.dev_pickle,
            conf.max_seq_len, 
            conf.reverse_source)
    print("Training set = {}\nValidation set = {}".format(
        len(train_source_seqs), len(dev_source_seqs)))

    # Define/Load models
    if os.path.exists(conf.save_path+"/encoderWA") and not conf.debug_mode:
        encoder = torch.load(conf.save_path+"/encoderWA")
        decoder = torch.load(conf.save_path+"/decoderWA")
    else:
        encoder = Encoder(
                conf.source_emb, 
                conf.hid_dim, 
                len(src_vocab), 
                conf.dropout)
        decoder = Decoder(
                conf.target_emb, 
                conf.hid_dim, 
                len(tar_vocab), 
                conf.dropout)

    if conf.debug_mode:
        train_source_seqs = train_source_seqs[:100]
        train_target_seqs = train_target_seqs[:100]
        dev_source_seqs = dev_source_seqs[:10]
        dev_target_seqs = dev_target_seqs[:10]
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
        print("*** Epoch [{:5d}] lr = {} ***".format(epoch, lr))

        train_loss = train(train_source_seqs, train_target_seqs, 
                encoder, decoder, lr, conf)
        print("Training loss: {:5.6f}".format(train_loss))

        ## BLEU score
        bleu_epoch = []
        for j, (src_trn, ref_trn, out_trn) in enumerate(evaluate(train_source_seqs, train_target_seqs,
                                                       encoder, decoder, conf)):
            for i in range(len(src_trn)):

                out_seq = utils.convert2sequence(out_trn[i], tar_idx2token)[: -6]
                ref_seq = utils.convert2sequence(ref_trn[i], tar_idx2token)[6:-6]

                bleu_trn = eval.BLEU(out_seq, ref_seq, bleu_n)

                bleu_epoch.append(bleu_trn)

        bleu = sum(bleu_epoch) / len(bleu_epoch)
        print('Bleu_' + str(bleu_n) + ' Score =', bleu)

        if bleu > best_bleu:
            best_bleu = bleu
            best_encoder = copy.deepcopy(encoder)
            best_decoder = copy.deepcopy(decoder)
            torch.save(encoder, conf.save_path+"/encoderWA")
            torch.save(decoder, conf.save_path+"/decoderWA")

    if not conf.debug_mode:
        test_source_seqs, test_target_seqs = utils.load_data(
                conf.test_path, 
                [src_vocab, tar_vocab], 
                conf.test_pickle,
                conf.max_seq_len, 
                conf.reverse_source)

        del train_source_seqs, train_target_seqs
        del dev_source_seqs, dev_target_seqs
        del src_vocab, tar_vocab

    bleus = []
    for _, (src, ref_trn, out_trn) in enumerate(evaluate(
        test_source_seqs, test_target_seqs, best_encoder, best_decoder, conf)):
        for i in range(len(src)):
            ## BLEU score
            out_seq = utils.convert2sequence(out_trn[i], tar_idx2token)[: -6]
            ref_seq = utils.convert2sequence(ref_trn[i], tar_idx2token)[6:-6]

            bleu = eval.BLEU(out_seq, ref_seq, bleu_n)

            print("Source\t{}".format(
                utils.convert2sequence(src[i], src_idx2token)))
            print("Ref\t{}".format(ref_seq))
            print("Output\t{}\n".format(out_seq))
            print('Bleu_' + str(bleu_n) + ' Score =', bleu)

            bleus.append(bleu)

    print('average BLEU score on test set = {:2.6f}'.format(
        sum(bleus) / len(bleus)))

    #with open(conf.save_path+"/encoderW", "wb") as f:
    #    torch.save(encoder, f)
    #with open(conf.save_path+"/decoderW", "wb") as f:
    #    torch.save(decoder, f)


if __name__ == "__main__":
    main()
