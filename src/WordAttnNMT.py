import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import random
import os
import pickle

import utils
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
        return Variable(torch.zeros(2, batch_size, self.hid_dim))

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

    def forward(self, x, h, encoder_out):
        x = self.embedding(x)

        batch_size, encoder_len, dim = encoder_out.size()
        last_hidden = h[-1]
        betas = Variable(torch.zeros(batch_size, encoder_len))
        for i in range(encoder_len):
            betas[:,i] = F.softmax(self.attn(
                torch.cat((last_hidden, encoder_out[:,i,:]), 1)))
        attn_weights = F.softmax(betas).unsqueeze(1)

        context = attn_weights.bmm(encoder_out)

        rnn_input = torch.cat((x.unsqueeze(1), context), 2)
        output, hidden = self.gru(rnn_input, h)

        output = F.log_softmax(self.out(
            torch.cat((output.squeeze(1), context.squeeze(1)), 1)))

        return output, hidden, attn_weights


def train(source, target, encoder, decoder, lr, conf):
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
        batch_size, max_len = x.shape
        x = Variable(torch.LongTensor(x.tolist()), volatile=False)
        y = Variable(torch.LongTensor(y.tolist()))

        enc_h = encoder.init_hidden(batch_size)

        if conf.cuda:
            x = x.cuda()
            y = y.cuda()
            enc_h = enc_h.cuda()

        encoder_out, enc_h = encoder(x, enc_h, x_len-1)
        dec_h = enc_h[:decoder.n_layers]

        decoder_input = y[:, 0]
        target_len = y.size(1)
        use_teacher_forcing = random.random() < conf.teaching_ratio

        if use_teacher_forcing:
            for i in range(1, target_len):
                decoder_out, dec_h, attn = decoder(decoder_input, dec_h, encoder_out)
                loss += utils.loss_in_batch(decoder_out, y[:,i], mask[:,i], loss_fn)
                decoder_input = y[:, i]

        else:
            for i in range(1, target_len):
                decoder_out, dec_h, attn = decoder(decoder_input, dec_h, encoder_out)
                loss += utils.loss_in_batch(decoder_out, y[:,i], mask[:,i], loss_fn)

                topv, topi = decoder_out.data.topk(1)
                ni = topi[:,0]
                decoder_input = Variable(torch.LongTensor(ni.tolist()))

        total_loss += loss
        loss /= batch_size
        loss.backward()

        enc_opt.step()
        dec_opt.step()

    return total_loss.data[0] / len(source)


def evaluate(source, target, encoder, decoder, conf, max_len=30):
    encoder.eval()
    decoder.eval()
    loss_fn = nn.NLLLoss()

    total_loss = 0
    translations = []
    for batch, (x, x_len, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, 1, False)):
        loss = 0

        src = x[:,1:] # skip <SOS>
        batch_size, src_len = src.shape
        src = Variable(torch.LongTensor(src.tolist()), volatile=False)
        y = Variable(torch.LongTensor(y.tolist()))

        enc_h = encoder.init_hidden(batch_size)

        if conf.cuda:
            src = src.cuda()
            y = y.cuda()
            enc_h = enc_h.cuda()

        encoder_out, enc_h = encoder(src, enc_h, x_len-1)
        dec_h = enc_h[:decoder.n_layers]

        attn_matrix = torch.zeros(max_len, src_len)
        translation = [1]
        char = x[:,0]
        for i in range(1, max_len+1):
            char = Variable(torch.LongTensor(char.tolist()))
            decoder_out, dec_h, attn = decoder(char, dec_h, encoder_out)
            loss += utils.loss_in_batch(decoder_out, y[:,i], mask[:,i], loss_fn)

            attn_matrix[i-1] = attn.data[0]
            char = decoder_out.data.topk(1)[1][:,0]
            translation.append(char[0])

            if char[0] == 2:
                break

        total_loss += loss
        translations.append(translation)

    return total_loss / len(source), translations, attn_matrix


def main():
    conf = config.Config()

    if os.path.exists(conf.data_path + "/src_vocab.p"):
        src_vocab = pickle.load(open(conf.data_path + "/src_vocab.p", "rb"))
        tar_vocab = pickle.load(open(conf.data_path + "/tar_vocab.p", "rb"))
        src_idx2token = pickle.load(open(conf.data_path + "/src_idx2token.p", "rb"))
        tar_idx2token = pickle.load(open(conf.data_path + "/tar_idx2token.p", "rb"))
    else:
        src_vocab, src_idx2token, tar_vocab, tar_idx2token = utils.build_char_vocab(
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

    test_source_seqs, test_target_seqs = utils.load_data(
            conf.test_path, 
            [src_vocab, tar_vocab], 
            conf.test_pickle,
            conf.max_seq_len, 
            conf.reverse_source)

    print("Training set = {}\nValidation set = {}\nTest set = {}".format(
        len(train_source_seqs), len(dev_source_seqs), len(test_source_seqs)))

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

    if conf.cuda:
        encoder.cuda()
        decoder.cuda()

    lr = conf.lr
    for epoch in range(conf.epochs):
        print("*** Epoch [{:5d}] lr = {} ***".format(epoch, lr))

        train_loss = train(train_source_seqs, train_target_seqs, 
                encoder, decoder, lr, conf)
        print("Training loss: {:5.6f}".format(train_loss))

        dev_loss, _, _ = evaluate(dev_source_seqs, dev_target_seqs, 
                encoder, decoder, conf):
        print("Validation loss: {:5.6f}".format(dev_loss))

    translations, attn_matrix = evaluate(test_source_seqs, 
            test_target_seqs, encoder, decoder, conf)

    for i in range(len(test_source_seqs)):
        print("Source\t{}".format(
            utils.convert2sequence(test_source_seqs[i], src_idx2token)))
        print("Ref\t{}".format(
            utils.convert2sequence(test_target_seqs[i], tar_idx2token)))
        print("Output\t{}\n".format(
            utils.convert2sequence(translations[i], tar_idx2token)))


    with open(conf.save_path+"/encoderWA", "wb") as f:
        torch.save(encoder, f)
    with open(conf.save_path+"/decoderWA", "wb") as f:
        torch.save(decoder, f)


if __name__ == "__main__":
    main()
