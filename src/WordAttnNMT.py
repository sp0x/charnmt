import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

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

        self.embedding = nn.Embedding(vocab_size, src_emb)
        self.gru = nn.GRU(src_emb, hid_dim, n_layers, 
                batch_first=True, dropout=dropout, bidirectional=True)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.hid_dim))

    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.gru(x, h)
        return x, h


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

        self.embedding = nn.Embedding(vocab_size, tar_emb)
        self.attn = nn.Linear(hid_dim * 4 + tar_emb, 1)
        self.gru = nn.GRU(hid_dim*2, hid_dim, n_layers, 
                batch_first=True, dropout=dropout, bidirectional=True)
        self.out = nn.Linear(hid_dim*2, vocab_size)

    def forward(self, x, h, encoder_out):
        x = self.embedding(x)

        batch_size, encoder_len, dim = encoder_out.size()
        prev_h = h.transpose(0,1).contiguous().view(batch_size, -1)
        betas = Variable(torch.zeros(batch_size, encoder_len))
        for i in range(encoder_len):
            betas[:,i] = F.softmax(self.attn(
                torch.cat((x, prev_h, encoder_out[:,i,:]), 1)))
        attn_weights = F.softmax(betas)

        c = torch.bmm(attn_weights.unsqueeze(1), encoder_out)
        for i in range(self.n_layers):
            c = F.relu(c)
            c, h = self.gru(c, h)

        c = F.log_softmax(self.out(c[-1]))
        return c, h, attn_weights


def train(source, target, encoder, decoder, lr, conf):
    encoder.train()
    decoder.train()
    enc_opt = optim.Adam(encoder.parameters(), lr=lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    total_loss = 0
    for batch, (x, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, conf.batch_size, False)):
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        loss = 0

        batch_size, max_len = x.shape
        x = Variable(torch.LongTensor(x.tolist()), volatile=False)
        y = Variable(torch.LongTensor(y.tolist()))

        enc_h = encoder.init_hidden(batch_size)

        if conf.cuda:
            x = x.cuda()
            y = y.cuda()
            enc_h = enc_h.cuda()

        encoder_out, dec_h = encoder(x, enc_h)

        for i in range(1, y.size(1)):
            decoder_out, dec_h, attn = decoder(y[:,i-1], dec_h, encoder_out)
            loss += loss_fn(decoder_out, y[:,i])

        total_loss += loss
        loss.backward()

        enc_opt.step()
        dec_opt.step()

    return total_loss.data[0] / len(source)


def evaluate(source, target, encoder, decoder, conf, idx2char):
    encoder.eval()
    decoder.eval()
    loss_fn = nn.NLLLoss()

    total_loss = 0
    translations = []
    for batch, (x, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, 1, False)):
        loss = 0

        batch_size, max_len = x.shape
        x = Variable(torch.LongTensor(x.tolist()), volatile=False)
        y = Variable(torch.LongTensor(y.tolist()))

        enc_h = encoder.init_hidden(batch_size)

        if conf.cuda:
            x = x.cuda()
            y = y.cuda()
            enc_h = enc_h.cuda()

        encoder_out, dec_h = encoder(x, enc_h)

        translation = ""
        for i in range(1, y.size(1)):
            decoder_out, dec_h, attn = decoder(y[:,i-1], dec_h, encoder_out)
            loss += loss_fn(decoder_out, y[:,i])
            char = idx2char[decoder_out.data.topk(1)[1][0][0]]
            translation += char + " "

            if char == "<EOS>":
                break

        total_loss += loss
        translations.append(translation)

    return total_loss.data[0] / len(source), translations


def main():
    conf = config.Config()
    vocab, idx2char = utils.build_char_vocab(
            [conf.train_path, conf.dev_path, conf.test_path], conf.word_level)

    train_source_seqs, train_target_seqs = utils.load_data(
            conf.train_path, 
            vocab, 
            conf.train_pickle,
            conf.max_seq_len, 
            conf.reverse_source)

    encoder = Encoder(
            conf.source_emb, 
            conf.hid_dim, 
            len(vocab), 
            conf.dropout)
    decoder = Decoder(
            conf.target_emb, 
            conf.hid_dim, 
            len(vocab), 
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

        loss, translations = evaluate(train_source_seqs, train_target_seqs, 
                encoder, decoder, conf, idx2char)
        print("Validation loss: {:5.6f}".format(loss))

        if epoch == conf.epochs - 1:
            for i in range(len(train_source_seqs)):
                print("Source\t{}".format(
                    utils.convert2sequence(train_source_seqs[i], idx2char)))
                print("Ref\t{}".format(
                    utils.convert2sequence(train_target_seqs[i], idx2char)))
                print("Output\t{}\n".format(translations[i]))


    with open(conf.save_path+"/encoderW", "wb") as f:
        torch.save(encoder, f)
    with open(conf.save_path+"/decoderW", "wb") as f:
        torch.save(decoder, f)


if __name__ == "__main__":
    main()
