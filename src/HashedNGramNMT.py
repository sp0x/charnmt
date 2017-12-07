"""
Paper reference: http://www.petrovi.de/data/emnlp17b.pdf
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import os
import pickle

import utils
import eval
import config
from WordAttnNMT import Decoder, train, evaluate


class N_Gram_Embedding(nn.Module):

    def __init__(self, n_gram_size, emb_size, vocab, n=3):
        nn.Module.__init__(self)
        self.vocab = {v:k for k,v in vocab.items()}
        self.emb_size = emb_size
        self.n = n
        self.n_grams = n_gram_size - 4 # ignore special tokens <PAD>, <SOS>, <EOS>, <UNK>
        self.embedding = nn.Embedding(n_gram_size, emb_size, padding_idx=0)


    def _get_n_grams(self, word):
        n_grams = []
        for i in range(1, 1+self.n):
            n_grams += [self._hash(word[j:j+i]) for j in range(len(word)-i+1)]

        n_grams = Variable(torch.LongTensor(n_grams))
        return n_grams


    def _hash(self, word):
        return hash(word) % self.n_grams + 4


    def forward(self, word_idx):
        """
        word_idx: (batch_size, seq_len), Tensor
        """
        batch_size, seq_len = word_idx.size()
        embeddings = Variable(torch.zeros(batch_size, seq_len, self.emb_size))
        if word_idx.is_cuda:
            embeddings = embeddings.cuda()

        for b in range(batch_size):
            for i in range(seq_len):
                wi = word_idx[b,i].data.cpu().numpy()[0]
                if wi < 4:
                    embeddings[b,i,:] = self.embedding(
                            word_idx[b,i].view(1,1)).squeeze()
                else:
                    word = self.vocab[wi]
                    n_grams = self._get_n_grams(word)
                    if word_idx.is_cuda:
                        n_grams = n_grams.cuda()
                    n_gram_emb = self.embedding(n_grams).mean(0)
                    embeddings[b,i,:] = n_gram_emb

        return embeddings


class Encoder(nn.Module):

    def __init__(self, 
            src_emb, 
            hid_dim, 
            n_gram_size, 
            vocab, 
            dropout, 
            n_layers=1):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.hid_dim = hid_dim

        self.embedding = N_Gram_Embedding(n_gram_size, src_emb, vocab)
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


def main():
    conf = config.Config()

    # Load data
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
    print("Training set = {}\nValidation set = {}".format(
        len(train_source_seqs), len(dev_source_seqs)))

    # Define/Load models
    if os.path.exists(conf.save_path+"/encoderH") and not conf.debug_mode:
        encoder = torch.load(conf.save_path+"/encoderH")
        decoder = torch.load(conf.save_path+"/decoderH")
    else:
        encoder = Encoder(
                conf.source_emb, 
                conf.hid_dim, 
                conf.n_gram_size, 
                src_vocab, 
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

                bleu_trn = eval.BLEU([utils.convert2sequence(out_trn[i], tar_idx2token)],
                                     [utils.convert2sequence(ref_trn[i], tar_idx2token)[6:-6]])
                # print('Bleu Score =', bleu_trn)
                bleu_epoch.append(bleu_trn)

        bleu = sum(bleu_epoch) / len(bleu_epoch)
        print('Bleu Score =', bleu)



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

    for _, (src, ref, out) in enumerate(evaluate(
        test_source_seqs, test_target_seqs, encoder, decoder, conf)):
        for i in range(len(src)):
            print("Source\t{}".format(
                utils.convert2sequence(src[i], src_idx2token)))
            print("Ref\t{}".format(
                utils.convert2sequence(ref[i], tar_idx2token)))
            print("Output\t{}\n".format(
                utils.convert2sequence(out[i], tar_idx2token)))

            ## BLEU score
            bleu_score = eval.BLEU([utils.convert2sequence(out[i], tar_idx2token)],
                                   [utils.convert2sequence(ref[i], tar_idx2token)[6:-6]])
            print('Bleu Score =', bleu_score)



    with open(conf.save_path+"/encoderH", "wb") as f:
        torch.save(encoder, f)
    with open(conf.save_path+"/decoderH", "wb") as f:
        torch.save(decoder, f)


if __name__ == "__main__":
    main()
