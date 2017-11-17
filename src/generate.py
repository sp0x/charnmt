import torch
from torch.autograd import Variable

import numpy as np


class generator(object):

    def __init__(self, model, idx2char):
        self.model = model
        self.idx2char = idx2char


    def translate(self, source):
        """
        Translate a single sentence into target language

        ----------
        @params
            source: list or numpy array, source sentence to be translated

        @return
            translation: string, a sequence that has been translated
        ----------
        """
        source = np.array(source).reshape(1, -1)

        batch_size, max_len = source.shape
        enc_h = model.encoder.init_gru(batch_size)
        source = Variable(torch.Tensor(source.tolist()), volatile=True)

        #source = source.cuda()
        #enc_h = enc_h.cuda()
        #dec_h = dec_h.cuda()

        context, dec_h = self.model.encoder(source, enc_h)
        translation = ""
        char_idx = 1 # <SOS>
        while char_idx != 2 and len(translation) < max_len:
            prev_char = Variable(torch.LongTensor([char_idx]))#.cuda()
            next_char, dec_h, attn = self.model(prev_char, context, dec_h)
            char_idx = next_char.data.topk(1)[1][0][0] # greedy decoding
            translation += self.idx2char[char_idx]

        return translation


if __name__ == "__main__":
    import CharNMT
    import config
    import utils
    import random


    conf = config.Config()
    vocab, idx2char = utils.build_char_vocab(
            [conf.train_path, conf.dev_path, conf.test_path])

    test_source_seqs, test_target_seqs = utils.load_data(
            conf.train_path, 
            vocab, 
            conf.train_pickle,
            conf.max_seq_len, 
            conf.reverse_source)

    random.seed(123456)
    order = list(range(len(test_source_seqs)))
    random.shuffle(order)
    order = order[:1]

    test_source_seqs = np.array(test_source_seqs)[:conf.batch_size]
    test_target_seqs = np.array(test_target_seqs)[:conf.batch_size]

    save_file = conf.save_path + "/CharNMT_100"
    model = torch.load(save_file)

    """
    model = CharNMT.CharNMT(
            len(vocab), 
            conf.max_seq_len, 
            conf.source_emb, 
            conf.target_emb, 
            conf.hid_dim, 
            conf.dropout, 
            conf.stride)
    """
    gen = generator(model, idx2char)

    for batch, (x, y, mask) in enumerate(utils.batchify(
        test_source_seqs, test_target_seqs, conf.stride)):

        for i in range(x.shape[0]):
            translation = gen.translate(x[i])
            print("Source:\t{}\n".format(
                utils.convert2sequence(x[i], idx2char)))
            print("Reference:\t{}\n".format(
                utils.convert2sequence(y[i], idx2char)))
            print("Translation:\t{}\n".format(translation))

