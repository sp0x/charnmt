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
        enc_h, dec_h = model.init_hidden(batch_size)

        source = Variable(torch.Tensor(source.tolist()), volatile=True)

        context = self.model.compute_context(source, enc_h)
        translation = ""
        char_idx = 1 # <SOS>
        while char_idx != 2:
            prev_char = Variable(torch.LongTensor([char_idx]))
            next_char, dec_h = self.model(prev_char, context, dec_h)
            char_idx = next_char.data.topk(1)[1][0][0] # greedy decoding
            translation += self.idx2char[char_idx]

        return translation


if __name__ == "__main__":
    import CharNMT
    import config
    import utils

    conf = config.Config()
    vocab, idx2char = utils.build_char_vocab(
            [conf.train_path, conf.dev_path, conf.test_path])

    model = CharNMT.CharNMT(
            len(vocab), 
            conf.max_seq_len, 
            conf.source_emb, 
            conf.target_emb, 
            conf.hid_dim, 
            conf.dropout, 
            conf.stride)

    sample = "Ich lerne in New York."
    source_seq = [vocab["<SOS>"]] + [vocab[c] for c in sample] + [vocab["<EOS>"]]
    source_seq += [vocab["<PAD>"]] * (conf.stride - len(source_seq) % conf.stride)

    gen = generator(model, idx2char)
    translation = gen.translate(source_seq)
    print("Translation:\t{}\n".format(translation))
