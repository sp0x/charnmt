import pickle
import os


def build_char_vocab(filename):
    """
    Build character vocabulary from a file

    ----------
    @params
        filename: string, input filename

    @return
        vocab: dict, pairs of char tokens and their indices
        idx2char: list, mapping index to character
    ----------
    """
    vocab = {
            "<PAD>" : 0,
            "<SOS>" : 1,
            "<EOS>" : 2, 
            "<UNK>" : 3,
            }
    idx2char = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            source, target = line.split("<JOIN>")
            source = source[:-5].strip()
            target = target[:-5].strip()
            for c in source + target:
                if c not in vocab:
                    vocab[c] = len(vocab)
                    idx2char.append(c)
    
    return vocab, idx2char


def load_data(filename, vocab, save_path):
    """
    Load source and target language sequences, each list contains a list of 
    character indices converted from vocabulary

    ----------
    @params
        filename: string, input file path
        vocab: dict, the vocabulary generated from a large corpora

    @return
        source_seqs: list, a list of source language sentence
        target_seqs: list, a list of target language sentence
    ----------
    """
    if os.path.exists(save_path+"/source.p"):
        source_seqs = pickle.load(open(save_path+"/source.p", "rb"))
        target_seqs = pickle.load(open(save_path+"/target.p", "rb"))
        
        return source_seqs, target_seqs

    source_seqs = []
    target_seqs = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            source_seq = [vocab["<SOS>"]]
            target_seq = [vocab["<SOS>"]]

            source, target = line.split("<JOIN>")
            source = source[:-5].strip()
            target = target[:-5].strip()

            for c in source:
                source_seq.append(vocab[c])
            source_seq.append(vocab["<EOS>"])

            for c in target:
                target_seq.append(vocab[c])
            target_seq.append(vocab["<EOS>"])

            source_seqs.append(source_seq)
            target_seqs.append(target_seq)

    pickle.dump(source_seqs, open(save_path + "/source.p", "wb"))
    pickle.dump(target_seqs, open(save_path + "/target.p", "wb"))

    return source_seqs, target_seqs

