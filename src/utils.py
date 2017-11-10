import pickle
import os
import random
import numpy as np


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


def load_data(filename, vocab, save_path, max_len):
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
            
            if len(source) <= max_len:
  
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



def batchify(data, label, stride, batch_size=None, shuffle=False):
    
    if not batch_size:
        batch_size = len(data)

    data_size = len(data)
    order = list(range(data_size))
    if shuffle:
        random.shuffle(order)

    num_batches = int(np.ceil(1.*data_size / batch_size))
    
    for i in range(num_batches):
        
        start = i * batch_size
        indices = order[start: start+batch_size]
        padded_data = padding(data[indices], stride)
        
        yield padded_data, label[indices]

            
            

def padding(batch_data, stride):
    """
    For source sequence data in a batch, 
    zero-pad the short ones up to the multiples of stride
    
    ----------
    @param 
        batch_data: numpy array, has dimension (batch_size, seq_len, n_feature)
        stride: int, stride size
        
    @return 
        padded_data: numpy array, same format as batch_data, but padded
    ----------
    """
    lens = [data.shape[0] for data in batch_data]
    max_len = max(lens)
    max_len += stride - (max_len % stride)

    batch_size = len(batch_data)
    nfeature = batch_data[0].shape[1]
    padded_data = np.zeros([batch_size, max_len, nfeature])
    
    for i in range(batch_size):
        length = batch_data[i].shape[0]
        pad = np.pad(batch_data[i], (0, max_len-length), "constant")
        padded_data[i] = pad[:, :nfeature]

    return padded_data
    

