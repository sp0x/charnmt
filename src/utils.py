import pickle
import os
import random
import numpy as np


def build_char_vocab(filenames):
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
    for filename in filenames:
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


def load_data(file_path, vocab, pickle_path, max_len, reverse_source):
    """
    Load source and target language sequences, each list contains a list of 
    character indices converted from vocabulary

    ----------
    @params
        file_path: string, input file path
        vocab: dict, the vocabulary generated from a large corpora
        pickle_path: string, the location of the pickled data
        max_len: int, the maximum source sequence length, used to filter longer 
            sequences
        reverse_source: bool, reverse the source sentence order, which may 
            improve the final performance of machine translation

    @return
        source_seqs: list, a list of source language sentence
        target_seqs: list, a list of target language sentence
    ----------
    """

    def reverse_order(seq):
        for i in range(len(seq)):
            seq[i] = list(reversed(seq[i]))
        return seq

    if os.path.exists(pickle_path.format("source")):
        source_seqs = pickle.load(open(pickle_path.format("source"), "rb"))
        target_seqs = pickle.load(open(pickle_path.format("target"), "rb"))

        if reverse_source:
            source_seqs = reverse_order(source_seqs)
        
        return source_seqs, target_seqs

    source_seqs = []
    target_seqs = []
    with open(file_path, "r", encoding="utf-8") as f:
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

    pickle.dump(source_seqs, open(pickle_path.format("source"), "wb"))
    pickle.dump(target_seqs, open(pickle_path.format("target"), "wb"))

    if reverse_source:
        source_seqs = reverse_order(source_seqs)

    return source_seqs, target_seqs


def batchify(data, label, stride, batch_size=None, shuffle=False):
    
    if not batch_size:
        batch_size = len(data)

    data = np.array(data)
    label = np.array(label)

    data_size = len(data)
    order = list(range(data_size))
    if shuffle:
        random.shuffle(order)

    num_batches = int(np.ceil(1.*data_size / batch_size))
    
    for i in range(num_batches):
        
        start = i * batch_size
        indices = order[start: start+batch_size]

        padded_data = pad_data(data[indices], stride)
        padded_label, label_mask = pad_label(label[indices])
        
        yield padded_data, padded_label, label_mask
            

def pad_data(batch_data, stride):
    """
    For source sequence data in a batch, 
    zero-pad the short ones up to the multiples of stride
    
    ----------
    @param 
        batch_data: numpy array, has dimension (batch_size, seq_len)
        stride: int, stride size
        
    @return 
        padded_data: numpy array, same format as batch_data, but padded
    ----------
    """
    lens = [len(data) for data in batch_data]
    max_len = max(lens)
    max_len += stride - (max_len % stride)

    batch_size = len(batch_data)
    n_tokens = len(batch_data[0])
    padded_data = np.zeros([batch_size, max_len], dtype=np.int32)
    
    for i in range(batch_size):
        length = len(batch_data[i])
        pad = np.pad(batch_data[i], (0, max_len-length), "constant")
        padded_data[i] = pad

    return padded_data
    

def pad_label(batch_label):
    """
    For target sequence data in a batch, 
    zero-pad the short ones up to the max. length in the batch
    
    ----------
    @param 
        batch_label: numpy array, has dimension (batch_size, seq_len)
        
    @return 
        padded_label: numpy array, same format as batch_label, but padded
        label_mask: numpy array, the padded entries and the first entry 
            are zeros.
    ----------
    """
    lens = [len(data) for data in batch_label]
    max_len = max(lens)

    batch_size = len(batch_label)
    n_tokens = len(batch_label[0])
    padded_label = np.zeros([batch_size, max_len], dtype=np.int32)
    label_mask = np.zeros([batch_size, max_len])
    
    for i in range(batch_size):
        length = len(batch_label[i])
        pad = np.pad(batch_label[i], (0, max_len-length), "constant")
        padded_label[i] = pad
        label_mask[1:length] = 1

    return padded_label, label_mask

