# -*- coding: utf-8 -*-
"""
@Author: Viola

This module preprocess the dataset before passed into the model.
It convert the sentences with words to sequences with mapped indices.

"""

import re
import random
import collections
import torch
import numpy as np
import nltk

#%%
    
PATH = './de-en/'

def load_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            
            text = line.strip()
            example['text'] = text[:]
            
            if i >= 6:
                data.append(example)

    random.seed(1)
    random.shuffle(data)  # First 6 lines are not included.
    return data

#%%
train_de = load_data(PATH + 'train.tags.de-en.de')
train_en = load_data(PATH + 'train.tags.de-en.en')

#%%
def tokenize_merge(dataset):

    corpus = []

    for tt in dataset:

        sentence = tt['text']
        tokens = nltk.word_tokenize(sentence)

        corpus.append(" ".join(tokens))

    return " \n".join(corpus)


corpus_en = tokenize_merge(train_en)
with open("tokenized_en.txt", "w") as f:
    f.write(corpus_en)


corpus_de = tokenize_merge(train_de)
with open("tokenized_de.txt", "w") as f:
    f.write(corpus_de)
    
    
#%%
def get_vocab_mapping(data_file):
    
#    START = "<S>"
#    END = "<EOS>"
#    END_PADDING = "<PAD>"
#    UNKNOWN = "<UNK>"
    
    word_counter = collections.Counter()
    
    with open(data_file) as f:
        for line in f:
            word_counter.update(line.split())
    
    vocabulary = set([word for word in word_counter if word_counter[word] > 0]) 
    vocabulary = list(vocabulary)
#    vocabulary = [START, END, END_PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}
    
    return indices_to_words, word_indices, vocabulary


id2w_en, w2id_en, vocab_en = get_vocab_mapping("tokenized_en.txt")
id2w_de, w2id_de, vocab_de = get_vocab_mapping("tokenized_de.txt")


#%%    

def text2idx(data_file, w_id):
    
    idx_corpus = []
    
    with open(data_file) as f:
    
        for i, line in enumerate(f):
            
            row = []
            
            line = line.replace(u'\xa0', u' ')
            line = line.replace(u'\x85', u' ')
    
            tokens = line.split(' ')[:-1]
            tokens = list(filter(lambda x: len(x) > 0, tokens))

            for token in tokens:
                
                idx = w_id[token]
                row.append(idx)

            idx_corpus.append(' '.join([str(i) for i in row]))

    return ' \n'.join(idx_corpus)

# Processed corpus with index representation
trn_en = text2idx("tokenized_en.txt", w2id_en)
with open("train_idx_en.txt", "w") as f:
    f.write(trn_en)

trn_de = text2idx("tokenized_de.txt", w2id_de)
with open("train_idx_de.txt", "w") as f:
    f.write(trn_de)

######################################################
######################################################

#%%
    
def get_max_seq_len(dataset):
    
    '''
    Return the longest sentence in the training english set
    '''

    seq_len = []
    
    for tt in dataset:
        
        sentence = tt['text']
        tokens = nltk.word_tokenize(sentence)
        length = len(tokens)
        
        seq_len.append(length)
        
    return max(seq_len)

# get_max_seq_len(train_en)



#%%
# This is the iterator we'll use during training. 
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield batch

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    targets = []
    for d in batch:
        vectors.append(d["index_seq"])
        targets.append(d["target_seq"])
    return vectors, targets
