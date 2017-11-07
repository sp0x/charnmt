# -*- coding: utf-8 -*-
"""
@Author: Viola

This module pre-processes the datasets before passed into the model.
It maps the de-en sentence pairs with tokenized words to sequences.

"""

import re
import random
import collections
import torch
import numpy as np
import nltk

#%%
    
with open("train.txt", "r", encoding="utf8") as f:
    
    DE_seq = []
    EN_seq = []
    
    for i, line in enumerate(f):
        
        line = line.split('<JOIN>')
        DE_seq.append(line[0])
        EN_seq.append(line[1])


#%%
def mapper(sequence):
    
    corpus = ' '.join(sequence)
    vocab = []

    for i in corpus:
        if i not in vocab:
            vocab.append(i)
                 
    vocab = set(vocab)
    
    mapper = {}
    
    onehot_mat = np.eye(len(vocab))
    
    for i, v in enumerate(vocab):
        mapper[v] = onehot_mat[i]
        
    return mapper, len(mapper)


DE_mapper, DE_mapper_length = mapper(DE_seq)
EN_mapper, EN_mapper_length = mapper(EN_seq)

#%%

import string

printable = list(string.printable)

DE_tokens = list(DE_mapper.keys())
EN_tokens = list(EN_mapper.keys())

DE_out_of_range = [i for i in DE_tokens if i not in printable]
EN_out_of_range = [i for i in EN_tokens if i not in printable]

