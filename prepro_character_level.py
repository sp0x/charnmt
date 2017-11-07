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
    
PATH = './de-en/'

END = "<EOS>"
END_PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def load_data(path):
    data = []
    with open(path, 'r',  encoding="utf8") as f:
        for i, line in enumerate(f): 
            example = {}

            if line.startswith('<'):
                continue
            
            text = line.strip()
            example['text'] = text[:]

            data.append(example)

    random.seed(1)
    random.shuffle(data)
    return data

#%%
train_de = load_data(PATH + 'train.tags.de-en.de')
train_en = load_data(PATH + 'train.tags.de-en.en')

#%%
def build_corpus(dataset_from, dataset_to):

    corpus = []

    for i, (tt_from, tt_to) in enumerate(zip(dataset_from, dataset_to)):

        tokens_from = [tt_from['text']] + [END]
        tokens_to = [tt_to['text']] + [END]

        s_from = " ".join(tokens_from)
        s_to = " ".join(tokens_to)
        
        corpus.append(" ".join([s_from, s_to]))

    return " \n".join(corpus)


corpus_from_to = build_corpus(train_de, train_en)
with open("DATA_processed_new.txt", "w") as f:
    f.write(corpus_from_to)
    
#%%
    
with open("DATA_processed_new.txt", "r", encoding="utf8") as f:
    
    DE_seq = []
    EN_seq = []
    
    for i, line in enumerate(f):
        
        line = line.split('<EOS>')
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

#%%
# Below is the list of special characters that are directly used from the source language to the translated language

special_common = [i for i in DE_out_of_range if i in EN_out_of_range]

'''

'ū', '你', '”', '–', 'ë', 'É', 'ç', 'é', 'è', 'Ç', 'ä', '…', 'ć', '葱', 'ã', 'ï', 'í', 'ê', '£', 'à', '’', 'ó', 'ī', '€', 'á', 'ø', '“', 'Å', '♪', '♫', 'ñ', '¡', '²', 'â', '‘', '送', 'Č', 'ô', 'ā', '—', 'ü', '\xa0', 'ö'

'''

#%%
# Special characters that are only appeared in German corpus
special_DE = [i for i in DE_out_of_range if i not in special_common]
'''

['»', '¼', 'Ü', '¥', '‟', 'ý', '‚', '\ufeff', '×', '³', '\xad', '\x85', '©', '\x8a', 'š', 'ß', 'Ӓ', 'Ö', '，', '¾', '›', '\x9a', '‹', '\x9f', '½', '‒', '™', '„', '«', 'ú', 'β', '´', '°', 'к', '®', '\x96', 'œ', 'Ä']


> '×' (in DE) is translated as 'by' (in EN), meaning "times".

'''


# Special characters that are only appeared in English corpus
special_EN = [i for i in EN_out_of_range if i not in special_common]
'''

['่', '\x94', 'ย', '\x80', 'ร', 'อ', 'ē']

'''

#%%

    