# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import random
import collections
import torch
import numpy as np


#%%
    
path = './de-en/'



def load_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            
            # TODO
            # Strip out the parse information and the phrase labels--- ELiminate the <x> </x> symbols  
            
#            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            text = line.strip()
            example['text'] = text[:]
            data.append(example)

    random.seed(1)
    random.shuffle(data)
    return data

#%%
train_de = load_data(path + 'train.tags.de-en.de')
train_en = load_data(path + 'train.tags.de-en.en')

#%%

max_seq_length = 20

def sentence_to_padded_index_sequence(datasets):
    '''
    Annotates datasets with feature vectors.
    
    @param
    datasets: list of datasets
    
    '''
    
    START = "<S>"
    END = "<EOS>"
    END_PADDING = "<PAD>"
    UNKNOWN = "<UNK>"
    SEQ_LEN = max_seq_length + 1
    
    # Extract vocabulary
    def tokenize(string):
        return string.lower().split()
    
    word_counter = collections.Counter()
    for example in datasets[0]:
        word_counter.update(tokenize(example['text']))
    
    # Only enter word into vocabulary if it appears > 25 times. Add special symbols to vocabulary.
    vocabulary = set([word for word in word_counter if word_counter[word] > 0]) 
    vocabulary = list(vocabulary)
    vocabulary = [START, END, END_PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}
        
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['index_sequence'] = torch.zeros((SEQ_LEN))
            
            token_sequence = [START] + tokenize(example['text']) + [END]

            for i in range(SEQ_LEN):
                if i < len(token_sequence):
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                else:
                    index = word_indices[END_PADDING]
                example['index_sequence'][i] = index
                
            example['target_sequence'] = example["index_sequence"][1:]
            example['index_sequence'] = example['index_sequence'].long().view(1,-1)
            example['target_sequence'] = example['target_sequence'].long().view(1,-1)
            
    return indices_to_words, word_indices


#%%
id2w_en, w2id_en = sentence_to_padded_index_sequence([train_en])
id2w_de, w2id_de = sentence_to_padded_index_sequence([train_de])

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

#%%
    
import nltk

train_en_tokens_len = []

for tt in train_en:
    
    sentence = tt['text']
    tokens = nltk.word_tokenize(sentence)
    length = len(tokens)
    
    train_en_tokens_len.append(length)
    
    
print('The longest sentence in the training english set is', max(train_en_tokens_len))


#%%    

def text2idx(dataset, w_id):
    
    lines = []
    
    for tt in dataset:
        
        line = []
        
        sentence = tt['text']
#        tokens = nltk.word_tokenize(sentence)
        tokens = sentence.split(' ')
        
        for word in tokens:
            
            idx = w_id[word.lower()]
            line.append(idx)
            
#        lines.append(' '.join(lines) )
        lines.append(line)
        
    return lines
        
        
trn_en = text2idx(train_en[:5], w2id_en)


#%%
            
            



















