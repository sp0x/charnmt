# -*- coding: utf-8 -*-
"""
@Author: Viola

This module pre-processes the datasets before passed into the model.
It maps the de-en sentence pairs with tokenized words to sequences.

# Below is the list of special characters that are directly used from the source language to the translated language

# special_common = [i for i in DE_out_of_range if i in EN_out_of_range]
---
'ū', '你', '”', '–', 'ë', 'É', 'ç', 'é', 'è', 'Ç', 'ä', '…', 'ć', '葱', 'ã', 'ï', 'í', 'ê', '£', 'à', '’', 'ó', 'ī', '€', 'á', 'ø', '“', 'Å', '♪', '♫', 'ñ', '¡', '²', 'â', '‘', '送', 'Č', 'ô', 'ā', '—', 'ü', '\xa0', 'ö'

# Special characters that are only appeared in German corpus
# special_DE = [i for i in DE_out_of_range if i not in special_common]
---
'»', '¼', 'Ü', '¥', '‟', 'ý', '‚', '\ufeff', '×', '³', '\xad', '\x85', '©', '\x8a', 'š', 'ß', 'Ӓ', 'Ö', '，', '¾', '›', '\x9a', '‹', '\x9f', '½', '‒', '™', '„', '«', 'ú', 'β', '´', '°', 'к', '®', '\x96', 'œ', 'Ä']

> eg. '×' (in DE) is translated as 'by' (in EN), meaning "times".

# Special characters that are only appeared in English corpus
# special_EN = [i for i in EN_out_of_range if i not in special_common]
---
['่', '\x94', 'ย', '\x80', 'ร', 'อ', 'ē']

"""

import random
import os
import xml.etree.ElementTree as ET


#%%

def group_files(dev_tst='dev', de_en='de'):
    
    group = []
    
    for file in os.listdir("../data/de-en"):
        if file.endswith('xml'):
            filename_ls = file.split('.')
            if filename_ls[2].startswith(dev_tst) and filename_ls[4] == de_en:
                group.append(file)
                
    return group
            

dev_de = group_files('dev', 'de')
dev_en = group_files('dev', 'en')
tst_de = group_files('tst', 'de')
tst_en = group_files('tst', 'en')

#%%

for xml_file in dev_de:
    
    segs = []
  
    tree = ET.parse("../data/de-en/" + xml_file).getroot()

    for doc in tree.findall("refset")[0].findall("doc"):
        seg = doc.findall("seg")
        for entry in seg:
            segs.append(entry.text.strip())
            print(entry.text.strip())

#%%
file_path = "../data/de-en/IWSLT16.TED.dev2010.de-en.en.xml"

tree = ET.parse(file_path).getroot()

n = 0 

segs = []
for doc in tree.findall("refset")[0].findall("doc"):
    seg = doc.findall("seg")
    for entry in seg:
        if n > 2000:
            break
        segs.append(entry.text.strip())
        print(entry.text.strip())
        n += 1







#%%
unk = '变'

special_token_dict = {'ū':unk,
             '你':unk,
             '“':'"',
             '–':'-',
             'ë':'e',
             'É':'E',
             'ç':'c',
             'é':'e',
             'è':'e',
             'Ç':'C',
             'ä':'a',
             '…':'...',
             'ć':'c',
             '葱':unk,
             'ã':'a',
             'ï':'i',
             'í':'i',
             'ê':'e',  
            'à':'a',
             '’':"'",
             'ó':'o',
             'ī':'i',
             'á':'a',
             'ø':'o',
             '“':'"',
             'Å':'A',
             'ñ':'n',
             '¡':'',
             'â':'a',
             '’':"'",
             '送':unk,
             'Č':'C',
             'ô':'o',
             'ā':'a',
             '—':'-',
             '\xa0':' ',
             '»':'',
             '‟':'"',
             'ý':'y',
             '‚':"'",
             '\ufeff':' ',
             '\xad':' ',
             '\x85':' ',
             '©':'',
             '\x8a':' ',
             'š':'s',
             '，':',',
             '›':'',
             '\x9a':' ',
             '‹':'',
             '\x9f':' ',
             '‒':'-',
             '™':'',
             '„':'"',
             '«':'',
             'ú':'u',
             'β':'ß',
             '´':"'",
             'к':'k',
             '®':'',
             '\x96':' ',
             'œ':'oe',
             '\x94':' ',
             'ย':'',
             '\x80':' ',
             'ร':'',
             'อ':'',
             'ē':'e',
                         '่':''}

#%%
def handle_special_tokens(input):
    
    output = ''
    prev_char = ''
    
    for char in input:
        
        if char in special_token_dict:
            c = special_token_dict[char]
    
        else:
            c = char
            
        if prev_char != ' ' or c != ' ':
            output += c
        
        if c != '':
            prev_char = c
            
    return output.strip()



#%%
    
PATH = '../data/de-en/'

END = "<EOS>"
END_PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def load_data(path):
    data = []
    with open(path, 'r',  encoding="utf8") as f:
        for i, line in enumerate(f): 
            example = {}

            if line[0] == '<' and line[-1] == '>':
                continue
            
            line = handle_special_tokens(line)
            
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
        
        corpus.append("<JOIN>".join([s_from, s_to]))

    return " \n".join(corpus)


corpus_from_to = build_corpus(train_de, train_en)

with open("../data/train.txt", "w") as f:
    f.write(corpus_from_to)
    
