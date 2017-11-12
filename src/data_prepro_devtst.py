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



unk = '变'

special_token_dict = {
        'ū':unk,
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
        '':''
        }

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

def group_files(dev_tst='dev', de_en='de'):
    
    group = []
    
    for file in os.listdir("../data/de-en"):
        if file.endswith('xml'):
            filename_ls = file.split('.')
            if filename_ls[2].startswith(dev_tst) and filename_ls[4] == de_en:
                group.append(file)
                
    return sorted(group)
            

dev_de = group_files('dev', 'de')
dev_en = group_files('dev', 'en')
tst_de = group_files('tst', 'de')
tst_en = group_files('tst', 'en')

#%%

def parse_xml(xml_file):
    segs = []
  
    tree = ET.parse("../data/de-en/" + xml_file).getroot()
    lang = xml_file.split(".")[-2]
    tag = "srcset" if lang == "de" else "refset" 

    for doc in tree.findall(tag)[0].findall("doc"):
        seg = doc.findall("seg")
        for entry in seg:
            text = handle_special_tokens(entry.text)
            segs.append(text)
    
    return segs


def parse_xml_list(filelist):
    seg_list = []
    for f in filelist:
        segs = parse_xml(f)
        seg_list += segs
    return seg_list


dev_de_list = parse_xml_list(dev_de)
dev_en_list = parse_xml_list(dev_en)
tst_de_list = parse_xml_list(tst_de)
tst_en_list = parse_xml_list(tst_en)

print("dev_de\t{}\ndev_en\t{}\ntst_de\t{}\ntst_en\t{}".format(
    len(dev_de_list), len(dev_en_list), len(tst_de_list), len(tst_en_list)))


#%%

END = "<EOS>"
JOIN = "<JOIN>"

PATH = '../data'
dev_file = PATH + '/dev.txt'
tst_file = PATH + '/test.txt'

def join(src, tgt, save_path):
    assert len(src) == len(tgt), \
            "Lengths of source and target sentences must be the same."

    with open(save_path, "w") as f:
        for i in range(len(src)):
            content = src[i] + END + JOIN + tgt[i] + END
            f.write(content + "\n")


join(dev_de_list, dev_en_list, dev_file)
join(tst_de_list, tst_en_list, tst_file)

