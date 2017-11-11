# -*- coding: utf-8 -*-

'''
According to this paper: http://www.aclweb.org/anthology/I/I05/I05-2014.pdf
A Bleu_4 Score in Word Level corresponds to Bleu_18 Score in Character Level.
Hence, in this project, we are using a Character Level Bleus of values: 5, 10, 15, 20.

Ref: 
https://github.com/vikasnar/Bleu
https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/evaluate.py
'''

import os
import math
import operator
from functools import reduce


def fetch_data(cand, ref):
    """
    Store each reference and candidate sentences as a list
    """
    references = []
    if '.txt' in ref:
        reference_file = open(ref, 'r', encoding='utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                if not f.startswith('.'):
                    reference_file = open(os.path.join(root, f), 
                                          'r', encoding='utf-8')
                    references.append(reference_file.readlines())
    candidate_file = open(cand, 'r', encoding='utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n):
    """
    Character Level n-gram counter
    
    @return 
        precision and brevity penalty
    """
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    
    for s_i in range(len(candidate)):
        
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[s_i]
            ngram_d = {}
            chars = [i for i in ref_sentence.strip()]
            ref_lengths.append(len(chars))
            limits = len(chars) - n + 1
            
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(chars[i:i+n])
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
            
        # candidate
        cand_sentence = candidate[s_i]
        cand_dict = {}
        chars = [i for i in cand_sentence.strip()]
        limits = len(chars) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(chars[i:i + n])
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
                
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(chars))
        c += len(chars)
        
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
        
    bp = brevity_penalty(c, r)
    
    return pr, bp


def clip_count(cand_d, ref_ds):
    """
    Count the clip count for each ngram considering all references
    """
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """
    Find the closest length of reference to that of candidate
    """
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(candidate, references):
    precisions = []
    for i in range(5, 21, 5):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu



#%%
if __name__ == "__main__":
    candidate, references = fetch_data("../data/eval/candidate.txt", "../data/eval/references")
    bleu = BLEU(candidate, references)
    print(bleu)
    out = open('bleu_out.txt', 'w')
    out.write(str(bleu))
    out.close()
