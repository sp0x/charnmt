# -*- coding: utf-8 -*-
'''
According to this paper: http://www.aclweb.org/anthology/I/I05/I05-2014.pdf
A Bleu_4 Score in Word Level corresponds to Bleu_18 Score in Character Level.
Hence, in this project, we are using a Character Level Bleus of values: 5, 10, 15, 20.
'''

from nltk.translate.bleu_score import sentence_bleu

def BLEU(candidate, reference, n=4):
    """
    ----------
    @params
        candidate: string - predicted sequence of translation
        reference: string - reference sentence (Ground True)
        n: integer - n for n-gram
    ----------
    @return
        BLEU score - float
    """

    ref = [reference.strip().split()]
    can = candidate.strip().split()

    if len(candidate) == 0:

        bleu = 0

    else:

        if n == 1:
            bleu = sentence_bleu(ref, can, weights=(1, 0, 0, 0))
        elif n == 2:
            bleu = sentence_bleu(ref, can, weights=(.5, .5, 0, 0))
        elif n == 3:
            bleu = sentence_bleu(ref, can, weights=(.33, .33, .33, 0))
        elif n ==4:
            bleu = sentence_bleu(ref, can, weights=(.25, .25, .25, .25))
        else:
            print('Please Define n in BLEU function with integers from 1 to 4.')

    return bleu


if __name__ == "__main__":
    candidate = 'It is a guide to action which ensures that the military always obeys the commands of the party.'
    reference = 'It is a guide to action that ensures that the military will forever heed Party commands.'
    bleu = BLEU(candidate, reference, 2)
    print(bleu)
