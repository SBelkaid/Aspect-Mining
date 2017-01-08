# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 23:48:44 2017

@author: soufyanbelkaid
"""

import pickle
from gensim.models import Word2Vec

candidates = pickle.load(open('candidate_aspects.pickle', 'w+'))
model = Word2Vec.load('Word2Vec.model')

def match_category(aspects):
    for cand in set(zip(*zip(*aspects)[1])[1]):
        try:
            print cand, model.most_similar(cand), '\n\n'
        except KeyError:
            continue


if __name__ == '__main__':
    match_category(candidates)
    