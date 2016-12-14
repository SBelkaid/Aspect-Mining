# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import sys
import pickle


#usage = """
#run script:
#python create_model.py <path_to_json>
#"""
#
#if not len(sys.argv) > 2:
#    print usage
#    sys.exit(1)
#f = sys.argv[1]
f = 'DATA/reviews.json'

splitted = open(f, 'r').read().split('\n')
list_of_dictionairies = list()
for el in splitted:
    try:
        list_of_dictionairies.append(json.loads(el))
    except ValueError:
        continue

dutch_reviews = [d for d in list_of_dictionairies if d['language']=='dutch']
pickle.dump(dutch_reviews, open('nl_reviews_list_dict.pickle', 'w+'))

    