# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 23:48:44 2017

@author: soufyanbelkaid
"""

import pickle
from gensim.models import Word2Vec
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import os


candidates = pickle.load(open('candidate_aspects.pickle','r'))
model = Word2Vec.load('Word2Vec.model')
POSSIBLE_PROPERTIES = {'Bathroom': {'nl_word': ['badkamer'], 'syns': []},
 'Beds': {'nl_word': ['bed', 'bedden'], 'syns': []},
 'Breakfast': {'nl_word': ['ontbijt'], 'syns': []},
 'Car parking': {'nl_word': ['parkeren'], 'syns': []},
 'Cleanliness': {'nl_word': ['hygi\xc3\xabne', 'hygiene', 'schoon'],
  'syns': []},
 'Facilities': {'nl_word': ['faciliteiten'], 'syns': []},
 'Interior/exterior': {'nl_word': ['interieur'], 'syns': []},
 'Internet': {'nl_word': ['internet'], 'syns': []},
 'Location': {'nl_word': ['locatie'], 'syns': []},
 'Noisiness': {'nl_word': ['geluid'], 'syns': []},
 'Reservation/check-in/check-out': {'nl_word': ['inchecken', 'uitchecken'],
  'syns': []},
 'Restaurant': {'nl_word': ['restaurant'], 'syns': []},
 'Rooms': {'nl_word': ['kamer'], 'syns': []},
 'Staff': {'nl_word': ['personeel'], 'syns': []},
 'Swimming pool': {'nl_word': ['zwembad'], 'syns': []},
 'Transportation': {'nl_word': ['vervoer'], 'syns': []},
 'Value-for-money': {'nl_word': ['waarde', 'geld'], 'syns': []}}


def construct_sub_graph(word_vec_sim, seed):
    sub_G = nx.DiGraph()
    sub_G.add_node(seed, color='blue')
    for sim_word, sim_val in word_vec_sim:
        try:
            sim_word = sim_word.decode('utf8')
            sub_G.add_node(sim_word)
            sub_G.add_weighted_edges_from([(seed,sim_word,"%.4f" % sim_val)])
        except (UnicodeDecodeError, UnicodeEncodeError, IndexError), e:
            print str(e)
    sub_G.remove_edges_from(sub_G.selfloop_edges())
    return sub_G


def create_taxo(aspect_dictionary):
    """
    This function is called create_taxo because it collects all the words
    that are similar to the nl_word in the dictionairy given as a parameter.
    The similar words are extracted from the Word2Vec model
    """
    taxo = nx.DiGraph()
    for key, val in aspect_dictionary.items():
        try:
            seed = aspect_dictionary[key]['nl_word'][0]
            sim_words = model.most_similar(seed.encode('utf8'))
            sub_graph = construct_sub_graph(sim_words, seed)    
            taxo.add_nodes_from(sub_graph)
            taxo.add_edges_from(sub_graph.edges())
        except (UnicodeDecodeError, UnicodeEncodeError, IndexError), e:
            print str(e)
            continue
    return taxo


if __name__ == '__main__':
    taxonomy = create_taxo(POSSIBLE_PROPERTIES)
    write_dot(taxonomy, 'neato_taxo.dot')
    os.system('neato -Tpng neato_taxo.dot > neato_taxo.png')
    os.system('open neato_taxo.png')
#    nx.draw(taxonomy, pos)

    