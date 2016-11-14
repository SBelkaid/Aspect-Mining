"""
This script creates a graph from a seed word. It starts bottom and find all 
paths to the top-most node in Open Dutch WordNet.
Any structured knowledge base can be used, but for no ODWN is used. for more 
information visit http://wordpress.let.vupr.nl/odwn/
make sure to install ODWN and add it to you PYTHONPATH
https://github.com/cltl/OpenDutchWordnet
fro mac install graphviz using brew otherwise pygraphviz will not work
This script also contain code to perform KMeans on a matrix.

"""
from OpenDutchWordnet import Wn_grid_parser
from gensim.models import Word2Vec
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque
from sklearn.cluster import KMeans
import networkx as nx
import sys


dwn = Wn_grid_parser(Wn_grid_parser.odwn)
dwn.load_synonyms_dicts()
model = Word2Vec.load('w2v_6-10-2016.model')

ASPECTS = {'TREATMENT':['effectief', 'behandelen', 'pijnloos', 'niks_voelen',
    'deskundig', 'adequaat', 'z\'n_vak_verstaan'],
    'APPOINTMENT':['wachtrij', 'lang wachten', 'eindeloos', 'in_de_wacht_staan'],
    'TELEPHONE_ACCESS':['telefoon', 'telefoneren', 'bellen'],
    'DIGNITY_AND_RESPECT':['vriendelijk', 'arrogant', 'rustig', 'aardig', 'informeel',
    'vertrouwen_geven'],
    'PAY_ATTENTION':['luisteren','aandacht', 'meeleven', 'begrijpen', 'gesprek',
    'tijd', 'tijd_geven', 'vertrouwen', 'aandachtig'],
    'GENERAL':['goed', 'slecht', 'niet_normaal'],
    'CLEANLINESS':['schoon', 'vies', 'rommelig'],
    'LOCATION':['goed_bereikbaar', 'bereikbaar', 'onbereikbaar', 'ver_weg'],
    'BUILDING':['wachtruimte', 'spreekkamer'],
    'VALUE_FOR_MONEY':['goedkoop', 'duur'],
    'PROVIDING_INFORMATION':['informatie', 'toelichten', 'duidelijk', 'duidelijkheid',
    'helder', 'helderheid', 'advies']}


def return_direct_hypo(word):
    synsets_word = dwn.lemma2synsets[word]
    print "searching for hypernyms of the word {}".format(word)
    higher_synsets = []
    if synsets_word:
        for ss in synsets_word:
            print "All the hyponyms of {}".format(ss)
            ss_obj = dwn.synsets_find_synset(ss)
            relations = ss_obj.get_relations('has_hyperonym')
            for rel_obj in relations:
                hypo_obj = dwn.synsets_find_synset(rel_obj.get_target())
                hypo_obj_id = hypo_obj.get_id()
                higher_synsets.append(hypo_obj_id)
                print "hypernym synset id", hypo_obj_id,\
                "lemmas", dwn.synset2lemmas[hypo_obj_id]
    return sorted(filter(lambda x:x[1],\
    [(x, x.split('-')[-2]) for x in higher_synsets]))


def swc(seed_synset_id):
    """
    return the path to the top most node from a given synset.
    :param seed_synset_id: synset_id
    :type seedsynset_id: str
    :return: path to the top most node
    :rtype: deque(str)
    """
    pool = deque()
    pool.append(seed_synset_id)
    path = deque()
    G=nx.DiGraph()

    while pool:
        ss_id = pool.popleft()
        synset = dwn.synsets_find_synset(ss_id)
        hypo_rels = synset.get_relations('has_hyperonym')
        hypo_ids = [rel.get_target() for rel in hypo_rels]
        hypo_ids.append(ss_id)
        G.add_nodes_from(hypo_ids)
        [G.add_edge(hypernym, ss_id) for hypernym in hypo_ids]            
        print ss_id, 'has {} hypernyms'.format(len(hypo_rels))
        for hypo in hypo_rels:
            hypo_id = hypo.get_target()
            if hypo_id in path:
                print "already in path", hypo_id
                continue
            if hypo_id in pool:
                continue   
            pool.append(hypo_id)
        path.append(ss_id)
    G.remove_edges_from(G.selfloop_edges())
    return path, G


def do_kmeans(word_vectors, divide_by_x):
    'this is vector quantification'
    num_clusters = word_vectors.shape[0]/divide_by_x
    #starting kmeans, could take long
    print "amount of clusters is {}".format(num_clusters)
    kmeans = KMeans(n_clusters=num_clusters)
    idx = kmeans.fit_predict(word_vectors)
    return idx


def print_kmeans_output(model, idx):
    word_centroid_map = dict(zip( model.index2word, idx))
    # For the first 10 clusters
    for cluster in xrange(0,10):
        # Print the cluster number  
        print "\nCluster %d" % cluster
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
        print words


def create_labels(synsets):
    labels = {}
    for syn in synsets:
        lex = dwn.synset2lemmas[syn].copy() #copy so original stays intact
        try:
            lex = lex.pop() #first in set
            labels[syn] = lex
        except KeyError:
            continue
    return labels


if __name__ == "__main__":
#    top_hypo = return_direct_hypo('behulpzaam')
#    print dwn.synset2lemmas[top_hypo[0][0]]
    if dwn.lemma2synsets['meegaand']:
        synset = dwn.lemma2synsets['meegaand'].copy()
        p,g = swc(synset.pop())
    else:
        sys.exit()
    labels = create_labels(g.nodes())
    pos = graphviz_layout(g, prog='dot')
    nx.draw(g, pos, labels=labels)
#    nx.nx_pydot.write_dot(g, 'graph.dot')
    
    
# KMEANS CLUSTERING STARTS HERE ----------------------------------
#    idx = do_kmeans(model.syn0, 5)
#    np.save(open('kmeans_idx_5', 'w'), idx)
#    nx.drawing.nx_agraph.write_dot(g, 'chaos.dot')
#    print "run the following 'dot -Tpng chaos.dot > chaos.png'"
#    print_kmeans_output(model, np.load('kmeans_indexes.npy'))
    

    
