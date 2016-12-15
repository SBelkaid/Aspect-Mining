# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:38:03 20

This script reads hotel review corpus and creates a W2v model from that data. 
This model can be used to link aspects to the correct categories. These 
categories are extracted from a knowledge base. In this case that KB is 
WordNet. 

to do:

aspects can be phrases so think of a way to account for a phrase as
an aspect. Now every word of the same phrase is seen as its own aspect



    Objective of mining direct opinions: Given an opinionated document d,
    1. discover all opinion quintuples (oj, fjk, ooijkl, hi, tl) in d, and
    2. identify all the synonyms (Wjk) and feature indicators Ijk of each feature fjk in d.
    1. Identify object features that have been commented on. For instance, in the sentence,
    “The picture quality of this camera is amazing,” the object feature is “picture quality”.
    2. Determine whether the opinions on the features are positive, negative or neutral.
    In the above sentence, the opinion on the feature “picture quality” is positive.


@author: soufyanbelkaid
"""


import json
import nltk
import os
import gensim
import time
from nltk.stem import SnowballStemmer
from KafNafParserPy import KafNafParser
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from subprocess import Popen, PIPE
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom
from xml.etree import ElementTree
from sklearn.metrics import classification_report


data = json.loads(open('dutchies.json', 'r').read())
#DATABASE = '~/Programming/terminology_extractor/hotel_reviews.db'
DATABASE = '/Users/nadiamanarbelkaid/Aspect_mining/hotel_reviews.db'
#CMD_EXTRACTOR_SCRIPT = '~/Programming/terminology_extractor/extract_patterns.py'
CMD_EXTRACTOR_SCRIPT = '/Users/nadiamanarbelkaid/terminology_extractor/extract_patterns.py'
#PATH_ANNOTATED_DATA = '/Users/soufyanbelkaid/Research/Aspect_mining_hotels/opinion_annotations_nl-master/kaf/hotel/'
PATH_ANNOTATED_DATA = '/Users/nadiamanarbelkaid/Aspect_mining/opinion_annotations_nl-master/kaf/hotel/'
POSSIBLE_PROPERTIES = {'Bathroom',
 'Beds',
 'Breakfast',
 'Car parking',
 'Cleanliness',
 'Facilities',
 'Interior/exterior',
 'Internet',
 'Location',
 'Noisiness',
 'Reservation/check-in/check-out',
 'Restaurant',
 'Rooms',
 'Staff',
 'Swimming pool',
 'Transportation',
 'Value-for-money'}



def return_feats(list_reviews):
    """
    extract some data contained in the dictionairies
    """
    all_aspects = list()
    unique_aspects = set()
    reviews = list()
    for el_dict in list_reviews:
        raw_text = el_dict['comment']
        subjectivity = el_dict['sentiment']
        aspects = el_dict['topics']
        if aspects:
            all_aspects.append(aspects)
            unique_aspects.update(aspects)
        else:
            all_aspects.append(None)
        reviews.append(raw_text)
    print "amount of unique aspects is {}".format(len(unique_aspects))
    return all_aspects, reviews
        
      
def preprocess(files, tagged=False):
    """
    returns list of tokenized sentences, sentence splitted and words have been lowered 
    :param list_files: A list of paths to the files to open and read
    :type list_files: list(str)
    :return: list of tokenized words for every file
    :rtype: list(str)
    """
    stemmer = SnowballStemmer(language='dutch')
    tokenized = []
    tagged_normalized = []
    for f in files:
        # print f
        tokenized_sentences = nltk.sent_tokenize(f)
        container = []
        for sentence in tokenized_sentences:
            container.append(nltk.word_tokenize(sentence))
#            container.append([stemmer.stem(word) for word in nltk.word_tokenize(sentence)])            

        tokenized.extend(container)
    return tokenized


def extract_sentence(term_id, token_dict, term_dict):
    """
    extract the sentence of a word given its term id 
    :param term_id: identifier of the term 
    :param token_dict: a dictionairy of tokens and the kaf/naf info contained
    """
    sentence_number = token_dict[term_id]['sent']
    sentence_terms = list()
    for token_id in token_dict:
        sent = token_dict[token_id]['sent']
        if sent == sentence_number:
            sentence_terms.append(token_id)
    lemmas, lemmas_pos = [], []
    for t_id in sorted(sentence_terms):
        lemmas.append(term_dict[t_id]['lemma'])
        lemmas_pos.append(term_dict[t_id]['morphofeat'])
    return lemmas, lemmas_pos


def handle_phrase_props(phrase_list):
    info = phrase_list[0]
    phrase = [prop['aspect'] for prop in phrase_list]
    info['aspect'] = phrase
    info['pos'] = [prop['pos'] for prop in phrase_list]
    info['morphofeat'] = [prop['morphofeat'] for prop in phrase_list]
    info['tid'] = [prop['tid'] for prop in phrase_list]
    return info           
    
    
def handle_properties(property_elements, terms, tokens_dict):
    print "amount of properties {}".format(len(property_elements))
    category_and_terms = list()
    term_dict = dict()
    #build term info dictionairy
    for term_el in terms:
        term_node = term_el.node
        term_id = term_node.get('tid')
        term_polarity = term_node.xpath('sentiment/@polarity')
        term_info = dict(term_node.attrib)
        if term_polarity:
            #can't update list to ._Attrib
            term_info.update({'polarity':term_polarity[0]})
        term_dict[term_id] = term_info
    #property info and term info combined
    for prop_el in property_elements:
        prop_el_category = prop_el.node.xpath('@lemma')
        target_ids = prop_el.node.xpath('./references/span/target/@id')
        combine=False
        if len(target_ids) > 1:
            combine=True
            print combine
        into_one_aspect = []
#        print prop_el_category 
        for t_id in target_ids:
#            print "\t TERM ID %s is a target" % t_id
            term_final = term_dict[t_id].copy() #so the original is not changed
#            for clarity changing the key 'lemma' to 'aspect'
            term_final['aspect'] = term_final.pop('lemma')
            sent_tokens, sent_pos = extract_sentence(t_id, tokens_dict, term_dict)
            term_final.update({'sentence_tokens': sent_tokens, \
                    'sentence_pos':sent_pos})
            if not combine:
                category_and_terms.append((prop_el_category,term_final))                
            else:
#                into_one_aspect.append((term_final['aspect'], \
#                term_final['morphofeat'],term_final['pos']))
                into_one_aspect.append(term_final)
#    add phrases
        if into_one_aspect:
            phrase_term_final = handle_phrase_props(into_one_aspect)
            category_and_terms.append((prop_el_category,phrase_term_final)) 
    return category_and_terms, term_dict


def read_training_data(file_name):
    """
    read kaf/naf and matches the aspects with the words
    """
    parser = KafNafParser(PATH_ANNOTATED_DATA+file_name)
    terms = list(parser.get_terms())
#    create token dictionairy containing naf info                    
    tokens_container = dict()
    for token_el in parser.get_tokens():
        token_node = token_el.node
        token_id = token_node.get('wid').replace('w','t')
        token_info = token_node.attrib
        tokens_container[token_id] = token_info
    properties = list(parser.get_properties())
    handled_properties, term_dict = handle_properties(properties, terms, tokens_container)
    return terms, properties, handled_properties, term_dict, tokens_container


def convert_props(list_handled_props):
    """bad fix"""    
    X = zip(*list_handled_props)[1]
    for prop_info in X:
        for key, val in prop_info.items():
            if isinstance(val, list):
                prop_info[key] = ' '.join(val)
    return X
        

def get_context_numbers(term_id, term_dict):
    """
    for phrases, something else has to be devised since, maybe
    we can use the first term of the phrase to find the previous
    and use the last term of the phrase to find the next term
    
    For now if the previous token is a '.' then it will assign <BEGIN>
    """
    term_number = int(term_id[term_id.index(term_id)+1:])
    if term_number != 1:
        term_previous = 't'+str(term_number - 1)
    if term_number - 1 == 0:
        term_previous = '<BEGIN>'
    if term_number -1 != 0 and term_dict[term_previous]['lemma'] == '.': 
        term_previous = '<BEGIN>'
    next_num = 't'+str(term_number + 1)
    if term_dict.get(next_num):
        term_next = next_num
    else:
        term_next = '<END>'
#    print term_previous
#    print term_number
#    print term_next
    return term_previous, term_id, term_next


def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    if not isinstance(elem, Element):
        print "Must pass a XML Element to prettify()"        
        return
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
    
    
    
def return_mods(words_found, term_dict, path_to_db):
    """
    Ruben's terminology extractor. For now this function only works with 
    the first words found in WordNet by search_in_dwn

    :param words_found: list of term-ids, if not t-id then <END> or <BEGIN>
    :type words_found: list
    :param path_to_db: path to database containing ngrams
    :param term_dict: dictionairy with term info. Keys are tid 
    """
    top = Element('patterns')

    comment = Comment('Pattern file for terminology extractor')
    top.append(comment)
    #ALREADY SET-UP STORAGE FOR LATER USAGE 
#    container = {}
#    container[word] = defaultdict(list) #INIT DEFAULTDICT TO STORE MODIFIERS
    child = SubElement(top, 'pattern', {'len':str(len(words_found))})
#    child.len = len(words_found)

    ## CAN ADD PATTERNS HERE
    for i in range(len(words_found)):
#   only the pos tag of the aspect
        if words_found[0] == '<BEGIN>':
            print "stopping function, because beginning sentence"             
            return
        if i == 1:
            SubElement(child, 'p',{
                    "key":"pos",
                    "position": str(i),
                    "values":term_dict[words_found[i]]['pos'].lower()
                })
        context = term_dict.get(words_found[i])
        if context and i != 1 :
#           string of the context words
            SubElement(child, 'p',{
                    "key":"tokens",
                    "position": str(i),
                    "values":context['lemma'].lower()
                })
    #store pattern in memory
    pat = list(top)[1]
    pattern_tuple = tuple(child.attrib for child in pat.findall('p'))
    if pattern_tuple in patterns:
        return
    else:
        patterns.append(pattern_tuple)
#    #STORE PATTERNS FILE
    if not os.path.isdir('patterns'):
        os.mkdir('patterns')

#    logging.info("{} writing pattern file".format(time.strftime('%H:%M:%S')))
    file_name = os.path.abspath('.')+'/patterns/xml_pattern-{}.xml'.format(time.strftime('%d-%m-%y-%H:%M:%S'))
    print file_name
    with open(file_name, 'w', 0) as f: #0 is for not buffering
        f.write(prettify(top).encode('utf8'))
 ## CALL THE TERMINOLOGY EXTRACTOR WITH THE NEWLY CREATED PATTERNS
    cmd = ' '.join(['python', CMD_EXTRACTOR_SCRIPT, '-d', path_to_db, '-p', file_name])
#    logging.info(cmd)
#    print cmd
#    logging.info("{} calling terminology extractor".format(time.strftime('%H:%M:%S')))
    process = Popen(cmd, stdout=PIPE, shell=True)
    output, err = process.communicate()    
    if output:
        store_output_extractor(output)
    return top


def test_function():
    training_props = []
    for file_name in os.listdir(PATH_ANNOTATED_DATA):
        print file_name
        terms, props, handled_props, term_dict,\
            tokens_dict = read_training_data(file_name)
        training_props.extend(handled_props)

        for e in zip(*training_props)[1]:
            if isinstance(e['aspect'], str):
                try:
                    return_mods(get_context_numbers(e['tid'], term_dict), term_dict, DATABASE)    
                except KeyError:
                    print "term_id not found {}".format(e['tid'])                    
                    print e
            print "AMOUNT OF ASPECTS {}".format(len(aspects))


def store_output_extractor(raw_output):
    candidate_terms = zip(*[e.split() for e in raw_output.splitlines()])[2]
    for candidate in candidate_terms:
        aspects.append(candidate)
    
    
    
def test_fun_2():
    terms, props, handled_props, term_dict,\
        tokens_dict = read_training_data(os.listdir(PATH_ANNOTATED_DATA)[0])
    return return_mods(get_context_numbers('t11', term_dict), term_dict, DATABASE)
    

if __name__ == '__main__':
    processed_data = preprocess([d['comment'] for d in data])
    model = gensim.models.Word2Vec(processed_data)
    print "CREATING W2V MODEL"
    aspects, reviews = return_feats(data)
#    
    training_props = []
    for file_name in os.listdir(PATH_ANNOTATED_DATA):
        print file_name
        terms, props, handled_props, term_dict,\
            tokens_dict = read_training_data(file_name)
        training_props.extend(handled_props)
#    print "training SVM model"
    vectorizer = DictVectorizer()
    X = convert_props(training_props)
    X = vectorizer.fit_transform(X)
    mb = MultiLabelBinarizer()
    Y = mb.fit_transform(zip(*training_props)[0])
    X_train, X_test, y_train, y_test=train_test_split(X, Y,\
            test_size=0.1, random_state=0)
#    clf = svm.SVC()
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    clf.fit(X_train, y_train)
#    print set(clf.predict(X_test))
    clf.score(X_test, y_test)
    predicted = clf.predict(X_test)
    classification_report(y_test, predicted)


    aspects = list()
    patterns = list()
    test_function()
#
#    terms, props, handled_props, term_dict,\
#        tokens_dict = read_training_data(os.listdir(PATH_ANNOTATED_DATA)[0])