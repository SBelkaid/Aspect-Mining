# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:38:03 20

This script reads hotel review corpus and creates a W2v model from that data. 
This model can be used to link aspects to the correct categories. These 
categories are extracted from a knowledge base.

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
DATABASE = '~/Programming/terminology_extractor/hotel_reviews_v2.db'
#DATABASE = '/Users/nadiamanarbelkaid/Aspect_mining/hotel_reviews_v2.db'
CMD_EXTRACTOR_SCRIPT = '~/Research/terminology_extractor/extract_patterns.py'
#CMD_EXTRACTOR_SCRIPT = '/Users/nadiamanarbelkaid/terminology_extractor/extract_patterns.py'
PATH_ANNOTATED_DATA = '/Users/soufyanbelkaid/Research/Aspect-Mining/opinion_annotations_nl-master/kaf/hotel/'
#PATH_ANNOTATED_DATA = '/Users/nadiamanarbelkaid/Aspect_mining/opinion_annotations_nl-master/kaf/hotel/'
POSSIBLE_PROPERTIES = {'Bathroom':['badkamer'],
 'Beds':['bed', 'bedden'],
 'Breakfast':['ontbijt'],
 'Car parking':['parkeren'],
 'Cleanliness':['hygiëne','schoon'],
 'Facilities':['faciliteiten'],
 'Interior/exterior':['interieur'],
 'Internet':['internet'],
 'Location':['locatie'],
 'Noisiness':['geluid'],
 'Reservation/check-in/check-out':['inchecken', 'uitchecken'],
 'Restaurant':['restaurant'],
 'Rooms':['kamer'],
 'Staff':['personeel'],
 'Swimming pool':['zwembad'],
 'Transportation':['vervoer'],
 'Value-for-money':['waarde','geld']}


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
        into_one_aspect = []
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


def create_pattern_file(words_found, term_dict, top_element):
    child = SubElement(top_element, 'pattern', {'len':str(len(words_found))})
    ## CAN ADD PATTERNS HERE
    for i in range(len(words_found)):
#   only the pos tag of the aspect
        if words_found[0] == '<BEGIN>':
            #no contect words, simple pattern can't be extracted
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
    print "added patterns to top_element"
    

def run_extractor(pattern_file, path_to_db):
    if not os.path.isdir('patterns'):
        os.mkdir('patterns')

#    logging.info("{} writing pattern file".format(time.strftime('%H:%M:%S')))
    file_name = os.path.abspath('.')+'/patterns/xml_pattern-{}.xml'.format(time.strftime('%d-%m-%y-%H:%M:%S'))
#    print file_name
    with open(file_name, 'w', 0) as f: #0 is for not buffering
        f.write(prettify(pattern_file).encode('utf8'))
 ## CALL THE TERMINOLOGY EXTRACTOR WITH THE NEWLY CREATED PATTERNS
    cmd = ' '.join(['python', CMD_EXTRACTOR_SCRIPT, '-d', path_to_db, '-p', file_name])
#    logging.info(cmd)
#    print cmd
#    logging.info("{} calling terminology extractor".format(time.strftime('%H:%M:%S')))
    process = Popen(cmd, stdout=PIPE, shell=True)
    output, err = process.communicate()    
    if output:
        store_output_extractor(output)
   

def start(amount_files=10):
    top = Element('patterns') #this will be the main pattern file.
    comment = Comment('Pattern file for terminology extractor')
    top.append(comment)
    training_props = []
    count = 0
    for file_name in os.listdir(PATH_ANNOTATED_DATA):
        if count == amount_files:
            break
        print file_name
        terms, props, handled_props, term_dict,\
            tokens_dict = read_training_data(file_name)
        training_props.extend(handled_props)

        for e in zip(*training_props)[1]:
            if isinstance(e['aspect'], str):
                try:
                    create_pattern_file(get_context_numbers(e['tid'], term_dict), term_dict, top)
                except KeyError:
                    print "term_id not found {} in file {}".format(e['tid'], file_name)                    
                    print e
            print "AMOUNT OF ASPECTS {}".format(len(aspects))
        count+=1
    return top


def store_output_extractor(raw_output):
    try:
        candidates = json.loads(raw_output)
        for key, val in candidates.items():
            #index 1, only interested in middle
            aspects.append((key, zip(val[0][0].split(), val[0][1].split())[1]))
    except ValueError, e:
        print "check terminology extractor output"
        raise e


def start_classification(training_props):
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
    clf.score(X_test, y_test)
    predicted = clf.predict(X_test)
    print classification_report(y_test, predicted)
        

if __name__ == '__main__':
    processed_data = preprocess([d['comment'] for d in data])
    print "CREATING W2V MODEL"
    model = gensim.models.Word2Vec(processed_data)
    print "DONE"
    print "SEARCHING FOR ASPECTS"
    aspects = list()
    patterns = list()
    pattern_file = start()
    run_extractor(pattern_file, DATABASE)
#
#    terms, props, handled_props, term_dict,\
#        tokens_dict = read_training_data(os.listdir(PATH_ANNOTATED_DATA)[0])