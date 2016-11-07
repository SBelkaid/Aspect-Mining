import sys
import nltk
import re
import gensim
from retrogade import load_data


dutch_tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')

usage = """
python create_w2v.py path_to_files

path_to_files: path to the dir containing all the seperate files"""

if not len(sys.argv) == 2:

	sys.exit(1)

file_dir = sys.argv[1]

def preprocess(files, tagged=False):
    """
    returns list of tokenized sentences, sentence splitted and words have been lowered 
    :param list_files: A list of paths to the files to open and read
    :type list_files: list(str)
    :return: list of tokenized words for every file
    :rtype: list(str)
    """
    tokenized = []
    tagged_normalized = []
    for f in files:
        print f
        tokenized_sentences = nltk.sent_tokenize(open(f, 'r').read().decode('utf8'))
        container = []
        for sentence in tokenized_sentences:
            container.append(nltk.word_tokenize(sentence))
            # sentence_hack = []
            # for word in sentence:
                # sentence_hack.extend(re.split(r'[\./]', word.lower())) #stupid hack for erronous words such 
    #             #as cyste.classificatie reeel/compositie
            # container.append(sentence_hack)
        tokenized.extend(container)
        # tokenized.extend([map(lambda x: x , [PATTERN.split(word.lower()) for word in sentence]) for sentence in tokenized_sentences])
    return tokenized


if __name__ == '__main__':
    files, folders, paths = load_data(file_dir, '*')
    processed_data = preprocess(files)
    model = gensim.models.Word2Vec(processed_data)



