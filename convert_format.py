import sys
import pandas as pd
import os
from tempfile import NamedTemporaryFile
from HTMLParser import HTMLParser

usage = """
python convert_format.py csv_file
"""
if len(sys.argv) != 2:
    print usage
    sys.exit(1)

columns = ['"id',
	 'date',
	 'profession',
	 'city',
	 'doctor',
	 'disease',
	 'specialty',
	 'review_text',
	 'overall_rating',
	 'appointment_rating',
	 'therapy_rating',
	 'staff_attention_rating',
	 'information_rating',
	 'listen_rating',
	 'accommodation_rating',
	 'polarity>5',
	 'polarity=>5',
	 'polarity>7',
	 'polarity>=7"']

file_name = sys.argv[1]
html_parser = HTMLParser()
data = [e.split('\t') for e in open(file_name, 'r').read().split('\r')]
for e2 in data:
	e2[7] = html_parser.unescape(unicode(e2[7], errors='replace'))
	e2[-1] = e2[-1].strip(',')

df = pd.DataFrame(data[1:], columns=columns)
reviews = df['review_text'].values
if not os.path.exists('DATA/seperate_files'):
	os.mkdir('DATA/seperate_files')
for rev in reviews:
	with NamedTemporaryFile(suffix='.txt', delete=False, dir='DATA/seperate_files') as f:
		f.write(rev.encode('utf8'))