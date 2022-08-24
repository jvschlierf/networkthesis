
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy


import re
import argparse as arg
import sys


nlp = spacy.load('en_core_web_sm')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = arg.ArgumentParser()
parser.add_argument('dir', type=str, help='directory to clean')

args = parser.parse_args(sys.argv[1:])

dir = args.dir

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleantext = re.sub(r'[^\w\s]', '', sentence)
    rem_url=re.sub(r'http\S+', '[URL]',cleantext)
    rem_num = re.sub('[0-9]+', '[NUM]', rem_url)
    lemma_words = [token.lemma_ for token in nlp(rem_num) if not token.is_stop]
    return " ".join(lemma_words)


if __name__ == '__main__':
    print('Cleaning data...')
    files = os.listdir(os.path.join('../../../Files/',dir))

    files = [files.remove(file) for file in files if file.endswith('.pickle')]

    for file in tqdm(files):
        df = pd.read_pickle(os.path.join('../../../Files/',dir,file))
        df['cleanTitle']=df['title'].map(lambda s:preprocess(s)) 
        df.to_pickle(os.path.join('../../../Files/',dir,file))

    print('Done!')

