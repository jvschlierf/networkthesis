
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
parser.add_argument('ptype',type=str, help='either "Submissions" or "Comments"' )

args = parser.parse_args(sys.argv[1:])

dirs = args.dir

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
    files = os.listdir(os.path.join('../../../Files/',dirs))

    files =  [file for file in files if file.endswith('.pickle')]

    files = ('CovidVaccineInjury.pickle', 'CovidVaccinated.pickle', 'NoNewNormal.pickle','lostgeneration.pickle', 'EUnews.pickle', 'chomsky.pickle', 'conservatives.pickle',
    'AncientTruehistory.pickle', 'The_Ultimate.pickle', 'LockdownCriticalLeft.pickle', 'InternationalLeft.pickle', 'anime_titties.pickle', 'WomenInNews.pickle','ReallyAmerican.pickle',
    'NEWPOLITIC.pickle', 'NoNoNewNormal.pickle' ,'ConspiracyUltra.pickle', 'AutisticPride.pickle','Sino.pickle' ,'conspiracyNOPOL.pickle', 'Palestine.pickle' ,'EcoNewsNetwork.pickle',
    'QAnonCasualties.pickle' ,'LouderWithCrowder.pickle')
    if args.ptype == 'Comments':
        for file in tqdm(files):
            df = pd.read_pickle(os.path.join('../../../Files/',dirs,file))
            df['cleanBody'] = df['body'].apply(preprocess)
            df.to_pickle(os.path.join('../../../Files/',dirs,file))

    elif args.ptype == 'Submissions':
        for file in tqdm(files):
            df = pd.read_pickle(os.path.join('../../../Files/',dirs,file))
            df['text'] = df['title']  + ' ' + df['selftext']
            df['cleanText']=df['text'].map(lambda s:preprocess(s)) 
            df.to_pickle(os.path.join('../../../Files/',dirs,file))

    print('Done!')

