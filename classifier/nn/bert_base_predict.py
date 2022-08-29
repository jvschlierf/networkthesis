
import os
import torch
from transformers import  AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from transformers import TextClassificationPipeline
from tqdm import tqdm
import argparse as arg
import sys


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = arg.ArgumentParser()
parser.add_argument('dir_path', type=str, help='directory or path file to predict')
parser.add_argument('model_dir',type=str, help='directory of model files')
parser.add_argument('output_dir', type=str, help='directory for output files')
parser.add_argument('field', type=str, help='Field of data to predict on')

model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)
model = AutoModelForSequenceClassification.from_pretrained('../../../Files/models/bert_base_cased_model/fully_trained_5/checkpoint-2158/')

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, batch_size = 25, device=1)

files = os.listdir('../../../Files/Submissions/score/')

# remove any files that are not in the .pickle type 
files = [f for f in files if f.endswith('.pickle')]

print(f"setup completed, scoring {len(files)} subreddits")

for file in tqdm(files):
    print(file)
    test = pd.read_pickle(f'../../../Files/Submissions/score/{file}')
    testlist = []
    for i,j in test.iterrows():
        testlist.append(j['cleanTitle'][0:500])
    #score each submisssion title
    results = classifier(testlist)

    for i, j in test.iterrows():
        test.at[i, 'pred_1'] = np.int64(results[i]['label'][-1])
        test.at[i, 'conf_1'] = results[i]['score']
    
    test.to_pickle(f'../../../Files/Submissions/score/done/{file}')