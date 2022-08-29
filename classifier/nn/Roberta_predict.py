
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


args = parser.parse_args(sys.argv[1:])


model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding='max_length', truncation=True, max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(os.path.join('../../../Files/models/', args.model_dir))

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=1)

try: 
    files = os.listdir(os.path.join('../../../Files/', args.dir_path))

    # remove any files that are not in the .pickle type 
    files = [f for f in files if f.endswith('.pickle')]
except NotADirectoryError:
    files = [args.dir_path.split('/')[-1]]

print(f"setup completed, scoring {len(files)} subreddits")


if len(files) == 1:
    test = pd.read_pickle(os.path.join('../../../Files/', args.dir_path))
    testlist = []
    for i,j in test.iterrows():
        testlist.append(j[args.field][0:500])
    #score each submisssion title
    results = classifier(testlist)

    for i, j in test.iterrows():
        test.at[i, 'pred_1'] = np.int64(results[i]['label'][-1])
        test.at[i, 'conf_1'] = results[i]['score']
    
    test.to_pickle(os.path.join('../../../Files/', args.output_dir))

else:

    for file in tqdm(files):
        print(file)
        
        test = pd.read_pickle(os.path.join('../../../Files/', args.dir_path, file))
        testlist = []
        for i,j in test.iterrows():
            testlist.append(j[args.field][0:500])
        #score each submisssion title
        results = classifier(testlist)

        for i, j in test.iterrows():
            test.at[i, 'pred_1'] = np.int64(results[i]['label'][-1])
            test.at[i, 'conf_1'] = results[i]['score']
        
        test.to_pickle(os.path.join('../../../Files/', args.output_dir, file))