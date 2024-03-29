
import os
import torch
from transformers import  AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
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
parser.add_argument('device', type=int, default=0, help='Device to use')


args = parser.parse_args(sys.argv[1:])

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=500, padding=True, truncation=True,  add_special_tokens=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512, padding='max_length', truncation=True, add_special_tokens = True, return_tensors='pt')
model = AutoModelForSequenceClassification.from_pretrained(os.path.join('../../../Files/models/', args.model_dir))

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, truncation=True,  max_length=500, device=args.device, batch_size=128, return_all_scores=True)

try: 
    files = os.listdir(os.path.join('../../../Files/', args.dir_path))

    # remove any files that are not in the .pickle type 
    files = [f for f in files if f.endswith('.pickle')]
except NotADirectoryError:
    files = [args.dir_path.split('/')[-1]]


print(classifier.device)
print(f"setup completed, scoring {len(files)} subreddits")


if len(files) == 1:
    test = pd.read_pickle(os.path.join('../../../Files/', args.dir_path))

    data = Dataset.from_pandas(test[[args.field]])

    
    #score
    results = classifier(data[args.field])

    for i, j in tqdm(test.iterrows(), total=len(test)):
        test.at[i, 'pred_1'] = np.int64(results[i]['label'][-1])
        test.at[i, 'conf_1'] = results[i]['score']

    test.to_pickle(os.path.join('../../../Files/', args.output_dir))

else:

    for file in tqdm(files):
        print(file)
        
        test = pd.read_pickle(os.path.join('../../../Files/', args.dir_path, file))
        data = Dataset.from_pandas(test[[args.field, 'num_comments']])

    
        #score
        print(f"scoring {file}, length: {len(test)}")
        results = classifier(data[args.field])
        

        print("scored")
        score_0 = [results[i][0]['score'] for i in range(len(results))]
        score_1 = [results[i][1]['score'] for i in range(len(results))]
        score_2 = [results[i][2]['score'] for i in range(len(results))]

        test['label_0'] = score_0
        test['label_1'] = score_1
        test['label_2'] = score_2
         
        test.to_pickle(os.path.join('../../../Files/', args.output_dir, ("d_"+file)))
        print(f"saved {file}")
