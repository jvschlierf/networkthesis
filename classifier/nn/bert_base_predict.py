
import os
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from transformers import TextClassificationPipeline

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

test = pd.read_pickle('../../../Files/Submissions/train/test_split_submission.pickle')
test['text'] = test['cleanTitle']
test = test[['text']]

test = test.values.tolist()

model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding="max_length", truncation=True)
model = AutoModelForSequenceClassification.from_pretrained('../../../Files/models/bert_base_cased_model/fully_trained/checkpoint-3237/')
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

from tqdm import tqdm
predictions = []
for i in tqdm(range(len(test))):
    predictions.append(classifier(test[i]))


df = pd.DataFrame(predictions)
df.to_pickle('../../../Files/models/bert_base_cased_predictions.pickle')