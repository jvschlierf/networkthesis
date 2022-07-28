"""
File to control text classification models as part of Analysis of Pro- & Anti- Vaccine behavior on reddit
"""
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
import numpy as np
from datasets import Dataset
import pandas as pd

df = pd.read_pickle('../../../Files/Submissions/train/submission_train_sm.pickle')

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = DistilBertTokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

#, warmup = 600, max_sequence_length=128, training_rate=1e-5,

df['text'] = df['title']
df['labels'] = df['label']
df = df[['text', 'labels']]

dataset = Dataset.from_pandas(df)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
dataset_splitted = tokenized_dataset.shuffle(1337).train_test_split(0.1)




from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

for name, param in model.named_parameters():
    if name in ['classifier.weight', 'classifier.bias']:
        param.requires_grad = True
    else:
        param.requires_grad = False




training_args = TrainingArguments(
    load_best_model_at_end=True,
    output_dir = '../../Files/models/distilbert_model/',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32, 
    evaluation_strategy='epoch',
    logging_dir='../../Files/logs/', 
    save_strategy = "epoch",
    save_steps=10_000, save_total_limit=2, )



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_splitted['train'],
    eval_dataset=dataset_splitted['test'],
    compute_metrics=compute_metrics,
)


trainer.train()

